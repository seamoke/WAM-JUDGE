"""Full-model LingBot-VA joint video/action RFT on real and pseudo data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler

from robotwin_critic.two_stage_rft.action_only_dataset import (
    AllTransitionChunkDataset,
    DeterministicFractionDataset,
    FirstTransitionChunkDataset,
    GeneratedChunkDataset,
    RatioMixedDataset,
    UnionRFTDataset,
    mixed_pad_latent_batch_collate,
)
from robotwin_critic.two_stage_rft.log_online_collection_swanlab import (
    numeric_metrics,
)
from robotwin_critic.two_stage_rft.protocol import sha256_file
from robotwin_critic.two_stage_rft.pseudo_preflight import (
    build_pseudo_preflight_report,
    flatten_preflight_summary,
    unexpected_pseudo_preflight_failure_report,
    write_preflight_report,
)
from robotwin_critic.two_stage_rft.production_auxiliary_step import (
    initialize_production_auxiliary_state,
    production_auxiliary_train_step,
)
from robotwin_critic.two_stage_rft.model_completeness import (
    reject_existing_snapshot_targets,
    require_complete_transformer,
    require_snapshot_invocation,
    write_snapshot_marker,
)
from robotwin_critic.two_stage_rft.rft_schedule import (
    SCHEDULE_MODES,
    auxiliary_schedule_report,
    auxiliary_update_plan,
    effective_pseudo_loss_weight,
    expected_sample_exposure,
    next_sampler_batch,
    resolve_optimizer_steps,
    resolve_base_auxiliary_steps,
    should_rebuild_lr_scheduler,
    validate_base_auxiliary_contract,
)
from robotwin_critic.two_stage_rft.wam_shape_preflight import (
    checkpoint_attention_report,
    loaded_model_attention_report,
    require_matching_rank_reports,
)


def rebuild_lr_scheduler(optimizer, config, train_module):
    """Recreate a total-step-dependent official scheduler when required."""
    scheduler_type = getattr(config, "lr_scheduler_type", "constant").strip().lower()
    if scheduler_type == "cosine":
        lr_lambda = lambda step: train_module.warmup_cosine_lambda(
            step,
            warmup_steps=config.warmup_steps,
            total_steps=config.num_steps,
            min_lr_ratio=getattr(config, "min_lr_ratio", 0.0),
        )
    else:
        raise ValueError(f"Unsupported lr_scheduler_type={scheduler_type!r}")
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


class ExternalSwanLabSink:
    """Emit metrics to the long-lived online RFT parent process."""

    @staticmethod
    def log(data, step=None, *args, **kwargs) -> None:
        del args, kwargs

        def convert(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().item()
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            return float(value)

        payload = {
            "step": None if step is None else int(step),
            "metrics": {str(key): convert(value) for key, value in data.items()},
        }
        print("SWANLAB_METRIC_EVENT " + json.dumps(payload), flush=True)

    @staticmethod
    def finish(*args, **kwargs) -> None:
        del args, kwargs


class ExternalSwanLabRun:
    url = "parent://online-rft-swanlab"


def enable_full_finetune(model) -> dict:
    """Enable every model parameter, matching the official WAM trainer."""
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    trainable = sum(parameter.numel() for parameter in model.parameters())
    frozen = sum(
        parameter.numel()
        for parameter in model.parameters()
        if not parameter.requires_grad
    )
    if trainable <= 0 or frozen:
        raise RuntimeError(
            f"Full RFT parameter contract failed: trainable={trainable}, frozen={frozen}"
        )
    return {
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
    }


def verify_full_optimizer(model, optimizer) -> dict:
    model_parameters = {
        id(parameter): parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    optimizer_parameters = {
        id(parameter): parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    missing = set(model_parameters) - set(optimizer_parameters)
    extra = set(optimizer_parameters) - set(model_parameters)
    if missing or extra:
        raise RuntimeError(
            "Full RFT optimizer does not exactly cover trainable transformer "
            f"parameters: missing={len(missing)}, extra={len(extra)}"
        )
    return {
        "optimizer_parameter_tensors": len(optimizer_parameters),
        "optimizer_parameters": sum(
            parameter.numel() for parameter in optimizer_parameters.values()
        ),
    }


def summarize_pseudo_buffer(path: str | Path) -> dict[str, float]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    result: dict[str, float] = {
        "rft_buffer/pseudo_chunks": float(len(rows)),
        "rft_buffer/tasks": float(len({str(row.get("task", "unknown")) for row in rows})),
        "rft_buffer/contexts": float(len({str(row.get("context_id", "")) for row in rows})),
    }
    domains = [str(row.get("domain", "unknown")) for row in rows]
    for domain in sorted(set(domains)):
        result[f"rft_buffer/domain_{domain}_fraction"] = (
            domains.count(domain) / len(domains) if domains else 0.0
        )
    for name, getter in (
        ("process_score", lambda row: row.get("process_score")),
        (
            "action_score",
            lambda row: row.get("action_critic", {}).get("action_score"),
        ),
    ):
        values = [float(value) for row in rows if (value := getter(row)) is not None]
        if values:
            for key, value in numeric_metrics(name, values).items():
                statistic = key.rsplit("/", 1)[-1]
                result[f"rft_buffer/{name}_{statistic}"] = value
    return result


def filter_real_dataset_by_split(
    dataset, split_manifest: str | Path, *, stages: tuple[str, ...]
) -> dict[str, int]:
    """Restrict the official source loader to manifest-selected episodes."""
    manifest = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
    allowed: dict[str, set[int]] = {}
    for task in manifest["tasks"]:
        for domain in task["domains"].values():
            repo = str(Path(domain["source_repo"]).resolve())
            episode_ids = allowed.setdefault(repo, set())
            for stage in stages:
                episode_ids.update(
                    int(index)
                    for index in domain[f"{stage}_source_episode_indices"]
                )

    found: set[str] = set()
    kept_segments = 0
    for source_dataset in dataset._datasets:
        repo = str(Path(source_dataset.root).resolve())
        selected = allowed.get(repo)
        if selected is None:
            source_dataset.new_metas = []
            continue
        found.add(repo)
        source_dataset.new_metas = [
            meta
            for meta in source_dataset.new_metas
            if int(meta["episode_index"]) in selected
        ]
        kept_segments += len(source_dataset.new_metas)
    missing = sorted(set(allowed) - found)
    if missing:
        raise RuntimeError(f"Real source repos missing from official loader: {missing[:5]}")
    dataset._datasets = [source for source in dataset._datasets if source.new_metas]
    if not dataset._datasets or kept_segments <= 0:
        raise RuntimeError("Manifest filtering removed every real training segment")
    dataset.item_id_to_dataset_id, dataset.acc_dset_num = (
        dataset._get_item_id_to_dataset_id()
    )
    return {
        "source_repos": len(found),
        "selected_episodes": sum(len(indices) for indices in allowed.values()),
        "kept_segments": kept_segments,
    }


def write_source_counts(
    trainer, counts: torch.Tensor, real_fraction: float
) -> dict | None:
    counts = counts.clone()
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(counts)
    if trainer.config.rank != 0:
        return None
    real, pseudo = (int(value) for value in counts.cpu().tolist())
    total = real + pseudo
    result = {
        "real_samples": real,
        "pseudo_samples": pseudo,
        "total_samples": total,
        "observed_real_fraction": real / total if total else 0.0,
        "observed_pseudo_fraction": pseudo / total if total else 0.0,
        "target_real_fraction": real_fraction,
        "objective": "joint_video_action_flow_matching",
        "parameter_scope": "full_transformer",
    }
    output = Path(trainer.config.save_root) / "rft_source_counts.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="robotwin_train")
    parser.add_argument("--pseudo-jsonl", required=True)
    parser.add_argument("--legacy-pseudo-action-waiver-sha256")
    parser.add_argument("--legacy-pseudo-action-waiver-rows", type=int)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-selection-mode",
        choices=("naive", "process", "action", "dual"),
        default="dual",
    )
    parser.add_argument("--real-fraction", type=float, default=0.7)
    parser.add_argument(
        "--real-data-fraction",
        type=float,
        default=1.0,
        help="Deterministic fraction of real chunks used by this run.",
    )
    parser.add_argument("--data-fraction-seed", type=int, default=42)
    parser.add_argument(
        "--mixing-mode",
        choices=("ratio", "union", "auxiliary"),
        default="ratio",
        help="Ratio sampling for iterative RFT, or each real+pseudo item once/epoch.",
    )
    parser.add_argument(
        "--schedule-mode",
        choices=SCHEDULE_MODES,
        default="steps",
        help="Iterative steps, Base-aligned auxiliary pseudo, or exact epochs.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=0,
        help="When positive, derive optimizer steps from exactly this many epochs.",
    )
    parser.add_argument(
        "--real-data-mode",
        choices=("stage1", "stage2", "stage1-stage2", "stage1-stage2-visible"),
        default="stage1-stage2-visible",
        help=(
            "stage1-stage2 reads and filters the original dataset; "
            "stage1-stage2-visible reads the preselected action-visible union."
        ),
    )
    parser.add_argument(
        "--real-chunk-mode",
        choices=("full", "first-transition", "all-transitions"),
        default="full",
        help=(
            "Use full official real segments, or crop every real sample to the "
            "same F=2 transition used by online pseudo chunks, or expose every "
            "adjacent F=2 transition from every selected real segment."
        ),
    )
    parser.add_argument("--save-root")
    parser.add_argument("--invocation-id", required=True)
    parser.add_argument("--pseudo-global-batch", type=int, default=8)
    parser.add_argument("--pseudo-loss-weight", type=float, default=0.25)
    parser.add_argument("--pseudo-loss-warmup-steps", type=int, default=3000)
    parser.add_argument("--pseudo-sampler-seed", type=int, default=43)
    parser.add_argument("--base-auxiliary-smoke-steps", type=int, default=0)
    parser.add_argument("--base-real-regression-steps", type=int, default=0)
    parser.add_argument(
        "--base-auxiliary-activation-checkpointing",
        action="store_true",
    )
    parser.add_argument(
        "--base-real-regression-activation-checkpointing",
        action="store_true",
    )
    parser.add_argument("--pseudo-preflight-samples", type=int, default=32)
    parser.add_argument("--pseudo-preflight-seed", type=int, default=20260815)
    parser.add_argument("--outer-step", type=int, default=0)
    parser.add_argument("--swanlab-step-offset", type=int, default=0)
    args = parser.parse_args()
    if args.pseudo_loss_weight < 0:
        parser.error("--pseudo-loss-weight must be non-negative")
    if args.pseudo_loss_warmup_steps < 0:
        parser.error("--pseudo-loss-warmup-steps must be non-negative")
    if args.base_auxiliary_smoke_steps and args.schedule_mode != "base-auxiliary-pseudo":
        parser.error(
            "--base-auxiliary-smoke-steps requires "
            "--schedule-mode base-auxiliary-pseudo"
        )
    if (
        args.base_auxiliary_activation_checkpointing
        and args.schedule_mode != "base-auxiliary-pseudo"
    ):
        parser.error(
            "--base-auxiliary-activation-checkpointing requires "
            "--schedule-mode base-auxiliary-pseudo"
        )
    if args.base_real_regression_steps:
        if args.schedule_mode != "base-auxiliary-pseudo":
            parser.error(
                "--base-real-regression-steps requires "
                "--schedule-mode base-auxiliary-pseudo"
            )
        if args.pseudo_loss_weight != 0:
            parser.error("real-only regression requires --pseudo-loss-weight 0")
    elif args.base_real_regression_activation_checkpointing:
        parser.error(
            "--base-real-regression-activation-checkpointing requires "
            "--base-real-regression-steps"
        )

    import wan_va.train as train_module
    from wan_va.configs import VA_CONFIGS
    from wan_va.dataset import MultiLatentLeRobotDataset
    from wan_va.distributed.util import init_distributed
    from wan_va.utils import init_logger, logger

    class JointRFTTrainer(train_module.Trainer):
        """Official full trainer with only the dataset source replaced."""

        def __init__(self, config):
            self.auxiliary_mode = args.schedule_mode == "base-auxiliary-pseudo"
            original_factory = train_module.MultiLatentLeRobotDataset
            original_collate = train_module.pad_latent_batch_collate
            original_swanlab_init = None
            original_swanlab_functions = None
            external_swanlab_sink = None
            external_swanlab_run = None
            dataset_report = {}

            def mixed_factory(config, num_init_worker):
                if args.real_data_mode in ("stage2", "stage1-stage2"):
                    manifest = json.loads(
                        args.split_manifest.read_text(encoding="utf-8")
                    )
                    config.dataset_path = str(Path(manifest["source_root"]).resolve())
                real = MultiLatentLeRobotDataset(
                    config=config, num_init_worker=num_init_worker
                )
                if args.real_data_mode in ("stage2", "stage1-stage2"):
                    filter_real_dataset_by_split(
                        real,
                        args.split_manifest,
                        stages=(
                            ("stage2",)
                            if args.real_data_mode == "stage2"
                            else ("stage1", "stage2")
                        ),
                    )
                if args.real_chunk_mode == "first-transition":
                    real = FirstTransitionChunkDataset(
                        real,
                        frame_chunk_size=int(config.frame_chunk_size),
                    )
                elif args.real_chunk_mode == "all-transitions":
                    real = AllTransitionChunkDataset(
                        real,
                        frame_chunk_size=int(config.frame_chunk_size),
                    )
                full_real_items = len(real)
                if args.real_data_fraction != 1.0:
                    real = DeterministicFractionDataset(
                        real,
                        fraction=args.real_data_fraction,
                        seed=args.data_fraction_seed,
                    )
                pseudo = GeneratedChunkDataset(
                    args.pseudo_jsonl,
                    config,
                    expected_split_sha256=sha256_file(args.split_manifest),
                    expected_selection_mode=args.expected_selection_mode,
                    split_manifest_path=args.split_manifest,
                    legacy_pseudo_action_waiver_sha256=args.legacy_pseudo_action_waiver_sha256,
                    legacy_pseudo_action_waiver_rows=args.legacy_pseudo_action_waiver_rows,
                )
                dataset_report.update(
                    {
                        "real_items": len(real),
                        "full_real_items": full_real_items,
                        "real_data_fraction": args.real_data_fraction,
                        "data_fraction_seed": args.data_fraction_seed,
                        "pseudo_items": len(pseudo),
                        "union_items": len(real) + len(pseudo),
                    }
                )
                if args.mixing_mode == "union":
                    return UnionRFTDataset(real, pseudo)
                return RatioMixedDataset(real, pseudo, real_fraction=args.real_fraction)

            def mixed_collate(batch):
                return mixed_pad_latent_batch_collate(batch, original_collate)

            if not self.auxiliary_mode:
                train_module.MultiLatentLeRobotDataset = mixed_factory
                train_module.pad_latent_batch_collate = mixed_collate
            if getattr(config, "enable_swanlab", False) and config.rank == 0:
                import swanlab

                if os.getenv("LINGBOT_SWANLAB_EXTERNAL") == "1":
                    external_swanlab_sink = ExternalSwanLabSink()
                    external_swanlab_run = ExternalSwanLabRun()
                    original_swanlab_functions = {
                        "login": swanlab.login,
                        "init": swanlab.init,
                        "log": swanlab.log,
                        "finish": swanlab.finish,
                    }
                    swanlab.login = lambda *args, **kwargs: None
                    swanlab.init = lambda **kwargs: external_swanlab_run
                    swanlab.log = external_swanlab_sink.log
                    swanlab.finish = external_swanlab_sink.finish
                else:
                    original_swanlab_init = swanlab.init

                    def resumable_swanlab_init(**kwargs):
                        run_id = os.getenv("LINGBOT_SWANLAB_RUN_ID")
                        if run_id:
                            kwargs["id"] = run_id
                            kwargs["resume"] = "allow"
                        group = os.getenv("LINGBOT_SWANLAB_GROUP")
                        if group:
                            kwargs["group"] = group
                        return original_swanlab_init(**kwargs)

                    swanlab.init = resumable_swanlab_init
            try:
                super().__init__(config)
            finally:
                train_module.MultiLatentLeRobotDataset = original_factory
                train_module.pad_latent_batch_collate = original_collate
                if original_swanlab_functions is not None:
                    self._swanlab = external_swanlab_sink
                    self.swanlab_run = external_swanlab_run
                    for name, function in original_swanlab_functions.items():
                        setattr(swanlab, name, function)
                elif original_swanlab_init is not None:
                    swanlab.init = original_swanlab_init
            model_shape_report = loaded_model_attention_report(
                self.transformer, rank=int(config.rank)
            )
            logger.info("RFT loaded model attention preflight: %s", model_shape_report)
            report = enable_full_finetune(self.transformer)
            optimizer_report = verify_full_optimizer(
                self.transformer, self.optimizer
            )
            self.rft_source_counts = torch.zeros(
                2, dtype=torch.long, device=self.device
            )
            global_batch_size = (
                int(config.batch_size)
                * int(config.world_size)
                * int(self.gradient_accumulation_steps)
            )
            scheduler_type = getattr(self.config, "lr_scheduler_type", "constant")
            if self.auxiliary_mode:
                _, auxiliary_save_steps = resolve_base_auxiliary_steps(
                    int(self.config.num_steps),
                    args.base_auxiliary_smoke_steps,
                    args.base_real_regression_steps,
                    self.config.save_steps,
                )
                validate_base_auxiliary_contract(
                    mixing_mode=args.mixing_mode,
                    real_chunk_mode=args.real_chunk_mode,
                    batch_size_per_rank=int(config.batch_size),
                    global_batch_size=global_batch_size,
                    activation_checkpointing=getattr(
                        config, "enable_activation_checkpointing", False
                    ),
                    warmup_steps=int(config.warmup_steps),
                    scheduler_type=scheduler_type,
                    max_episode_frames=int(config.max_episode_frames),
                    num_epochs=args.num_epochs,
                    real_data_mode=args.real_data_mode,
                    real_data_root=config.dataset_path,
                    prepared_data_root=args.split_manifest.parent,
                    required_activation_checkpointing=(
                        args.base_auxiliary_activation_checkpointing
                        or
                        args.base_real_regression_activation_checkpointing
                    ),
                )
                pseudo = GeneratedChunkDataset(
                    args.pseudo_jsonl,
                    config,
                    expected_split_sha256=sha256_file(args.split_manifest),
                    expected_selection_mode=args.expected_selection_mode,
                    split_manifest_path=args.split_manifest,
                    legacy_pseudo_action_waiver_sha256=args.legacy_pseudo_action_waiver_sha256,
                    legacy_pseudo_action_waiver_rows=args.legacy_pseudo_action_waiver_rows,
                )
                try:
                    preflight_report = build_pseudo_preflight_report(
                        self.train_loader.dataset,
                        pseudo,
                        frame_chunk_size=int(config.frame_chunk_size),
                        sample_count=args.pseudo_preflight_samples,
                        seed=args.pseudo_preflight_seed,
                    )
                except Exception as exc:
                    preflight_report = unexpected_pseudo_preflight_failure_report(
                        exc,
                        frame_chunk_size=int(config.frame_chunk_size),
                        sample_count=args.pseudo_preflight_samples,
                        seed=args.pseudo_preflight_seed,
                    )
                distributed = torch.distributed.is_initialized()
                local_failed = not preflight_report["ok"]
                failed = torch.tensor(
                    int(local_failed), dtype=torch.int32, device=self.device
                )
                if distributed:
                    torch.distributed.all_reduce(
                        failed, op=torch.distributed.ReduceOp.MAX
                    )
                global_failed = bool(failed.item())
                if distributed:
                    rank_violations = [None] * torch.distributed.get_world_size()
                    torch.distributed.all_gather_object(
                        rank_violations, list(preflight_report["violations"])
                    )
                    if int(config.rank) == 0:
                        preflight_report["ranks"] = [
                            {
                                "rank": rank,
                                "ok": not bool(violations),
                                "violations": violations,
                            }
                            for rank, violations in enumerate(rank_violations)
                        ]
                        preflight_report["ok"] = not global_failed
                write_error = ""
                if int(config.rank) == 0:
                    try:
                        write_preflight_report(
                            preflight_report,
                            Path(self.config.save_root)
                            / "pseudo_preflight_report.json",
                        )
                    except Exception as exc:
                        write_error = f"{type(exc).__name__}: {exc}"
                write_failed = torch.tensor(
                    int(bool(write_error)), dtype=torch.int32, device=self.device
                )
                if distributed:
                    torch.distributed.all_reduce(
                        write_failed, op=torch.distributed.ReduceOp.MAX
                    )
                if write_failed.item():
                    raise RuntimeError(
                        "rank 0 could not write pseudo preflight report"
                        + (f": {write_error}" if write_error else "")
                    )
                if global_failed:
                    raise ValueError(
                        "pseudo preflight failed; see pseudo_preflight_report.json"
                    )
                dataset_report.update(flatten_preflight_summary(preflight_report))
                self.pseudo_sampler = DistributedSampler(
                    pseudo,
                    num_replicas=int(config.world_size),
                    rank=int(config.rank),
                    shuffle=True,
                    seed=args.pseudo_sampler_seed,
                    drop_last=False,
                )
                pseudo_loader_kwargs = {
                    "dataset": pseudo,
                    "batch_size": 1,
                    "sampler": self.pseudo_sampler,
                    "shuffle": False,
                    "num_workers": 0,
                    "pin_memory": True,
                }
                self.pseudo_loader = DataLoader(**pseudo_loader_kwargs)
                self.pseudo_loader_iter = None
                self.pseudo_epoch = 0
                dataset_report.update(
                    {
                        "real_items": len(self.train_loader.dataset),
                        "full_real_items": len(self.train_loader.dataset),
                        "real_data_fraction": 1.0,
                        "data_fraction_seed": args.data_fraction_seed,
                        "pseudo_items": len(pseudo),
                        "union_items": len(self.train_loader.dataset) + len(pseudo),
                        "pseudo_validation_mode": pseudo.provenance_report[
                            "validation_mode"
                        ],
                        "package_split_sha256": pseudo.provenance_report[
                            "package_split_sha256"
                        ],
                        "current_split_sha256": pseudo.provenance_report[
                            "current_split_sha256"
                        ],
                        "pseudo_validated_rows": pseudo.provenance_report[
                            "validated_rows"
                        ],
                    }
                )
                self.auxiliary_report = auxiliary_schedule_report(
                    optimizer_steps=int(self.config.num_steps),
                    world_size=int(config.world_size),
                    real_global_batch=global_batch_size,
                    pseudo_global_batch=args.pseudo_global_batch,
                    pseudo_loss_weight=args.pseudo_loss_weight,
                    pseudo_loss_warmup_steps=args.pseudo_loss_warmup_steps,
                )
                self.pseudo_microbatches_per_update = int(
                    self.auxiliary_report["pseudo_microbatches_per_rank_update"]
                )
                initialize_production_auxiliary_state(
                    self, train_module=train_module,
                    pseudo_loss_weight=args.pseudo_loss_weight,
                    pseudo_loss_warmup_steps=args.pseudo_loss_warmup_steps,
                    pseudo_sampler_seed=args.pseudo_sampler_seed,
                    pseudo_microbatches_per_update=self.pseudo_microbatches_per_update,
                )
            if args.schedule_mode == "epochs" and args.mixing_mode != "union":
                raise ValueError("epochs schedule requires --mixing-mode union")
            trainer_steps = int(self.config.num_steps)
            final_steps = resolve_optimizer_steps(
                schedule_mode=args.schedule_mode,
                configured_steps=trainer_steps,
                num_epochs=args.num_epochs,
                loader_microbatches=len(self.train_loader),
                gradient_accumulation_steps=int(self.gradient_accumulation_steps),
                real_items=dataset_report["real_items"],
                pseudo_items=dataset_report["pseudo_items"],
            )
            self.config.num_steps = final_steps
            if self.auxiliary_mode:
                self.config.save_steps = auxiliary_save_steps
                self.config.save_interval = self.config.save_steps[0]
                dataset_report.update(
                    {
                        "save_interval": self.config.save_interval,
                        "save_steps": self.config.save_steps,
                    }
                )
            else:
                self.config.save_steps = [final_steps]
                self.config.save_interval = final_steps
            stale_target_error = ""
            if int(self.config.rank) == 0:
                try:
                    reject_existing_snapshot_targets(
                        self.config.save_root, list(self.config.save_steps)
                    )
                except FileExistsError as exc:
                    stale_target_error = str(exc)
            stale_target_failed = torch.tensor(
                int(bool(stale_target_error)), dtype=torch.int32, device=self.device
            )
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    stale_target_failed, op=torch.distributed.ReduceOp.MAX
                )
            if stale_target_failed.item():
                raise FileExistsError(
                    stale_target_error
                    or "rank 0 found a stale/existing inference snapshot target"
                )
            scheduler_rebuilt = should_rebuild_lr_scheduler(
                schedule_mode=args.schedule_mode,
                scheduler_type=scheduler_type,
                trainer_steps=trainer_steps,
                final_steps=final_steps,
            )
            if scheduler_rebuilt:
                self.lr_scheduler = rebuild_lr_scheduler(
                    self.optimizer, self.config, train_module
                )
            expected_real_fraction = (
                (
                    1.0
                    if args.pseudo_loss_weight == 0
                    else 64 / (64 + args.pseudo_global_batch)
                )
                if self.auxiliary_mode
                else (
                    dataset_report["real_items"] / dataset_report["union_items"]
                    if args.mixing_mode == "union"
                    else args.real_fraction
                )
            )
            self.expected_real_fraction = expected_real_fraction
            self.dataset_report = {
                **dataset_report,
                "schedule_mode": args.schedule_mode,
                "mixing_mode": args.mixing_mode,
                "num_epochs": args.num_epochs,
                "steps_per_epoch": (
                    len(self.train_loader) // int(self.gradient_accumulation_steps)
                    if args.num_epochs
                    else 0
                ),
                "optimizer_steps": int(self.config.num_steps),
                "smoke_mode": int(bool(args.base_auxiliary_smoke_steps)),
                "smoke_steps": int(args.base_auxiliary_smoke_steps),
                "real_only_regression": int(bool(args.base_real_regression_steps)),
                "real_only_regression_steps": int(args.base_real_regression_steps),
                "real_only_regression_activation_checkpointing": int(
                    args.base_real_regression_activation_checkpointing
                ),
                "base_auxiliary_activation_checkpointing": int(
                    args.base_auxiliary_activation_checkpointing
                ),
                "wam_collection_model": os.getenv("INITIAL_MODEL"),
                "rft_initial_model": os.getenv(
                    "RFT_INITIAL_MODEL", os.getenv("WAN_VA_MODEL_PATH")
                ),
                "trainer_reference_steps": trainer_steps,
                "lr_scheduler_rebuilt": scheduler_rebuilt,
                "expected_real_fraction": expected_real_fraction,
                "expected_pseudo_fraction": 1.0 - expected_real_fraction,
            }
            if self.auxiliary_mode:
                self.dataset_report.update(self.auxiliary_report)
                self.dataset_report.update(
                    {
                        "pseudo_sampler_seed": args.pseudo_sampler_seed,
                        "real_sampler_seed": 42,
                        "rng_isolation": (
                            "synchronous pseudo fetch and model RNG inside "
                            "torch.random.fork_rng(update,rank,microbatch)"
                        ),
                        "trajectory_note": (
                            "Real sampler/RNG stream is preserved, but model trajectory "
                            "necessarily diverges because pseudo gradients are added."
                        ),
                    }
                )
            else:
                self.dataset_report.update(
                    {
                        "global_batch_size": global_batch_size,
                        **expected_sample_exposure(
                        real_items=dataset_report["real_items"],
                        pseudo_items=dataset_report["pseudo_items"],
                        optimizer_steps=final_steps,
                        global_batch_size=global_batch_size,
                        base_reference_steps=trainer_steps,
                        schedule_mode=args.schedule_mode,
                        real_draw_fraction=expected_real_fraction,
                        ),
                    }
                )
            if self.config.rank == 0:
                report_path = Path(self.config.save_root) / "rft_dataset_report.json"
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(
                    json.dumps(self.dataset_report, indent=2) + "\n",
                    encoding="utf-8",
                )
            self.defer_swanlab_finish = True
            self.buffer_metrics = summarize_pseudo_buffer(args.pseudo_jsonl)
            self.buffer_metrics["rft/outer_step"] = float(args.outer_step)
            self.buffer_metrics.update(
                {
                    f"rft_dataset/{key}": float(value)
                    for key, value in self.dataset_report.items()
                    if isinstance(value, (int, float))
                }
            )
            if self.config.rank == 0 and self.swanlab_run is not None:
                original_log = self._swanlab.log

                def offset_log(data, step=None, *log_args, **log_kwargs):
                    if step is not None:
                        step = int(step) + args.swanlab_step_offset
                    return original_log(
                        data, *log_args, step=step, **log_kwargs
                    )

                self._swanlab.log = offset_log
                self._swanlab.log(self.buffer_metrics, step=0)
            logger.info(
                "Joint full-model RFT: trainable=%d frozen=%d optimizer=%d "
                "objective=official latent_loss+action_loss",
                report["trainable_parameters"],
                report["frozen_parameters"],
                optimizer_report["optimizer_parameters"],
            )
            logger.info(
                "RFT data: real_chunk_mode=%s mixing_mode=%s epochs=%d "
                "real_items=%d pseudo_items=%d steps=%d batch_size_per_rank=%d "
                "real_fraction=%.3f",
                args.real_chunk_mode,
                args.mixing_mode,
                args.num_epochs,
                dataset_report["real_items"],
                dataset_report["pseudo_items"],
                int(self.config.num_steps),
                int(config.batch_size),
                expected_real_fraction,
            )
            logger.info(
                "RFT exposure: %s",
                json.dumps(self.dataset_report, sort_keys=True),
            )

        def _get_next_pseudo_batch(self):
            batch, self.pseudo_loader_iter, self.pseudo_epoch = next_sampler_batch(
                self.pseudo_loader,
                self.pseudo_loader_iter,
                self.pseudo_sampler,
                self.pseudo_epoch,
            )
            return batch

        def _auxiliary_train_step(self, batch, batch_idx):
            return production_auxiliary_train_step(self, batch, batch_idx)

        def _train_step(self, batch, batch_idx):
            if self.auxiliary_mode:
                losses = self._auxiliary_train_step(batch, batch_idx)
            else:
                source = batch.pop("_rft_source")
                self.rft_source_counts += torch.bincount(
                    source.to(self.device).flatten(), minlength=2
                )
                losses = super()._train_step(batch, batch_idx)
            source_metrics = None
            if losses["should_log"]:
                counts = self.rft_source_counts.clone()
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(counts)
                real, pseudo = (int(value) for value in counts.tolist())
                total = real + pseudo
                source_metrics = {
                    "rft_source/cumulative_real_samples": real,
                    "rft_source/cumulative_pseudo_samples": pseudo,
                    "rft_source/cumulative_real_fraction": real / total
                    if total
                    else 0.0,
                    "rft_source/cumulative_pseudo_fraction": pseudo / total
                    if total
                    else 0.0,
                }
            if (
                losses["should_log"]
                and self.config.rank == 0
                and self.swanlab_run is not None
            ):
                total = losses["latent_loss"] + losses["action_loss"]
                metrics = {
                    "loss_metrics/last_microbatch_total_loss": total.item(),
                    "gpu_rank0/allocated_gib": torch.cuda.memory_allocated(
                        self.device
                    )
                    / (1024**3),
                    "gpu_rank0/reserved_gib": torch.cuda.memory_reserved(
                        self.device
                    )
                    / (1024**3),
                    "gpu_rank0/peak_allocated_gib": torch.cuda.max_memory_allocated(
                        self.device
                    )
                    / (1024**3),
                }
                if self.auxiliary_mode:
                    metrics.update(
                        {
                            "loss_metrics/real_latent_loss_unscaled": losses[
                                "real_latent_loss_unscaled"
                            ].item(),
                            "loss_metrics/real_action_loss_unscaled": losses[
                                "real_action_loss_unscaled"
                            ].item(),
                            "loss_metrics/pseudo_latent_loss_unscaled": losses[
                                "pseudo_latent_loss_unscaled"
                            ].item(),
                            "loss_metrics/pseudo_action_loss_unscaled": losses[
                                "pseudo_action_loss_unscaled"
                            ].item(),
                            "loss_metrics/combined_update_loss": losses[
                                "combined_update_loss"
                            ].item(),
                            "rft_dataset/pseudo_loss_weight_target": float(
                                args.pseudo_loss_weight
                            ),
                            "rft_dataset/pseudo_loss_weight_effective": losses[
                                "pseudo_loss_weight_effective"
                            ],
                            "rft_dataset/real_global_batch": 64.0,
                            "rft_dataset/pseudo_global_batch": float(
                                args.pseudo_global_batch
                            ),
                        }
                    )
                if self.dataset_report["steps_per_epoch"]:
                    metrics["rft/epoch_progress"] = (
                        float(self.step) / self.dataset_report["steps_per_epoch"]
                    )
                if source_metrics is not None:
                    metrics.update(source_metrics)
                self._swanlab.log(metrics, step=self.step)
            return losses

        def _finish_swanlab(self):
            if getattr(self, "defer_swanlab_finish", False):
                return
            return super()._finish_swanlab()

        def save_checkpoint(self):
            """Validate and attest inherited inference snapshot saves collectively."""
            super().save_checkpoint()
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            expected = (
                Path(self.config.save_root) / "checkpoints"
                / f"checkpoint_step_{self.step}" / "transformer"
            )
            local_failed = 0
            if self.config.rank == 0:
                try:
                    require_complete_transformer(
                        expected,
                        context=f"checkpoint_step_{self.step} after inherited save",
                    )
                    write_snapshot_marker(expected.parent, args.invocation_id)
                    require_snapshot_invocation(expected.parent, args.invocation_id)
                    logger.info(
                        "Saved inference model snapshot (not a resumable optimizer checkpoint): %s",
                        expected.parent,
                    )
                except (OSError, RuntimeError, ValueError):
                    local_failed = 1
            failed = torch.tensor(local_failed, dtype=torch.int32, device=self.device)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX)
            if failed.item():
                raise RuntimeError(
                    f"checkpoint_step_{self.step} is incomplete after inherited save: {expected}"
                )

        def train(self):
            try:
                super().train()
                final_transformer = (
                    Path(self.config.save_root) / "checkpoints"
                    / f"checkpoint_step_{self.step}" / "transformer"
                )
                local_final_failed = 0
                if self.config.rank == 0:
                    try:
                        require_snapshot_invocation(
                            final_transformer.parent, args.invocation_id
                        )
                    except RuntimeError:
                        local_final_failed = 1
                final_failed = torch.tensor(
                    local_final_failed, dtype=torch.int32, device=self.device
                )
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(final_failed, op=torch.distributed.ReduceOp.MAX)
                if final_failed.item():
                    raise RuntimeError(f"final checkpoint is incomplete: {final_transformer}")
                source_counts = write_source_counts(
                    self, self.rft_source_counts, self.expected_real_fraction
                )
                if self.config.rank == 0:
                    if source_counts is not None:
                        self.dataset_report.update(
                            {
                                "observed_real_samples": source_counts[
                                    "real_samples"
                                ],
                                "observed_pseudo_samples": source_counts[
                                    "pseudo_samples"
                                ],
                                **getattr(self, "last_aux_loss_metrics", {}),
                            }
                        )
                        report_path = (
                            Path(self.config.save_root) / "rft_dataset_report.json"
                        )
                        report_path.write_text(
                            json.dumps(self.dataset_report, indent=2) + "\n",
                            encoding="utf-8",
                        )
                    logger.info(
                        "RFT source counts written to %s", self.config.save_root
                    )
                    if self.swanlab_run is not None and source_counts is not None:
                        self._swanlab.log(
                            {
                                "rft_source/real_samples": source_counts["real_samples"],
                                "rft_source/pseudo_samples": source_counts["pseudo_samples"],
                                "rft_source/observed_real_fraction": source_counts[
                                    "observed_real_fraction"
                                ],
                                "rft_source/observed_pseudo_fraction": source_counts[
                                    "observed_pseudo_fraction"
                                ],
                            },
                            step=max(int(self.step) - 1, 0),
                        )
            finally:
                self.defer_swanlab_finish = False
                super()._finish_swanlab()

        # Deliberately inherit Trainer.compute_loss unchanged. The official loss is
        # latent flow matching + action flow matching for both real and pseudo data.

    init_logger()
    config = VA_CONFIGS[args.config_name]
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    init_distributed(world_size, local_rank, rank)
    transformer_path = Path(os.environ["WAN_VA_MODEL_PATH"]) / "transformer"
    try:
        local_shape_report = checkpoint_attention_report(transformer_path, rank=rank)
    except Exception as exc:
        local_shape_report = {
            "rank": rank,
            "transformer_path": str(transformer_path.expanduser().resolve()),
            "error": f"{type(exc).__name__}: {exc}",
        }
    rank_shape_reports = [None] * world_size
    torch.distributed.all_gather_object(rank_shape_reports, local_shape_report)
    require_matching_rank_reports(local_shape_report, rank_shape_reports)
    logger.info("RFT checkpoint attention preflight: %s", local_shape_report)
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    if args.save_root:
        config.save_root = args.save_root
    trainer = JointRFTTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
