"""Full-model LingBot-VA joint video/action RFT on real and pseudo data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from robotwin_critic.two_stage_rft.action_only_dataset import (
    FirstTransitionChunkDataset,
    GeneratedChunkDataset,
    RatioMixedDataset,
    mixed_pad_latent_batch_collate,
)
from robotwin_critic.two_stage_rft.log_online_collection_swanlab import (
    numeric_metrics,
)
from robotwin_critic.two_stage_rft.protocol import sha256_file


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
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-selection-mode",
        choices=("naive", "process", "action", "dual"),
        default="dual",
    )
    parser.add_argument("--real-fraction", type=float, default=0.7)
    parser.add_argument(
        "--real-data-mode",
        choices=("stage1", "stage1-stage2", "stage1-stage2-visible"),
        default="stage1-stage2-visible",
        help=(
            "stage1-stage2 reads and filters the original dataset; "
            "stage1-stage2-visible reads the preselected action-visible union."
        ),
    )
    parser.add_argument(
        "--real-chunk-mode",
        choices=("full", "first-transition"),
        default="full",
        help=(
            "Use full official real segments, or crop every real sample to the "
            "same F=2 transition used by online pseudo chunks."
        ),
    )
    parser.add_argument("--save-root")
    parser.add_argument("--outer-step", type=int, default=0)
    parser.add_argument("--swanlab-step-offset", type=int, default=0)
    args = parser.parse_args()

    import wan_va.train as train_module
    from wan_va.configs import VA_CONFIGS
    from wan_va.dataset import MultiLatentLeRobotDataset
    from wan_va.distributed.util import init_distributed
    from wan_va.utils import init_logger, logger

    class JointRFTTrainer(train_module.Trainer):
        """Official full trainer with only the dataset source replaced."""

        def __init__(self, config):
            original_factory = train_module.MultiLatentLeRobotDataset
            original_collate = train_module.pad_latent_batch_collate
            original_swanlab_init = None
            original_swanlab_functions = None
            external_swanlab_sink = None
            external_swanlab_run = None

            def mixed_factory(config, num_init_worker):
                if args.real_data_mode == "stage1-stage2":
                    manifest = json.loads(
                        args.split_manifest.read_text(encoding="utf-8")
                    )
                    config.dataset_path = str(Path(manifest["source_root"]).resolve())
                real = MultiLatentLeRobotDataset(
                    config=config, num_init_worker=num_init_worker
                )
                if args.real_data_mode == "stage1-stage2":
                    filter_real_dataset_by_split(
                        real,
                        args.split_manifest,
                        stages=("stage1", "stage2"),
                    )
                if args.real_chunk_mode == "first-transition":
                    real = FirstTransitionChunkDataset(
                        real,
                        frame_chunk_size=int(config.frame_chunk_size),
                    )
                pseudo = GeneratedChunkDataset(
                    args.pseudo_jsonl,
                    config,
                    expected_split_sha256=sha256_file(args.split_manifest),
                    expected_selection_mode=args.expected_selection_mode,
                )
                return RatioMixedDataset(
                    real,
                    pseudo,
                    real_fraction=args.real_fraction,
                )

            def mixed_collate(batch):
                return mixed_pad_latent_batch_collate(batch, original_collate)

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
            report = enable_full_finetune(self.transformer)
            optimizer_report = verify_full_optimizer(
                self.transformer, self.optimizer
            )
            self.rft_source_counts = torch.zeros(
                2, dtype=torch.long, device=self.device
            )
            self.defer_swanlab_finish = True
            self.buffer_metrics = summarize_pseudo_buffer(args.pseudo_jsonl)
            self.buffer_metrics["rft/outer_step"] = float(args.outer_step)
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
                "RFT data: real_chunk_mode=%s batch_size_per_rank=%d "
                "real_fraction=%.3f",
                args.real_chunk_mode,
                int(config.batch_size),
                args.real_fraction,
            )

        def _train_step(self, batch, batch_idx):
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
                if source_metrics is not None:
                    metrics.update(source_metrics)
                self._swanlab.log(metrics, step=self.step)
            return losses

        def _finish_swanlab(self):
            if getattr(self, "defer_swanlab_finish", False):
                return
            return super()._finish_swanlab()

        def train(self):
            try:
                super().train()
                source_counts = write_source_counts(
                    self, self.rft_source_counts, args.real_fraction
                )
                if self.config.rank == 0:
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
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    if args.save_root:
        config.save_root = args.save_root
    trainer = JointRFTTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
