"""Train only LingBot-VA action-specific modules on 70/30 real/pseudo chunks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from robotwin_critic.two_stage_rft.action_only_dataset import (
    GeneratedChunkDataset,
    RatioMixedDataset,
)
from robotwin_critic.two_stage_rft.protocol import sha256_file


ACTION_MODULES = (
    "action_embedder",
    "condition_embedder_action.time_embedder",
    "condition_embedder_action.time_proj",
    "action_proj_out",
)


def is_action_parameter(name: str) -> bool:
    qualified = f".{name}."
    return any(f".{module}." in qualified for module in ACTION_MODULES)


def freeze_video_backbone(model) -> dict:
    trainable = []
    frozen = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(is_action_parameter(name))
        (trainable if parameter.requires_grad else frozen).append(name)
    if not trainable:
        raise RuntimeError(
            f"No action-specific parameters matched expected names {ACTION_MODULES}"
        )
    return {
        "trainable_names": trainable,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "frozen_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if not parameter.requires_grad
        ),
    }


def action_flow_loss(trainer, input_dict, prediction) -> torch.Tensor:
    _, action_prediction = prediction
    target = input_dict["action_dict"]["targets"]
    batch_size, sequence, channels = action_prediction.shape
    frames = target.shape[-3]
    if sequence % frames:
        raise ValueError(
            f"Action token sequence {sequence} is not divisible by frames {frames}"
        )
    action_prediction = (
        action_prediction.reshape(batch_size, frames, sequence // frames, channels)
        .permute(0, 3, 1, 2)
        .unsqueeze(-1)
    )
    batch, frames = input_dict["action_dict"]["timesteps"].shape
    weights = trainer.train_scheduler_action.training_weight(
        input_dict["action_dict"]["timesteps"].flatten()
    ).reshape(batch, frames)
    mask = input_dict["action_dict"]["actions_mask"].float()
    loss = F.mse_loss(
        action_prediction.float(), target.float().detach(), reduction="none"
    )
    loss = loss * weights[:, None, :, None, None] * mask
    loss = loss.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
    mask = mask.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
    mask_per_frame = mask.sum(dim=1)
    per_frame = loss.sum(dim=1) / mask_per_frame.clamp(min=1)
    valid = mask_per_frame.gt(0).float()
    if "latents_mask" in input_dict["latent_dict"]:
        valid *= input_dict["latent_dict"]["latents_mask"].flatten().float()
    value = (per_frame * valid).sum() / valid.sum().clamp(min=1)
    return value / trainer.gradient_accumulation_steps


def configure_optimizer(trainer) -> None:
    from wan_va.utils import warmup_constant_lambda, warmup_cosine_lambda

    config = trainer.config
    parameters = [
        parameter
        for parameter in trainer.transformer.parameters()
        if parameter.requires_grad
    ]
    trainer.optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=1e-8,
        weight_decay=config.weight_decay,
        fused=True,
        foreach=False,
    )
    scheduler_type = getattr(config, "lr_scheduler_type", "constant").lower()
    if scheduler_type == "constant":
        function = lambda step: warmup_constant_lambda(
            step, warmup_steps=config.warmup_steps
        )
    elif scheduler_type == "cosine":
        function = lambda step: warmup_cosine_lambda(
            step,
            warmup_steps=config.warmup_steps,
            total_steps=config.num_steps,
            min_lr_ratio=getattr(config, "min_lr_ratio", 0.0),
        )
    else:
        raise ValueError(f"Unsupported lr_scheduler_type={scheduler_type}")
    trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.optimizer, lr_lambda=function
    )


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
    parser.add_argument("--save-root")
    args = parser.parse_args()

    # Delay WAM imports so unit tests can exercise freeze/loss helpers on CPU.
    import wan_va.train as train_module
    from wan_va.configs import VA_CONFIGS
    from wan_va.dataset import MultiLatentLeRobotDataset
    from wan_va.distributed.util import init_distributed
    from wan_va.utils import init_logger, logger

    class ActionOnlyTrainer(train_module.Trainer):
        def __init__(self, config):
            original_factory = train_module.MultiLatentLeRobotDataset

            def mixed_factory(config, num_init_worker):
                if int(config.batch_size) != 1:
                    raise ValueError(
                        "Action-only RFT requires per-rank batch_size=1 so the "
                        "70/30 source schedule weights full real trajectories and "
                        "single pseudo chunks as intended; use gradient accumulation "
                        "for a larger global batch"
                    )
                real = MultiLatentLeRobotDataset(
                    config=config, num_init_worker=num_init_worker
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

            train_module.MultiLatentLeRobotDataset = mixed_factory
            try:
                super().__init__(config)
            finally:
                train_module.MultiLatentLeRobotDataset = original_factory
            report = freeze_video_backbone(self.transformer)
            configure_optimizer(self)
            self.rft_source_counts = torch.zeros(
                2, dtype=torch.long, device=self.device
            )
            logger.info(
                "Action-only RFT: trainable=%d frozen=%d modules=%s",
                report["trainable_parameters"],
                report["frozen_parameters"],
                ACTION_MODULES,
            )

        def _train_step(self, batch, batch_idx):
            source = batch.pop("_rft_source")
            self.rft_source_counts += torch.bincount(
                source.to(self.device).flatten(), minlength=2
            )
            return super()._train_step(batch, batch_idx)

        def train(self):
            super().train()
            counts = self.rft_source_counts.clone()
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(counts)
            if self.config.rank == 0:
                real, pseudo = (int(value) for value in counts.cpu().tolist())
                total = real + pseudo
                result = {
                    "real_samples": real,
                    "pseudo_samples": pseudo,
                    "total_samples": total,
                    "observed_real_fraction": real / total if total else 0.0,
                    "observed_pseudo_fraction": pseudo / total if total else 0.0,
                    "target_real_fraction": args.real_fraction,
                }
                output = Path(self.config.save_root) / "rft_source_counts.json"
                output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                logger.info("RFT source counts: %s", result)

        def compute_loss(self, input_dict, prediction):
            action_loss = action_flow_loss(self, input_dict, prediction)
            latent_loss = action_loss.detach().new_zeros(())
            return latent_loss, action_loss

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
    trainer = ActionOnlyTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
