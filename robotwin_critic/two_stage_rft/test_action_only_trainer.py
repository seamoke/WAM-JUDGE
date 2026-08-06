from __future__ import annotations

import unittest
from types import SimpleNamespace

try:
    import torch

    from robotwin_critic.two_stage_rft.train_action_only_rft import (
        action_flow_loss,
        freeze_video_backbone,
    )
except ImportError:
    torch = None
    freeze_video_backbone = None
    action_flow_loss = None


@unittest.skipIf(torch is None, "torch is required")
class ActionOnlyTrainerTest(unittest.TestCase):
    def test_only_action_specific_modules_remain_trainable(self) -> None:
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.Linear(2, 2)
                self.patch_embedding_mlp = torch.nn.Linear(2, 2)
                self.action_embedder = torch.nn.Linear(2, 2)
                self.condition_embedder_action = torch.nn.ModuleDict(
                    {
                        "time_embedder": torch.nn.Linear(2, 2),
                        "time_proj": torch.nn.Linear(2, 2),
                        "text_embedder": torch.nn.Linear(2, 2),
                    }
                )
                self.action_proj_out = torch.nn.Linear(2, 2)

        model = TinyModel()
        report = freeze_video_backbone(model)
        trainable = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        self.assertTrue(trainable)
        self.assertTrue(
            all(
                any(
                    token in name
                    for token in (
                        "action_embedder",
                        "condition_embedder_action.time_embedder",
                        "condition_embedder_action.time_proj",
                        "action_proj_out",
                    )
                )
                for name in trainable
            )
        )
        self.assertFalse(model.blocks.weight.requires_grad)
        self.assertFalse(
            model.condition_embedder_action["text_embedder"].weight.requires_grad
        )
        self.assertGreater(report["frozen_parameters"], 0)

    def test_action_loss_honors_frame_and_channel_masks(self) -> None:
        class UnitScheduler:
            @staticmethod
            def training_weight(timesteps):
                return torch.ones_like(timesteps)

        trainer = SimpleNamespace(
            train_scheduler_action=UnitScheduler(),
            gradient_accumulation_steps=2,
        )
        target = torch.zeros(1, 30, 2, 16, 1)
        channel_mask = torch.ones_like(target, dtype=torch.bool)
        prediction = torch.ones(1, 32, 30)
        input_dict = {
            "action_dict": {
                "targets": target,
                "timesteps": torch.zeros(1, 2),
                "actions_mask": channel_mask,
            },
            "latent_dict": {
                "latents_mask": torch.tensor([[True, False]]),
            },
        }
        loss = action_flow_loss(trainer, input_dict, (None, prediction))
        self.assertAlmostEqual(float(loss), 0.5)

    def test_conditioning_frame_does_not_dilute_pseudo_action_loss(self) -> None:
        class UnitScheduler:
            @staticmethod
            def training_weight(timesteps):
                return torch.ones_like(timesteps)

        trainer = SimpleNamespace(
            train_scheduler_action=UnitScheduler(),
            gradient_accumulation_steps=2,
        )
        target = torch.zeros(1, 30, 2, 16, 1)
        channel_mask = torch.ones_like(target, dtype=torch.bool)
        channel_mask[:, :, 0] = False
        prediction = torch.ones(1, 32, 30)
        input_dict = {
            "action_dict": {
                "targets": target,
                "timesteps": torch.zeros(1, 2),
                "actions_mask": channel_mask,
            },
            "latent_dict": {
                "latents_mask": torch.tensor([[True, True]]),
            },
        }
        loss = action_flow_loss(trainer, input_dict, (None, prediction))
        self.assertAlmostEqual(float(loss), 0.5)


if __name__ == "__main__":
    unittest.main()
