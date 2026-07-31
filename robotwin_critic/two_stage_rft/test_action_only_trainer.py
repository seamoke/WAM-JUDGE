from __future__ import annotations

import unittest

try:
    import torch

    from robotwin_critic.two_stage_rft.train_action_only_rft import (
        freeze_video_backbone,
    )
except ImportError:
    torch = None
    freeze_video_backbone = None


@unittest.skipIf(torch is None, "torch is required")
class ActionOnlyTrainerTest(unittest.TestCase):
    def test_only_action_specific_modules_remain_trainable(self) -> None:
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.Linear(2, 2)
                self.patch_embedding_mlp = torch.nn.Linear(2, 2)
                self.action_embedder = torch.nn.Linear(2, 2)
                self.condition_embedder_action = torch.nn.Linear(2, 2)
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
                        "condition_embedder_action",
                        "action_proj_out",
                    )
                )
                for name in trainable
            )
        )
        self.assertFalse(model.blocks.weight.requires_grad)
        self.assertGreater(report["frozen_parameters"], 0)


if __name__ == "__main__":
    unittest.main()
