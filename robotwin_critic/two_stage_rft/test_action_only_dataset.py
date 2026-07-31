from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

try:
    from robotwin_critic.two_stage_rft.action_only_dataset import (
        RatioMixedDataset,
        generated_actions_to_tensor,
    )
except ImportError:
    RatioMixedDataset = None
    generated_actions_to_tensor = None


class TinyDataset:
    def __init__(self, prefix: str, length: int):
        self.prefix = prefix
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return f"{self.prefix}-{index}"


@unittest.skipIf(RatioMixedDataset is None, "torch is required")
class ActionOnlyDatasetTest(unittest.TestCase):
    def test_ratio_view_is_exactly_70_30_per_cycle(self) -> None:
        dataset = RatioMixedDataset(TinyDataset("r", 20), TinyDataset("p", 20))
        sources = [dataset.source_for_index(index) for index in range(10)]
        self.assertEqual(sources.count("real"), 7)
        self.assertEqual(sources.count("pseudo"), 3)

    def test_generated_relative_actions_follow_wam_channel_contract(self) -> None:
        used = list(range(7)) + [28] + list(range(7, 14)) + [29]
        inverse = [len(used)] * 30
        for index, channel in enumerate(used):
            inverse[channel] = index
        config = SimpleNamespace(
            action_per_frame=16,
            inverse_used_action_channel_ids=inverse,
            norm_stat={"q01": [-1.0] * 30, "q99": [1.0] * 30},
        )
        tensor, mask = generated_actions_to_tensor(
            np.zeros((16, 16), dtype=np.float32),
            latent_frames=2,
            config=config,
        )
        self.assertEqual(tuple(tensor.shape), (30, 2, 16, 1))
        self.assertEqual(tuple(mask.shape), (30, 2, 16, 1))
        self.assertEqual(int(mask[:, 0].sum()), 16 * 16)


if __name__ == "__main__":
    unittest.main()
