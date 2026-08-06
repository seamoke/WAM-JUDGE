from __future__ import annotations

import unittest

import torch

from robotwin_critic.two_stage_rft.batched_wam_server import collapse_cfg_batch


class BatchedWAMServerTest(unittest.TestCase):
    def test_cfg_scale_one_still_collapses_2b_to_b(self) -> None:
        positive = torch.full((2, 3), 2.0)
        negative = torch.full((2, 3), -4.0)
        result = collapse_cfg_batch(
            torch.cat([positive, negative]), scale=1.0, enabled=True
        )
        self.assertEqual(tuple(result.shape), (2, 3))
        self.assertTrue(torch.equal(result, positive))

    def test_cfg_guidance_combines_matching_batch_items(self) -> None:
        positive = torch.tensor([[2.0], [4.0]])
        negative = torch.tensor([[1.0], [-2.0]])
        result = collapse_cfg_batch(
            torch.cat([positive, negative]), scale=3.0, enabled=True
        )
        self.assertTrue(
            torch.equal(result, negative + 3.0 * (positive - negative))
        )


if __name__ == "__main__":
    unittest.main()
