from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch  # noqa: F401
except ImportError:
    torch = None

if torch is not None:
    from robotwin_critic.two_stage_rft.generate_wam_candidates import (
        load_context_batch,
    )
    from robotwin_critic.vlac_finetune.common import VideoDecodeError


class FakeReader:
    def read(self, path: str, frame_index: int) -> np.ndarray:
        del frame_index
        if "broken" in path:
            raise VideoDecodeError(f"Video decoded zero frames: {path}")
        return np.zeros((16, 16, 3), dtype=np.uint8)


class GenerateWAMCandidatesTest(unittest.TestCase):
    @unittest.skipIf(torch is None, "PyTorch is not installed in the test interpreter")
    def test_corrupt_context_is_skipped_without_dropping_valid_context(self) -> None:
        valid = {
            "context_id": "valid",
            "task": "pick",
            "history_frame_indices": [0],
            "video_paths": {
                "observation.images.cam_high": "valid-high.mp4",
                "observation.images.cam_left_wrist": "valid-left.mp4",
                "observation.images.cam_right_wrist": "valid-right.mp4",
            },
        }
        broken = {
            **valid,
            "context_id": "broken",
            "video_paths": {
                **valid["video_paths"],
                "observation.images.cam_left_wrist": "broken-left.mp4",
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            loaded, observations, paths, skipped = load_context_batch(
                [(0, valid), (1, broken)],
                FakeReader(),
                Path(temporary),
                rank=0,
            )
        self.assertEqual([context["context_id"] for _, context in loaded], ["valid"])
        self.assertEqual(len(observations), 1)
        self.assertEqual(len(paths), 1)
        self.assertEqual(skipped[0]["context_id"], "broken")
        self.assertEqual(skipped[0]["error_type"], "VideoDecodeError")


if __name__ == "__main__":
    unittest.main()
