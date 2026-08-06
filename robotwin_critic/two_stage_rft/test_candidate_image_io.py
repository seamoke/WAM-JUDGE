from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from robotwin_critic.two_stage_rft.generate_wam_candidates import save_image


class CandidateImageIoTest(unittest.TestCase):
    def test_unit_float_image_is_scaled_to_uint8(self) -> None:
        image = np.linspace(0.0, 1.0, 12 * 16 * 3, dtype=np.float32).reshape(12, 16, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            save_image(path, image)
            restored = np.asarray(Image.open(path))
        self.assertGreater(int(restored.max()), 250)
        self.assertGreater(float(restored.mean()), 100.0)

    def test_uint8_image_is_not_rescaled(self) -> None:
        image = np.full((8, 8, 3), 127, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            save_image(path, image)
            restored = np.asarray(Image.open(path))
        self.assertEqual(int(restored.min()), 127)
        self.assertEqual(int(restored.max()), 127)


if __name__ == "__main__":
    unittest.main()
