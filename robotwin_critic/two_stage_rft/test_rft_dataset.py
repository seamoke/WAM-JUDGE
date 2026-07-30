from __future__ import annotations

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

HAS_RFT_DEPS = (
    importlib.util.find_spec("pyarrow") is not None
    and importlib.util.find_spec("torch") is not None
)
if HAS_RFT_DEPS:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    from robotwin_critic.two_stage_rft.rft_dataset import (
        CAMERAS,
        build_dataset,
    )
    from robotwin_critic.two_stage_rft.validate_rft_dataset import (
        structural_validate,
    )


@unittest.skipUnless(
    HAS_RFT_DEPS, "pyarrow and torch are required for RFT materialization"
)
class RftDatasetTest(unittest.TestCase):
    def test_builds_non_destructive_loader_shaped_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_repo = root / "source" / "task-a"
            (source_repo / "meta").mkdir(parents=True)
            (source_repo / "meta" / "info.json").write_text(
                json.dumps(
                    {
                        "codebase_version": "v2.1",
                        "fps": 30,
                        "chunks_size": 1000,
                        "total_episodes": 1,
                        "total_frames": 17,
                        "features": {},
                    }
                )
            )
            (source_repo / "meta" / "tasks.jsonl").write_text(
                '{"task_index":0,"task":"move object"}\n'
            )
            parquet = source_repo / "data" / "chunk-000" / "episode_000000.parquet"
            parquet.parent.mkdir(parents=True)
            table = pa.table(
                {
                    "action": pa.array(
                        np.zeros((17, 16), dtype=np.float32).tolist(),
                        type=pa.list_(pa.float32(), 16),
                    ),
                    "episode_index": pa.array([0] * 17, type=pa.int64()),
                    "frame_index": pa.array(range(17), type=pa.int64()),
                    "index": pa.array(range(17), type=pa.int64()),
                    "timestamp": pa.array(
                        np.arange(17, dtype=np.float32) / 30, type=pa.float32()
                    ),
                }
            )
            pq.write_table(table, parquet)

            action_path = root / "actions.npy"
            np.save(action_path, np.zeros((17, 16), dtype=np.float32))
            latent_paths = {}
            for camera_index, camera in enumerate(CAMERAS):
                path = root / f"camera-{camera_index}.pth"
                torch.save(
                    {
                        "latent": torch.zeros(5, 4),
                        "latent_num_frames": 2,
                        "latent_height": 1,
                        "latent_width": 1,
                        "frame_ids": torch.tensor([0, 4, 8, 12, 16]),
                        "text_emb": torch.zeros(2, 8),
                    },
                    path,
                )
                latent_paths[camera] = str(path)
            candidates = root / "selected.jsonl"
            candidates.write_text(
                json.dumps(
                    {
                        "task": "task-a",
                        "text": "move object",
                        "source_repo": str(source_repo),
                        "source_parquet": str(parquet),
                        "start_frame": 0,
                        "end_frame": 17,
                        "fps": 30,
                        "action_path": str(action_path),
                        "latent_paths": latent_paths,
                        "consistency": {"accepted": True},
                        "process_score": 0.4,
                        "action_critic": {
                            "accepted": True,
                            "action_score": 0.9,
                        },
                    }
                )
                + "\n"
            )
            empty_emb = root / "empty_emb.pt"
            torch.save(torch.zeros(2, 8), empty_emb)
            output = root / "rft"
            manifest = build_dataset(
                candidates,
                output,
                empty_embedding=empty_emb,
                min_process_score=0.1,
            )
            self.assertEqual(manifest["episodes"], 1)
            self.assertTrue(parquet.is_file(), "Source data must remain untouched")
            self.assertEqual(structural_validate(output)["episodes"], 1)
            written = pq.read_table(
                output
                / "rft_selected_chunks"
                / "task-a"
                / "data"
                / "chunk-000"
                / "episode_000000.parquet"
            )
            self.assertEqual(written.num_rows, 17)


if __name__ == "__main__":
    unittest.main()
