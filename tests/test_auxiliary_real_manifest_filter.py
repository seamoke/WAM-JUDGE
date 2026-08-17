from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    from robotwin_critic.two_stage_rft.train_joint_rft import (
        build_real_dataset_for_mode,
    )
except ImportError:
    build_real_dataset_for_mode = None


@unittest.skipIf(build_real_dataset_for_mode is None, "training dependencies required")
class AuxiliaryRealManifestFilterTest(unittest.TestCase):
    def test_stage1_stage2_filters_the_original_source_dataset(self) -> None:
        class Config:
            dataset_path = "unfiltered"

        class Source:
            def __init__(self, root: Path):
                self.root = root
                self.new_metas = [
                    {"episode_index": episode_id} for episode_id in range(6)
                ]

            def __len__(self):
                return len(self.new_metas)

        class Dataset:
            def __init__(self, source: Source):
                self._datasets = [source]

            def _get_item_id_to_dataset_id(self):
                return (
                    {index: 0 for index in range(len(self._datasets[0]))},
                    {0: 0},
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            manifest = root / "split_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "source_root": str(repo),
                        "tasks": [
                            {
                                "domains": {
                                    "clean": {
                                        "source_repo": str(repo),
                                        "stage1_source_episode_indices": [1, 2],
                                        "stage2_source_episode_indices": [4],
                                    }
                                }
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            def factory(*, config, num_init_worker):
                self.assertEqual(config.dataset_path, str(repo.resolve()))
                self.assertEqual(num_init_worker, 7)
                return Dataset(Source(repo))

            dataset, report = build_real_dataset_for_mode(
                Config(),
                7,
                real_data_mode="stage1-stage2",
                split_manifest=manifest,
                dataset_factory=factory,
            )

            self.assertEqual(
                [meta["episode_index"] for meta in dataset._datasets[0].new_metas],
                [1, 2, 4],
            )
            self.assertTrue(report["filter_applied"])
            self.assertEqual(report["selected_episodes_by_stage"], {"stage1": 2, "stage2": 1})
            self.assertEqual(report["kept_segments_by_stage"], {"stage1": 2, "stage2": 1})


if __name__ == "__main__":
    unittest.main()
