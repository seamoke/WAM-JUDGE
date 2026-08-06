from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

try:
    import torch

    from robotwin_critic.two_stage_rft.train_joint_rft import (
        enable_full_finetune,
        filter_real_dataset_by_split,
        summarize_pseudo_buffer,
        verify_full_optimizer,
    )
except ImportError:
    torch = None
    enable_full_finetune = None
    verify_full_optimizer = None
    summarize_pseudo_buffer = None
    filter_real_dataset_by_split = None


@unittest.skipIf(enable_full_finetune is None, "torch is required")
class JointRFTTest(unittest.TestCase):
    def test_full_finetune_enables_every_parameter(self) -> None:
        model = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.Linear(4, 2),
        )
        for index, parameter in enumerate(model.parameters()):
            parameter.requires_grad_(index % 2 == 0)
        report = enable_full_finetune(model)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        self.assertEqual(report["frozen_parameters"], 0)
        self.assertEqual(
            report["trainable_parameters"],
            sum(parameter.numel() for parameter in model.parameters()),
        )
        optimizer = torch.optim.AdamW(model.parameters())
        optimizer_report = verify_full_optimizer(model, optimizer)
        self.assertEqual(
            optimizer_report["optimizer_parameters"],
            report["trainable_parameters"],
        )

    def test_optimizer_must_cover_all_trainable_parameters(self) -> None:
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
        optimizer = torch.optim.AdamW(model[0].parameters())
        with self.assertRaisesRegex(RuntimeError, "does not exactly cover"):
            verify_full_optimizer(model, optimizer)

    def test_pseudo_buffer_metrics_cover_both_rewards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "buffer.jsonl"
            rows = [
                {
                    "process_score": process,
                    "action_critic": {"action_score": action},
                }
                for process, action in ((6.0, 0.8), (10.0, 1.0))
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            metrics = summarize_pseudo_buffer(path)
            self.assertEqual(metrics["rft_buffer/pseudo_chunks"], 2.0)
            self.assertEqual(metrics["rft_buffer/process_score_mean"], 8.0)
            self.assertEqual(metrics["rft_buffer/action_score_min"], 0.8)

    def test_real_dataset_uses_manifest_stage1_and_stage2_episodes(self) -> None:
        class Source:
            def __init__(self, root, episode_ids):
                self.root = root
                self.new_metas = [
                    {"episode_index": episode_id} for episode_id in episode_ids
                ]

            def __len__(self):
                return len(self.new_metas)

        class Multi:
            def __init__(self, sources):
                self._datasets = sources

            def _get_item_id_to_dataset_id(self):
                mapping = {}
                offsets = {}
                index = 0
                for dataset_index, source in enumerate(self._datasets):
                    offsets[dataset_index] = index
                    for _ in source.new_metas:
                        mapping[index] = dataset_index
                        index += 1
                return mapping, offsets

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            extra = root / "extra"
            repo.mkdir()
            extra.mkdir()
            manifest = root / "split_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
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
                        ]
                    }
                ),
                encoding="utf-8",
            )
            dataset = Multi(
                [Source(repo, [0, 1, 2, 3, 4]), Source(extra, [0, 1])]
            )
            report = filter_real_dataset_by_split(
                dataset, manifest, stages=("stage1", "stage2")
            )
            self.assertEqual(
                [meta["episode_index"] for meta in dataset._datasets[0].new_metas],
                [1, 2, 4],
            )
            self.assertEqual(len(dataset._datasets), 1)
            self.assertEqual(report["selected_episodes"], 3)
            self.assertEqual(report["kept_segments"], 3)


if __name__ == "__main__":
    unittest.main()
