from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    import torch

    from robotwin_critic.two_stage_rft.action_only_dataset import (
        FirstTransitionChunkDataset,
        GeneratedChunkDataset,
        RatioMixedDataset,
        generated_actions_to_tensor,
        mixed_pad_latent_batch_collate,
    )
except ImportError:
    torch = None
    GeneratedChunkDataset = None
    RatioMixedDataset = None
    generated_actions_to_tensor = None
    FirstTransitionChunkDataset = None
    mixed_pad_latent_batch_collate = None


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
    def test_ratio_dataset_marks_source_without_mutating_inputs(self) -> None:
        real_item = {"value": torch.tensor(1)}
        pseudo_item = {"value": torch.tensor(2)}
        dataset = RatioMixedDataset([real_item], [pseudo_item], real_fraction=0.7)
        self.assertEqual(int(dataset[0]["_rft_source"]), 0)
        self.assertEqual(int(dataset[7]["_rft_source"]), 1)
        self.assertNotIn("_rft_source", real_item)
        self.assertNotIn("_rft_source", pseudo_item)

    def test_ratio_view_is_exactly_70_30_per_cycle(self) -> None:
        dataset = RatioMixedDataset(TinyDataset("r", 20), TinyDataset("p", 20))
        sources = [dataset.source_for_index(index) for index in range(10)]
        self.assertEqual(sources.count("real"), 7)
        self.assertEqual(sources.count("pseudo"), 3)

    def test_ratio_view_supports_20_80_per_cycle(self) -> None:
        dataset = RatioMixedDataset(
            TinyDataset("r", 20),
            TinyDataset("p", 20),
            real_fraction=0.2,
        )
        sources = [dataset.source_for_index(index) for index in range(10)]
        self.assertEqual(sources.count("real"), 2)
        self.assertEqual(sources.count("pseudo"), 8)

    def test_first_transition_real_chunk_has_fixed_two_frames(self) -> None:
        source = [
            {
                "latents": torch.arange(48 * 5).reshape(48, 5, 1, 1),
                "actions": torch.arange(30 * 5 * 16).reshape(30, 5, 16, 1),
                "actions_mask": torch.ones(30, 5, 16, 1, dtype=torch.bool),
                "text_emb": torch.zeros(3, 4),
            }
        ]
        item = FirstTransitionChunkDataset(source)[0]
        self.assertEqual(tuple(item["latents"].shape), (48, 2, 1, 1))
        self.assertEqual(tuple(item["actions"].shape), (30, 2, 16, 1))
        self.assertEqual(item["latents_mask"].tolist(), [True, True])

    def test_mixed_collate_preserves_source_labels(self) -> None:
        batch = [
            {
                "latents": torch.zeros(48, 2, 1, 1),
                "actions": torch.zeros(30, 2, 16, 1),
                "actions_mask": torch.ones(30, 2, 16, 1, dtype=torch.bool),
                "text_emb": torch.zeros(3, 4),
                "latents_mask": torch.ones(2, dtype=torch.bool),
                "_rft_source": torch.tensor(source),
            }
            for source in (0, 1)
        ]

        def base_collate(items):
            return {
                key: torch.stack([item[key] for item in items])
                for key in items[0]
            }

        result = mixed_pad_latent_batch_collate(batch, base_collate)
        self.assertEqual(result["_rft_source"].tolist(), [0, 1])
        self.assertEqual(tuple(result["latents"].shape), (2, 48, 2, 1, 1))

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
        self.assertEqual(int(mask[:, 0].sum()), 0)
        self.assertEqual(int(mask[:, 1].sum()), 16 * 16)
        self.assertEqual(float(tensor[:, 0].abs().sum()), 0.0)

    def test_packed_conditioning_block_is_rejected(self) -> None:
        used = list(range(7)) + [28] + list(range(7, 14)) + [29]
        inverse = [len(used)] * 30
        for index, channel in enumerate(used):
            inverse[channel] = index
        config = SimpleNamespace(
            action_per_frame=16,
            inverse_used_action_channel_ids=inverse,
            norm_stat={"q01": [-1.0] * 30, "q99": [1.0] * 30},
        )
        with self.assertRaisesRegex(ValueError, "conditioning block"):
            generated_actions_to_tensor(
                np.zeros((32, 16), dtype=np.float32),
                latent_frames=2,
                config=config,
            )

    def test_generated_dataset_uses_instance_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latent_path = root / "latents.pt"
            text_path = root / "text.pt"
            action_path = root / "actions.npy"
            empty_path = root / "empty.pt"
            selected_path = root / "selected.jsonl"
            pseudo_latents = torch.zeros(1, 48, 2, 2, 2)
            pseudo_latents[:, :, 1] = 2.0
            torch.save(pseudo_latents, latent_path)
            torch.save(torch.zeros(1, 3, 4), text_path)
            torch.save(torch.ones(1, 3, 4), empty_path)
            np.save(action_path, np.zeros((16, 16), dtype=np.float32))
            selected_path.write_text(
                json.dumps(
                    {
                        "split_manifest_sha256": "split",
                        "latent_path": str(latent_path),
                        "text_emb_path": str(text_path),
                        "action_path": str(action_path),
                        "rft_selection": {"mode": "dual"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            used = list(range(7)) + [28] + list(range(7, 14)) + [29]
            inverse = [len(used)] * 30
            for index, channel in enumerate(used):
                inverse[channel] = index
            config = SimpleNamespace(
                action_per_frame=16,
                inverse_used_action_channel_ids=inverse,
                norm_stat={"q01": [-1.0] * 30, "q99": [1.0] * 30},
                empty_emb_path=str(empty_path),
                frame_chunk_size=2,
                cfg_prob=0.0,
                param_dtype=torch.bfloat16,
            )
            dataset = GeneratedChunkDataset(
                selected_path,
                config,
                expected_split_sha256="split",
                expected_selection_mode="dual",
            )
            item = dataset[0]
            self.assertEqual(tuple(item["latents"].shape), (48, 2, 2, 2))
            self.assertEqual(item["latents_mask"].tolist(), [True, True])
            self.assertEqual(float(item["latents"][:, 0].abs().sum()), 0.0)
            self.assertGreater(float(item["latents"][:, 1].abs().sum()), 0.0)
            self.assertEqual(int(item["actions_mask"][:, 0].sum()), 0)
            self.assertEqual(item["text_emb"].dtype, torch.bfloat16)
            with self.assertRaisesRegex(ValueError, "selection modes"):
                GeneratedChunkDataset(
                    selected_path,
                    config,
                    expected_split_sha256="split",
                    expected_selection_mode="process",
                )


if __name__ == "__main__":
    unittest.main()
