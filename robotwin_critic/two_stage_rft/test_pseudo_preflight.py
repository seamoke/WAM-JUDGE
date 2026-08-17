from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

try:
    import torch
except ImportError:
    torch = None

from robotwin_critic.two_stage_rft.pseudo_preflight import (
    QUANTILE_SAMPLE_MAX_VALUES,
    PseudoPreflightError,
    _quantile_sample_positions,
    _stats,
    audit_pseudo_preflight,
    build_pseudo_preflight_report,
    flatten_preflight_summary,
    spread_sample_indices,
    unexpected_pseudo_preflight_failure_report,
)


class PseudoPreflightStaticTest(unittest.TestCase):
    def test_unexpected_failure_report_is_json_and_summary_compatible(self):
        report = unexpected_pseudo_preflight_failure_report(
            RuntimeError("dataset exploded"),
            sample_count=17,
            seed=23,
            frame_chunk_size=5,
        )
        self.assertEqual(
            (report["sample_count"], report["seed"], report["frame_chunk_size"]),
            (17, 23, 5),
        )
        self.assertFalse(report["ok"])
        self.assertEqual(
            report["violations"],
            ["unexpected local preflight exception: RuntimeError: dataset exploded"],
        )
        summary = flatten_preflight_summary(report)
        self.assertFalse(summary["pseudo_preflight_ok"])
        self.assertEqual(summary["pseudo_preflight_requested_samples"], 17)
        self.assertEqual(summary["pseudo_preflight_real_samples"], 0)
        self.assertIsNone(summary["pseudo_preflight_pseudo_actions_mean"])
        json.dumps(report, allow_nan=False)

    def test_summary_flattens_requested_scalars(self):
        names = ("mean", "std", "min", "max", "p01", "p50", "p99")
        stats = {
            source: {
                field: {name: float(index) for index, name in enumerate(names)}
                for field in ("latents", "text_emb", "actions")
            } | {"actions_mask": {"true_fraction": 0.75}}
            for source in ("real", "pseudo")
        }
        summary = flatten_preflight_summary({
            "ok": True,
            "sample_count": 32,
            "sampled_indices": {"real": [1, 4], "pseudo": [0]},
            "stats": stats,
        })
        self.assertEqual(summary["pseudo_preflight_requested_samples"], 32)
        self.assertEqual(summary["pseudo_preflight_real_samples"], 2)
        self.assertEqual(summary["pseudo_preflight_pseudo_samples"], 1)
        self.assertEqual(summary["pseudo_preflight_real_latents_p99"], 6.0)
        self.assertEqual(summary["pseudo_preflight_pseudo_action_mask_true_fraction"], 0.75)
        self.assertFalse(any("count" in key for key in summary))

    def test_joint_trainer_places_audit_before_pseudo_loader(self):
        source = Path(__file__).with_name("train_joint_rft.py").read_text(encoding="utf-8")
        auxiliary = source.index("if self.auxiliary_mode:", source.index("super().__init__(config)"))
        audit = source.index("build_pseudo_preflight_report(", auxiliary)
        loader = source.index("self.pseudo_loader = DataLoader", auxiliary)
        self.assertLess(audit, loader)
        self.assertIn('parser.add_argument("--pseudo-preflight-samples", type=int, default=32)', source)
        self.assertIn('parser.add_argument("--pseudo-preflight-seed", type=int, default=20260815)', source)
        self.assertIn("self.train_loader.dataset,\n                        pseudo,", source[audit:loader])

    def test_auxiliary_pseudo_loader_fetches_synchronously_inside_fork_rng(self):
        source = Path(__file__).with_name("train_joint_rft.py").read_text(encoding="utf-8")
        start = source.index("pseudo_loader_kwargs = {")
        end = source.index("self.pseudo_loader = DataLoader", start)
        loader = source[start:end]
        self.assertIn('"num_workers": 0', loader)
        self.assertNotIn("prefetch_factor", loader)
        self.assertNotIn("persistent_workers", loader)
        self.assertNotIn('int(config.load_worker)', loader)
        self.assertIn("synchronous pseudo fetch and model RNG", source)

    def test_joint_trainer_converts_builder_exception_before_collectives(self):
        source = Path(__file__).with_name("train_joint_rft.py").read_text(encoding="utf-8")
        build = source.index("preflight_report = build_pseudo_preflight_report(")
        caught = source.index("except Exception as exc:", build)
        converted = source.index("unexpected_pseudo_preflight_failure_report(", caught)
        reduce = source.index("torch.distributed.all_reduce(", converted)
        gather_size = source.index("torch.distributed.get_world_size()", reduce)
        gather = source.index("torch.distributed.all_gather_object(", gather_size)
        self.assertLess(build, caught)
        self.assertLess(caught, converted)
        self.assertLess(converted, reduce)
        self.assertLess(reduce, gather_size)
        self.assertLess(gather_size, gather)


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class PseudoPreflightTest(unittest.TestCase):
    @staticmethod
    def _items(count=5, *, frames=3, channels=7, action_width=11, batched_text=False):
        text_shape = (1, 6, 9) if batched_text else (6, 9)
        return [
            {
                "latents": torch.arange(
                    channels * frames * 2 * 4, dtype=torch.float32
                ).reshape(channels, frames, 2, 4) + i,
                "text_emb": torch.ones(text_shape) * i,
                "actions": torch.ones(channels, frames, action_width, 1) * i,
                "actions_mask": torch.ones(
                    channels, frames, action_width, 1, dtype=torch.bool
                ),
            }
            for i in range(count)
        ]

    def test_passing_dynamic_shapes_and_json_serializable(self):
        real = self._items(frames=8, batched_text=True)
        pseudo = self._items(frames=3)
        report = audit_pseudo_preflight(real, pseudo, frame_chunk_size=3, seed=4)
        self.assertTrue(report["ok"])
        self.assertEqual(report["stats"]["pseudo"]["latents"]["count"], 5 * 7 * 3 * 2 * 4)
        self.assertEqual(report["stats"]["pseudo"]["actions_mask"]["true_fraction"], 1.0)
        self.assertNotIn("min", report["stats"]["pseudo"]["actions_mask"])
        self.assertEqual((report["sample_count"], report["seed"], report["frame_chunk_size"]), (32, 4, 3))
        json.dumps(report, allow_nan=False)

    def test_nonfinite_rejection_carries_complete_report(self):
        real, pseudo = self._items(), self._items()
        pseudo[0]["actions"][0, 0, 0, 0] = float("nan")
        with self.assertRaises(PseudoPreflightError) as caught:
            audit_pseudo_preflight(real, pseudo, 3)
        report = caught.exception.report
        self.assertFalse(report["finite"]["pseudo"]["actions"])
        self.assertTrue(any("finite" in message for message in report["violations"]))
        json.dumps(report, allow_nan=False)

    def test_catastrophic_finite_scale_drift_is_rejected(self):
        real, pseudo = self._items(), self._items()
        for item in pseudo:
            item["latents"].fill_(1_000_000.0)
        report = build_pseudo_preflight_report(real, pseudo, 3)
        self.assertFalse(report["ok"])
        self.assertTrue(
            any("latents catastrophic scale drift" in value for value in report["violations"])
        )

    def test_zero_variance_pseudo_stream_is_rejected(self):
        real, pseudo = self._items(), self._items()
        for item in pseudo:
            item["text_emb"].zero_()
        report = build_pseudo_preflight_report(real, pseudo, 3)
        self.assertFalse(report["ok"])
        self.assertTrue(
            any("text_emb distribution collapsed" in value for value in report["violations"])
        )

    def test_ordinary_distribution_shift_remains_observable_but_allowed(self):
        real, pseudo = self._items(), self._items()
        for item in pseudo:
            item["latents"].mul_(1.5).add_(2.0)
        report = build_pseudo_preflight_report(real, pseudo, 3)
        self.assertTrue(report["ok"])
        self.assertNotEqual(
            report["stats"]["real"]["latents"]["mean"],
            report["stats"]["pseudo"]["latents"]["mean"],
        )

    def test_shape_mismatch_ignores_frame_dimension(self):
        real = self._items(frames=8, channels=7, action_width=11)
        pseudo = self._items(frames=3, channels=8, action_width=12)
        report = build_pseudo_preflight_report(real, pseudo, 3)
        self.assertFalse(report["ok"])
        self.assertTrue(any("latents shape mismatch" in value for value in report["violations"]))
        self.assertTrue(any("actions shape mismatch" in value for value in report["violations"]))

    def test_pseudo_frame_chunk_enforced_for_latents_actions_and_mask(self):
        report = build_pseudo_preflight_report(self._items(frames=8), self._items(frames=4), 3)
        for field in ("latents", "actions", "actions_mask"):
            self.assertTrue(any(f"pseudo {field} F" in value for value in report["violations"]))

    def test_action_mask_shape_and_dtype(self):
        real, pseudo = self._items(), self._items()
        pseudo[0]["actions_mask"] = torch.ones(7, 3, 10, 1, dtype=torch.bool)
        pseudo[1]["actions_mask"] = torch.ones(7, 3, 11, 1)
        report = build_pseudo_preflight_report(real, pseudo, 3)
        self.assertTrue(any("must equal actions shape" in value for value in report["violations"]))
        self.assertTrue(any("dtype torch.bool" in value for value in report["violations"]))

    def test_empty_datasets_and_nonpositive_sample_count(self):
        report = build_pseudo_preflight_report([], [], 3)
        self.assertEqual(report["violations"], ["real dataset is empty", "pseudo dataset is empty"])
        with self.assertRaisesRegex(ValueError, "sample_count must be positive"):
            build_pseudo_preflight_report(self._items(), self._items(), 3, sample_count=0)

    def test_spread_indices_are_deterministic_unique_and_seeded(self):
        first = spread_sample_indices(100, 8, 17)
        self.assertEqual(first, spread_sample_indices(100, 8, 17))
        self.assertNotEqual(first, spread_sample_indices(100, 8, 18))
        self.assertEqual(len(first), len(set(first)))

    def test_quantile_positions_are_bounded_deterministic_and_cover_endpoints(self):
        count = QUANTILE_SAMPLE_MAX_VALUES * 5 + 17
        first = _quantile_sample_positions(count)
        second = _quantile_sample_positions(count)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.numel(), QUANTILE_SAMPLE_MAX_VALUES)
        self.assertEqual((first[0].item(), first[-1].item()), (0, count - 1))
        self.assertTrue(bool((first[1:] > first[:-1]).all()))

    def test_large_stats_bound_quantile_input_and_keep_all_value_moments(self):
        # PyTorch quantile rejects inputs above 2**24 elements.  This tensor is
        # one element over that historical limit but uses only ~32 MiB itself.
        count = (1 << 24) + 1
        values = torch.zeros(count, dtype=torch.float16)
        values[-1] = 1.0
        real_quantile = torch.quantile

        def guarded_quantile(value, *args, **kwargs):
            self.assertLessEqual(value.numel(), QUANTILE_SAMPLE_MAX_VALUES)
            return real_quantile(value, *args, **kwargs)

        with mock.patch.object(torch, "quantile", side_effect=guarded_quantile):
            first = _stats([values])
            second = _stats([values])
        self.assertEqual(first, second)
        self.assertEqual(first["count"], count)
        self.assertEqual((first["min"], first["max"]), (0.0, 1.0))
        self.assertAlmostEqual(first["mean"], 1.0 / count, delta=1e-20)
        self.assertAlmostEqual(first["std"], ((count - 1) / (count * count)) ** 0.5)

    def test_small_stats_keep_exact_quantile_semantics_across_tensor_parts(self):
        parts = [torch.tensor([9.0, 1.0]), torch.tensor([5.0, 3.0, 7.0])]
        expected = torch.cat(parts).to(torch.float64)
        stats = _stats(parts)
        quantiles = torch.quantile(
            expected, torch.tensor([0.01, 0.5, 0.99], dtype=torch.float64)
        )
        self.assertEqual(stats["count"], expected.numel())
        self.assertEqual(stats["mean"], float(expected.mean()))
        self.assertEqual(stats["std"], float(expected.std(unbiased=False)))
        self.assertEqual(
            (stats["p01"], stats["p50"], stats["p99"]),
            tuple(float(value) for value in quantiles),
        )

    def test_global_cpu_rng_is_preserved_even_when_dataset_draws(self):
        class RandomDataset:
            def __len__(self):
                return 4

            def __getitem__(self, index):
                torch.rand(19)
                return PseudoPreflightTest._items(1)[0]

        torch.manual_seed(9876)
        before = torch.random.get_rng_state().clone()
        audit_pseudo_preflight(RandomDataset(), RandomDataset(), 3)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_python_numpy_and_torch_cpu_rng_are_preserved(self):
        import random
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy unavailable")

        class RandomDataset:
            def __len__(self): return 2
            def __getitem__(self, index):
                random.random(); np.random.rand(); torch.rand(1)
                return PseudoPreflightTest._items(1)[0]

        random.seed(41); np.random.seed(42); torch.manual_seed(43)
        py_before, np_before = random.getstate(), np.random.get_state()
        torch_before = torch.random.get_rng_state().clone()
        audit_pseudo_preflight(RandomDataset(), RandomDataset(), 3)
        self.assertEqual(py_before, random.getstate())
        self.assertEqual(np_before[0], np.random.get_state()[0])
        self.assertTrue(np.array_equal(np_before[1], np.random.get_state()[1]))
        self.assertTrue(torch.equal(torch_before, torch.random.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
