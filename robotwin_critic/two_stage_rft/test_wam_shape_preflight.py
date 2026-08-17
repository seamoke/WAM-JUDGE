from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from robotwin_critic.two_stage_rft.wam_shape_preflight import (
    checkpoint_attention_report,
    loaded_model_attention_report,
    require_matching_rank_reports,
)


class WamShapePreflightTest(unittest.TestCase):
    def test_checkpoint_headers_report_3072_projection_contract(self) -> None:
        shapes = {
            f"blocks.0.attn2.to_{name}.weight": [3072, 3072]
            for name in ("q", "k", "v")
        }

        class FakeHandle:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def keys(self):
                return shapes.keys()

            def get_slice(self, name):
                return SimpleNamespace(get_shape=lambda: shapes[name])

        with tempfile.TemporaryDirectory() as directory:
            transformer = Path(directory)
            (transformer / "config.json").write_text(
                json.dumps({"num_attention_heads": 24, "attention_head_dim": 128})
            )
            (transformer / "diffusion_pytorch_model.safetensors").touch()
            fake_module = SimpleNamespace(safe_open=lambda *args, **kwargs: FakeHandle())
            with mock.patch.dict("sys.modules", {"safetensors": fake_module}):
                report = checkpoint_attention_report(transformer, rank=2)
        self.assertEqual(report["expected_attention_inner_dim"], 3072)
        self.assertEqual(report["attention_projection_count"], 3)

    def test_checkpoint_headers_reject_512_wide_projection(self) -> None:
        shapes = {"blocks.0.attn2.to_k.weight": [512, 3072]}

        class FakeHandle:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def keys(self):
                return shapes.keys()

            def get_slice(self, name):
                return SimpleNamespace(get_shape=lambda: shapes[name])

        with tempfile.TemporaryDirectory() as directory:
            transformer = Path(directory)
            (transformer / "config.json").write_text(
                json.dumps({"num_attention_heads": 24, "attention_head_dim": 128})
            )
            (transformer / "diffusion_pytorch_model.safetensors").touch()
            fake_module = SimpleNamespace(safe_open=lambda *args, **kwargs: FakeHandle())
            with mock.patch.dict("sys.modules", {"safetensors": fake_module}):
                with self.assertRaisesRegex(RuntimeError, "rank=2.*512"):
                    checkpoint_attention_report(transformer, rank=2)

    def test_matching_reports_ignore_node_local_path(self) -> None:
        common = {
            "num_attention_heads": 24,
            "attention_head_dim": 128,
            "expected_attention_inner_dim": 3072,
            "attention_projection_count": 120,
            "config_sha256": "a" * 64,
            "tensor_shape_manifest_sha256": "b" * 64,
        }
        reports = [
            {**common, "rank": 0, "transformer_path": "/node0/checkpoint"},
            {**common, "rank": 2, "transformer_path": "/node1/checkpoint"},
        ]
        self.assertIsNone(require_matching_rank_reports(reports[0], reports))

    def test_rank_checkpoint_mismatch_is_fatal_and_reports_all_ranks(self) -> None:
        reports = [
            {
                "rank": rank,
                "transformer_path": f"/node{rank}/checkpoint",
                "num_attention_heads": 24,
                "attention_head_dim": 128,
                "expected_attention_inner_dim": 3072,
                "attention_projection_count": 120,
                "config_sha256": str(rank) * 64,
                "tensor_shape_manifest_sha256": "b" * 64,
            }
            for rank in (0, 2)
        ]
        with self.assertRaisesRegex(RuntimeError, "checkpoint/config mismatch") as caught:
            require_matching_rank_reports(reports[0], reports)
        self.assertIn('"rank": 2', str(caught.exception))

    def test_one_rank_inspection_error_is_collectively_fatal(self) -> None:
        good = {
            "rank": 0,
            "transformer_path": "/node0/checkpoint",
            "num_attention_heads": 24,
            "attention_head_dim": 128,
            "expected_attention_inner_dim": 3072,
            "attention_projection_count": 120,
            "config_sha256": "a" * 64,
            "tensor_shape_manifest_sha256": "b" * 64,
        }
        failed = {"rank": 2, "error": "missing checkpoint shard"}
        with self.assertRaisesRegex(RuntimeError, "preflight failed") as caught:
            require_matching_rank_reports(good, [good, failed])
        self.assertIn("missing checkpoint shard", str(caught.exception))

    def test_loaded_model_reports_expected_projection_contract(self) -> None:
        linear = lambda: SimpleNamespace(in_features=3072, out_features=3072)
        attention = SimpleNamespace(to_q=linear(), to_k=linear(), to_v=linear())
        model = SimpleNamespace(
            num_attention_heads=24,
            attention_head_dim=128,
            blocks=[SimpleNamespace(attn2=attention) for _ in range(2)],
        )
        report = loaded_model_attention_report(model, rank=2)
        self.assertEqual(report["expected_attention_inner_dim"], 3072)
        self.assertEqual(report["checked_attention_projections"], 6)

    def test_loaded_model_rejects_512_wide_key_projection_with_rank(self) -> None:
        attention = SimpleNamespace(
            to_q=SimpleNamespace(in_features=3072, out_features=3072),
            to_k=SimpleNamespace(in_features=3072, out_features=512),
            to_v=SimpleNamespace(in_features=3072, out_features=3072),
        )
        model = SimpleNamespace(
            num_attention_heads=24,
            attention_head_dim=128,
            blocks=[SimpleNamespace(attn2=attention)],
        )
        with self.assertRaisesRegex(RuntimeError, "rank=2.*512"):
            loaded_model_attention_report(model, rank=2)


if __name__ == "__main__":
    unittest.main()
