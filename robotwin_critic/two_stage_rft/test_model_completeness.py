"""Torch-free tests for fatal checkpoint completeness verification."""

from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

from robotwin_critic.two_stage_rft.model_completeness import (
    reject_existing_snapshot_targets,
    require_complete_transformer,
    require_snapshot_invocation,
    write_snapshot_marker,
)


class ModelCompletenessTest(unittest.TestCase):
    def test_missing_or_swallowed_checkpoint_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "checkpoint" / "transformer"
            with self.assertRaisesRegex(RuntimeError, "inherited save is incomplete"):
                require_complete_transformer(missing, context="inherited save")

    def test_complete_checkpoint_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            transformer = Path(directory) / "transformer"
            transformer.mkdir()
            (transformer / "config.json").write_text(json.dumps({"model": "test"}))
            (transformer / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
            self.assertIsNone(require_complete_transformer(transformer, context="final checkpoint"))

    def test_joint_trainer_checks_both_periodic_and_final_checkpoint(self) -> None:
        source = Path(__file__).with_name("train_joint_rft.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "require_complete_transformer"
        ]
        self.assertEqual(len(calls), 1)

    def test_existing_target_fails_closed_without_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "checkpoints" / "checkpoint_step_3"
            target.mkdir(parents=True)
            sentinel = target / "old"
            sentinel.write_text("keep")
            with self.assertRaisesRegex(FileExistsError, "stale/existing"):
                reject_existing_snapshot_targets(directory, [3, 6])
            self.assertEqual(sentinel.read_text(), "keep")

    def test_snapshot_marker_is_bound_to_invocation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint_step_3"
            transformer = checkpoint / "transformer"
            transformer.mkdir(parents=True)
            (transformer / "config.json").write_text("{}")
            (transformer / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
            write_snapshot_marker(checkpoint, "current")
            require_snapshot_invocation(checkpoint, "current")
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                require_snapshot_invocation(checkpoint, "stale")


if __name__ == "__main__":
    unittest.main()
