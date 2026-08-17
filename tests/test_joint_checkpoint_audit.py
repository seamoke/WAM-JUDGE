"""Focused finite-tensor tests; the behavioral test skips without Torch."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "robotwin_critic" / "two_stage_rft" / "audit_joint_checkpoint.py"


class JointCheckpointAuditTest(unittest.TestCase):
    def test_finite_status_is_part_of_every_pass_gate(self) -> None:
        source = AUDIT.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn('"base_finite": base_finite', source)
        self.assertIn('"checkpoint_finite": checkpoint_finite', source)
        self.assertIn('"finite": group_finite', source)
        self.assertIn("base_finite\n            and checkpoint_finite", source)

    def test_nonfinite_tensor_fails_when_torch_is_available(self) -> None:
        try:
            import torch
            from robotwin_critic.two_stage_rft import audit_joint_checkpoint as audit
        except ImportError:
            self.skipTest("Torch/safetensors runtime is unavailable locally")
        base_map = {
            prefix + "weight": "unused"
            for prefixes in audit.REQUIRED_GROUPS.values()
            for prefix in prefixes[:1]
        }
        checkpoint_map = dict(base_map)
        finite = {key: torch.ones(1) for key in base_map}
        nonfinite = {**finite, next(iter(base_map)): torch.tensor([float("nan")])}
        with mock.patch.object(audit, "weight_map", side_effect=[base_map, checkpoint_map]), \
             mock.patch.object(
                 audit, "load_tensor",
                 side_effect=lambda mapping, key: finite[key] if mapping is base_map else nonfinite[key],
             ):
            with self.assertRaises(RuntimeError) as raised:
                audit.compare(Path("base"), Path("checkpoint"))
        self.assertIn('"finite": false', str(raised.exception))


if __name__ == "__main__":
    unittest.main()
