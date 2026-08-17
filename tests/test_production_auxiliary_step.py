from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from robotwin_critic.two_stage_rft import production_auxiliary_step as auxiliary


SOURCE = Path(auxiliary.__file__).read_text(encoding="utf-8")


class _LocalOnlyGradient:
    def __init__(self, local: torch.Tensor) -> None:
        self.local = local
        self.to_local_calls = 0

    def to_local(self) -> torch.Tensor:
        self.to_local_calls += 1
        return self.local


def _trainer_with_gradients(*gradients: torch.Tensor | None):
    module = torch.nn.Module()
    for index, gradient in enumerate(gradients):
        parameter = torch.nn.Parameter(torch.ones(2))
        parameter.grad = gradient
        module.register_parameter(f"parameter_{index}", parameter)
    return SimpleNamespace(transformer=module, device=torch.device("cpu"))


def _trainer_with_named_gradients(named_gradients):
    module = torch.nn.Module()
    for name, gradient in named_gradients:
        parent = module
        parts = name.split(".")
        for part in parts[:-1]:
            if not hasattr(parent, part):
                parent.add_module(part, torch.nn.Module())
            parent = getattr(parent, part)
        parameter = torch.nn.Parameter(torch.ones(2))
        parameter.grad = gradient
        parent.register_parameter(parts[-1], parameter)
    return SimpleNamespace(transformer=module, device=torch.device("cpu"))


class ProductionAuxiliaryStepTest(unittest.TestCase):
    def test_finiteness_uses_tensor_or_rank_local_value(self) -> None:
        self.assertTrue(auxiliary._gradient_is_finite(torch.tensor([1.0, -2.0])))
        self.assertFalse(auxiliary._gradient_is_finite(torch.tensor([float("inf")])))

        finite = _LocalOnlyGradient(torch.tensor([3.0]))
        nonfinite = _LocalOnlyGradient(torch.tensor([float("nan")]))
        self.assertTrue(auxiliary._gradient_is_finite(finite))
        self.assertFalse(auxiliary._gradient_is_finite(nonfinite))
        self.assertEqual(finite.to_local_calls, 1)
        self.assertEqual(nonfinite.to_local_calls, 1)

    def test_source_forbids_isfinite_all_item_on_any_value(self) -> None:
        tree = ast.parse(SOURCE)
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "item":
                continue
            receiver = node.func.value
            if (
                isinstance(receiver, ast.Call)
                and isinstance(receiver.func, ast.Attribute)
                and receiver.func.attr == "all"
                and "isfinite" in ast.unparse(receiver)
            ):
                offenders.append(ast.unparse(node))
        self.assertEqual(offenders, [])
        self.assertEqual(SOURCE.count("dist.all_reduce(success"), 1)

    def test_validated_shards_accumulate_failures_before_one_collective(self) -> None:
        trainer = _trainer_with_gradients(None, torch.tensor([float("nan"), 0.0]))
        all_reduce = mock.Mock()
        with (
            mock.patch.object(auxiliary.dist, "is_available", return_value=True),
            mock.patch.object(auxiliary.dist, "is_initialized", return_value=True),
            mock.patch.object(auxiliary.dist, "all_reduce", all_reduce),
        ):
            with self.assertRaisesRegex(RuntimeError, "gradient validation failed"):
                auxiliary._validated_gradient_shards(trainer, "test boundary")
        all_reduce.assert_called_once()
        self.assertIs(all_reduce.call_args.kwargs["op"], auxiliary.dist.ReduceOp.MIN)
        self.assertNotIn("group", all_reduce.call_args.kwargs)

    def test_remote_rank_failure_raises_after_collective_on_locally_valid_rank(self) -> None:
        trainer = _trainer_with_gradients(torch.tensor([1.0, 2.0]))

        def inject_remote_missing(success, *, op):
            self.assertIs(op, auxiliary.dist.ReduceOp.MIN)
            success.fill_(0)

        with (
            mock.patch.object(auxiliary.dist, "is_available", return_value=True),
            mock.patch.object(auxiliary.dist, "is_initialized", return_value=True),
            mock.patch.object(
                auxiliary.dist, "all_reduce", side_effect=inject_remote_missing
            ) as all_reduce,
        ):
            with self.assertRaisesRegex(RuntimeError, "one or more ranks"):
                auxiliary._validated_gradient_shards(trainer, "test boundary")
        all_reduce.assert_called_once()

    def test_validated_shard_keeps_detached_clone(self) -> None:
        gradient = torch.tensor([1.0, 2.0])
        trainer = _trainer_with_gradients(gradient)
        shards = auxiliary._validated_gradient_shards(trainer, "test boundary")
        self.assertEqual(len(shards), 1)
        self.assertIsNot(shards[0][2], gradient)
        self.assertFalse(shards[0][2].requires_grad)
        torch.testing.assert_close(shards[0][2], gradient)

    def test_exact_official_unused_text_projection_may_have_no_gradient(self) -> None:
        trainer = _trainer_with_named_gradients(
            [
                (
                    "condition_embedder_action.text_embedder.linear_1.weight",
                    None,
                ),
                ("active.weight", torch.tensor([1.0, 2.0])),
            ]
        )
        shards = auxiliary._validated_gradient_shards(trainer, "real boundary")
        self.assertEqual([name for name, _, _ in shards], ["active.weight"])

    def test_similarly_named_missing_gradient_is_still_rejected(self) -> None:
        trainer = _trainer_with_named_gradients(
            [
                (
                    "condition_embedder_action.text_embedder.linear_3.weight",
                    None,
                ),
                ("active.weight", torch.tensor([1.0, 2.0])),
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "linear_3.weight"):
            auxiliary._validated_gradient_shards(trainer, "real boundary")


if __name__ == "__main__":
    unittest.main()
