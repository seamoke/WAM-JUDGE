import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from robotwin_critic.two_stage_rft.production_auxiliary_step import (
        _gradient_is_finite,
        _validated_gradient_shards,
    )


class _LocalShardWrapper:
    def __init__(self, local):
        self.local = local
        self.to_local_calls = 0

    def to_local(self):
        self.to_local_calls += 1
        return self.local


def _cpu_trainer(parameter):
    module = torch.nn.Module()
    module.register_parameter("weight", parameter)
    return SimpleNamespace(transformer=module, device=torch.device("cpu"))


class _NamedParameters:
    def __init__(self, parameters):
        self.parameters = parameters

    def named_parameters(self):
        return iter(self.parameters)


class ProductionAuxiliaryStepSourceTest(unittest.TestCase):
    def test_source_forbids_isfinite_all_item_chain(self):
        source = Path(__file__).with_name("production_auxiliary_step.py").read_text()
        forbidden = re.compile(r"torch\.isfinite\([^\n]*\)\.all\(\)\.item\(\)")
        self.assertIsNone(forbidden.search(source))

    def test_real_metric_snapshots_are_cloned_before_accumulator_reset(self):
        source = Path(__file__).with_name("production_auxiliary_step.py").read_text()
        for accumulator in ("_aux_real_latent", "_aux_real_action"):
            snapshot = re.compile(
                rf"dist_mean\(trainer\.{accumulator}\)\.detach\(\)\.clone\(\)"
            )
            self.assertRegex(source, snapshot)


@unittest.skipIf(torch is None, "torch unavailable")
class ProductionAuxiliaryStepTorchTest(unittest.TestCase):
    def test_finiteness_supports_tensor_and_to_local_wrapper(self):
        self.assertTrue(_gradient_is_finite(torch.tensor([1.0, -2.0])))
        self.assertFalse(_gradient_is_finite(torch.tensor([1.0, float("nan")])))

        wrapped = _LocalShardWrapper(torch.tensor([1.0, float("inf")]))
        self.assertFalse(_gradient_is_finite(wrapped))
        self.assertEqual(wrapped.to_local_calls, 1)

    def test_remote_rank_failure_makes_locally_valid_rank_raise(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        parameter.grad = torch.tensor([0.5, -0.5])
        trainer = _cpu_trainer(parameter)

        def report_remote_failure(success, op):
            self.assertEqual(op, torch.distributed.ReduceOp.MIN)
            self.assertEqual(success.dtype, torch.int32)
            self.assertEqual(success.device.type, "cpu")
            success.zero_()

        with (
            mock.patch.object(torch.distributed, "is_available", return_value=True),
            mock.patch.object(torch.distributed, "is_initialized", return_value=True),
            mock.patch.object(
                torch.distributed, "all_reduce", side_effect=report_remote_failure
            ) as all_reduce,
        ):
            with self.assertRaisesRegex(RuntimeError, "one or more ranks"):
                _validated_gradient_shards(trainer, "test boundary")

        all_reduce.assert_called_once()

    def test_missing_and_nonfinite_gradients_are_collected_before_raise(self):
        missing = SimpleNamespace(requires_grad=True, grad=None)
        wrapped = _LocalShardWrapper(torch.tensor([float("nan")]))
        nonfinite = SimpleNamespace(requires_grad=True, grad=wrapped)
        trainer = SimpleNamespace(
            transformer=_NamedParameters(
                [("missing", missing), ("nonfinite", nonfinite)]
            ),
            device=torch.device("cpu"),
        )

        with self.assertRaises(RuntimeError) as raised:
            _validated_gradient_shards(trainer, "test boundary")

        self.assertIn("missing", str(raised.exception))
        self.assertIn("nonfinite", str(raised.exception))
        self.assertEqual(wrapped.to_local_calls, 1)

    def test_valid_tensor_gradient_is_detached_cloned_after_one_agreement(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        parameter.grad = torch.tensor([0.5, -0.5])
        trainer = _cpu_trainer(parameter)

        with (
            mock.patch.object(torch.distributed, "is_available", return_value=True),
            mock.patch.object(torch.distributed, "is_initialized", return_value=True),
            mock.patch.object(torch.distributed, "all_reduce") as all_reduce,
        ):
            shards = _validated_gradient_shards(trainer, "test boundary")

        all_reduce.assert_called_once()
        self.assertEqual(len(shards), 1)
        self.assertEqual(shards[0][0], "weight")
        self.assertIsNot(shards[0][2], parameter.grad)
        self.assertFalse(shards[0][2].requires_grad)
        torch.testing.assert_close(shards[0][2], parameter.grad)
