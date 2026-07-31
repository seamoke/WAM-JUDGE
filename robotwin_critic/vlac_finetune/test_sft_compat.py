import unittest

from .sft_compat import (
    compatible_caching_allocator_warmup,
    compatible_internvl_loss_context,
    destroy_distributed_process_group,
)


class DummyModel:
    def __init__(self, tp_plan):
        self._tp_plan = tp_plan


class DummyDdp:
    def __init__(self, module):
        self.module = module


class DummyDistributed:
    def __init__(self, available=True, initialized=True):
        self.available = available
        self.initialized = initialized
        self.destroy_calls = 0

    def is_available(self):
        return self.available

    def is_initialized(self):
        return self.initialized

    def destroy_process_group(self):
        self.destroy_calls += 1


class SftCompatibilityTest(unittest.TestCase):
    def test_none_plan_uses_non_matching_sentinel_and_is_restored(self):
        observed = []

        def original(model, value):
            observed.append(list(model._tp_plan))
            return value

        model = DummyModel(None)
        wrapped = compatible_caching_allocator_warmup(original)
        self.assertEqual(wrapped(model, 7), 7)
        self.assertEqual(
            observed,
            [["__vlac_ddp_no_tensor_parallel_plan__"]],
        )
        self.assertIsNone(model._tp_plan)

    def test_existing_plan_is_unchanged(self):
        observed = []

        def original(model):
            observed.append(model._tp_plan)

        plan = ["model.layers.*.self_attn.q_proj"]
        model = DummyModel(plan)
        compatible_caching_allocator_warmup(original)(model)
        self.assertIs(observed[0], plan)
        self.assertIs(model._tp_plan, plan)

    def test_internvl_context_unwraps_ddp_model(self):
        underlying = object()
        observed = []

        def original(template, model, inputs):
            observed.append((template, model, inputs))
            return "context"

        template = object()
        inputs = {"input_ids": [1]}
        wrapped = compatible_internvl_loss_context(original)
        self.assertEqual(wrapped(template, DummyDdp(underlying), inputs), "context")
        self.assertEqual(observed, [(template, underlying, inputs)])

    def test_internvl_context_keeps_plain_model(self):
        model = object()

        def original(template, observed_model, inputs):
            return observed_model

        wrapped = compatible_internvl_loss_context(original)
        self.assertIs(wrapped(object(), model, {}), model)

    def test_destroy_initialized_process_group(self):
        dist = DummyDistributed()
        self.assertTrue(destroy_distributed_process_group(dist))
        self.assertEqual(dist.destroy_calls, 1)

    def test_skip_uninitialized_process_group(self):
        dist = DummyDistributed(initialized=False)
        self.assertFalse(destroy_distributed_process_group(dist))
        self.assertEqual(dist.destroy_calls, 0)


if __name__ == "__main__":
    unittest.main()
