"""Torch-free focused tests for joint RFT scheduling."""

import unittest
import tempfile
from pathlib import Path

from robotwin_critic.two_stage_rft.rft_schedule import (
    auxiliary_update_plan,
    auxiliary_schedule_report,
    effective_pseudo_loss_weight,
    expected_sample_exposure,
    next_sampler_batch,
    resolve_base_auxiliary_steps,
    resolve_optimizer_steps,
    should_rebuild_lr_scheduler,
    validate_base_auxiliary_contract,
)


class RFTScheduleTest(unittest.TestCase):
    def test_pseudo_loss_weight_linear_warmup(self) -> None:
        values = [
            effective_pseudo_loss_weight(
                target_weight=0.25, warmup_steps=3000, optimizer_step=step
            )
            for step in (0, 1499, 2999, 3000)
        ]
        self.assertEqual(values, [0.25 / 3000, 0.125, 0.25, 0.25])
        self.assertEqual(effective_pseudo_loss_weight(
            target_weight=0.25, warmup_steps=0, optimizer_step=0,
        ), 0.25)

    def test_pseudo_loss_weight_warmup_rejects_negative_values(self) -> None:
        for kwargs in (
            {"target_weight": -0.1, "warmup_steps": 1, "optimizer_step": 0},
            {"target_weight": 0.25, "warmup_steps": -1, "optimizer_step": 0},
            {"target_weight": 0.25, "warmup_steps": 1, "optimizer_step": -1},
        ):
            with self.assertRaisesRegex(ValueError, "non-negative"):
                effective_pseudo_loss_weight(**kwargs)

    def test_auxiliary_schedule_is_fixed_and_exactly_scaled(self) -> None:
        steps = resolve_optimizer_steps(
            schedule_mode="base-auxiliary-pseudo", configured_steps=15000,
            num_epochs=0, loader_microbatches=23465,
            gradient_accumulation_steps=8, real_items=2988,
            pseudo_items=20477,
        )
        self.assertEqual(steps, 15000)
        report = auxiliary_schedule_report(
            optimizer_steps=steps, world_size=8, real_global_batch=64,
            pseudo_global_batch=8, pseudo_loss_weight=0.25,
        )
        self.assertEqual(report["real_microbatches_per_rank_update"], 8)
        self.assertEqual(report["pseudo_microbatches_per_rank_update"], 1)
        self.assertEqual(report["pseudo_backward_scale"], 2.0)
        self.assertEqual(report["expected_real_sample_draws"], 960000)
        self.assertEqual(report["expected_pseudo_sample_draws"], 120000)
        self.assertEqual(
            report["fsdp_sync"],
            "final_real_backward_then_independent_final_pseudo_backward",
        )
        plan = auxiliary_update_plan(
            real_microbatches=8,
            pseudo_microbatches=1,
            pseudo_loss_weight=0.25,
        )
        self.assertEqual(plan["real_sync_flags"], [False] * 7 + [True])
        self.assertEqual(plan["pseudo_sync_flags"], [True])

    def test_zero_weight_draws_no_pseudo_and_syncs_final_real(self) -> None:
        plan = auxiliary_update_plan(
            real_microbatches=16,
            pseudo_microbatches=2,
            pseudo_loss_weight=0,
        )
        self.assertFalse(plan["pseudo_enabled"])
        self.assertEqual(plan["pseudo_draws_per_rank_update"], 0)
        self.assertEqual(plan["pseudo_sync_flags"], [])
        self.assertEqual(plan["real_sync_flags"], [False] * 15 + [True])
        report = auxiliary_schedule_report(
            optimizer_steps=15000, world_size=4,
            pseudo_global_batch=8, pseudo_loss_weight=0,
        )
        self.assertEqual(report["expected_pseudo_sample_draws"], 0)
        self.assertEqual(report["fsdp_sync"], "final_real_backward_only")

    def test_old_uniform_union_mode_is_rejected_for_runtime_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported schedule_mode"):
            resolve_optimizer_steps(
                schedule_mode="base-real-exposure", configured_steps=15000,
                num_epochs=0, loader_microbatches=23465,
                gradient_accumulation_steps=8, real_items=2988,
                pseudo_items=20477,
            )

    def test_pseudo_sampler_epoch_advances_on_exhaustion(self) -> None:
        class Sampler:
            def __init__(self):
                self.epochs = []

            def set_epoch(self, epoch):
                self.epochs.append(epoch)

        sampler = Sampler()
        loader = [["epoch-batch"]]
        batch, iterator, epoch = next_sampler_batch(loader, None, sampler, 0)
        self.assertEqual(batch, ["epoch-batch"])
        self.assertEqual(epoch, 0)
        batch, iterator, epoch = next_sampler_batch(
            loader, iterator, sampler, epoch
        )
        self.assertEqual(batch, ["epoch-batch"])
        self.assertEqual(epoch, 1)
        self.assertEqual(sampler.epochs, [0, 1])

    def test_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive and divisible"):
            auxiliary_schedule_report(
                optimizer_steps=15000, world_size=4, pseudo_global_batch=6
            )
        with self.assertRaisesRegex(ValueError, "divisible"):
            resolve_optimizer_steps(
                schedule_mode="epochs", configured_steps=1, num_epochs=3,
                loader_microbatches=5, gradient_accumulation_steps=2,
                real_items=1, pseudo_items=1,
            )
        with tempfile.TemporaryDirectory() as directory:
            prepared = Path(directory) / "prepared"
            stage1 = prepared / "stage1"
            stage1.mkdir(parents=True)
            values = dict(
                mixing_mode="auxiliary", real_chunk_mode="full", batch_size_per_rank=1,
                global_batch_size=64, activation_checkpointing=False,
                warmup_steps=10, scheduler_type="constant",
                max_episode_frames=1_000_000_000, num_epochs=0,
                real_data_mode="stage1", real_data_root=stage1,
                prepared_data_root=prepared,
            )
            validate_base_auxiliary_contract(**values)
            values["activation_checkpointing"] = True
            with self.assertRaisesRegex(ValueError, "activation_checkpointing"):
                validate_base_auxiliary_contract(**values)
            validate_base_auxiliary_contract(
                **values, required_activation_checkpointing=True
            )
            values["activation_checkpointing"] = False
            values["real_data_mode"] = "stage1-stage2-visible"
            values["real_data_root"] = prepared / "action_visible_real"
            with self.assertRaisesRegex(ValueError, "real_data_mode.*real_data_root"):
                validate_base_auxiliary_contract(**values)

    def test_real_only_regression_step_allowance(self) -> None:
        self.assertEqual(resolve_base_auxiliary_steps(500, smoke_steps=500), (500, [500]))
        with self.assertRaisesRegex(ValueError, "between 1 and 500"):
            resolve_base_auxiliary_steps(501, smoke_steps=501)
        self.assertEqual(resolve_base_auxiliary_steps(2000, regression_steps=2000), (2000, [2000]))
        with self.assertRaisesRegex(ValueError, "between 1 and 2000"):
            resolve_base_auxiliary_steps(2001, regression_steps=2001)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            resolve_base_auxiliary_steps(10, smoke_steps=10, regression_steps=10)

    def test_arbitrary_production_steps_preserve_save_schedule(self) -> None:
        self.assertEqual(
            resolve_base_auxiliary_steps(16000, configured_save_steps=list(range(2000, 16001, 2000))),
            (16000, list(range(2000, 16001, 2000))),
        )
        self.assertEqual(resolve_base_auxiliary_steps(123), (123, [123]))
        with self.assertRaisesRegex(ValueError, "must be positive"):
            resolve_base_auxiliary_steps(0)
        with self.assertRaisesRegex(ValueError, "no greater"):
            resolve_base_auxiliary_steps(100, configured_save_steps=[101])

    def test_non_base_reporting_has_no_base_guarantee(self) -> None:
        report = expected_sample_exposure(
            real_items=10, pseudo_items=5, optimizer_steps=3,
            global_batch_size=2, base_reference_steps=3,
            schedule_mode="epochs",
        )
        self.assertNotIn("ceil", report["optimizer_step_formula"])
        self.assertIn("No Base", report["exposure_guarantee"])
        self.assertNotIn("base_reference_real_draws", report)
        self.assertNotIn("expected_real_draw_ratio", report)

    def test_scheduler_rebuild_decision(self) -> None:
        cases = (
            ("steps", "constant", 15000, 15000, False),
            ("base-auxiliary-pseudo", "constant", 15000, 15000, False),
            ("epochs", "constant", 1, 5000, False),
            ("epochs", "cosine", 1, 5000, True),
        )
        for mode, scheduler, before, after, expected in cases:
            self.assertEqual(should_rebuild_lr_scheduler(
                schedule_mode=mode, scheduler_type=scheduler,
                trainer_steps=before, final_steps=after,
            ), expected)


if __name__ == "__main__":
    unittest.main()
