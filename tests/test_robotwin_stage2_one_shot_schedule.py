from __future__ import annotations

import re
import os
import subprocess
import unittest
import json
import hashlib
import tempfile
from pathlib import Path

from robotwin_critic.two_stage_rft.rft_schedule import (
    auxiliary_schedule_report,
    model_identity,
    resolve_optimizer_steps,
    resolve_base_auxiliary_steps,
    should_rebuild_lr_scheduler,
    valid_base_auxiliary_completion,
)


ROOT = Path(__file__).resolve().parents[1]


class OneShotScheduleStaticTest(unittest.TestCase):
    def test_base_auxiliary_steps_and_smoke_override(self) -> None:
        steps, save_steps = resolve_base_auxiliary_steps(15000)
        self.assertEqual((steps, save_steps), (15000, [3000, 6000, 9000, 12000, 15000]))
        self.assertEqual(resolve_base_auxiliary_steps(16000), (16000, [16000]))
        for configured, allowance in ((10, 9), (10, 101), (15000, 10)):
            with self.subTest(configured=configured, allowance=allowance):
                with self.assertRaises(ValueError):
                    resolve_base_auxiliary_steps(configured, allowance)
        self.assertEqual(resolve_base_auxiliary_steps(10, 10), (10, [10]))

    def test_launcher_requires_explicit_valid_smoke_environment(self) -> None:
        launcher = (ROOT / "script" / "run_robotwin_joint_rft.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('"${RFT_SMOKE_MODE:-}" != "1"', launcher)
        self.assertIn('10#$RFT_BASE_AUXILIARY_SMOKE_STEPS > 100', launcher)
        self.assertIn('NUM_STEPS="$BASE_AUXILIARY_SMOKE_STEPS"', launcher)
        self.assertIn('SAVE_INTERVAL="$BASE_AUXILIARY_SMOKE_STEPS"', launcher)
        self.assertIn('--base-auxiliary-smoke-steps "$BASE_AUXILIARY_SMOKE_STEPS"', launcher)
        self.assertIn('echo "smoke_mode=$SMOKE_MODE"', launcher)
        self.assertIn('echo "smoke_steps=$BASE_AUXILIARY_SMOKE_STEPS"', launcher)

        launcher_path = ROOT / "script" / "run_robotwin_joint_rft.sh"
        cases = (
            ({"RFT_BASE_AUXILIARY_SMOKE_STEPS": "10"}, "accepted only"),
            ({"RFT_SMOKE_MODE": "1", "RFT_BASE_AUXILIARY_SMOKE_STEPS": "abc"}, "1 through 100"),
            ({"RFT_SMOKE_MODE": "1", "RFT_BASE_AUXILIARY_SMOKE_STEPS": "0"}, "1 through 100"),
            ({"RFT_SMOKE_MODE": "1", "RFT_BASE_AUXILIARY_SMOKE_STEPS": "101"}, "1 through 100"),
        )
        for overrides, error in cases:
            with self.subTest(overrides=overrides):
                environment = {
                    **os.environ,
                    "STAGE1_CHECKPOINT": "/unused/initializer",
                    "RFT_SCHEDULE_MODE": "base-auxiliary-pseudo",
                    **overrides,
                }
                result = subprocess.run(
                    ["bash", str(launcher_path)],
                    env=environment,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(error, result.stderr)

    def test_h200_configs_are_base_aligned(self) -> None:
        expected = {
            "robotwin_stage2_one_shot_2xh200.env": "32",
            "robotwin_stage2_one_shot_4xh200.env": "16",
        }
        for name, accumulation in expected.items():
            text = (ROOT / "script" / "configs" / name).read_text(encoding="utf-8")
            self.assertIn("ONE_SHOT_TRAIN_SCHEDULE=base-auxiliary-pseudo", text)
            self.assertIn("ONE_SHOT_BASE_REFERENCE_STEPS=15000", text)
            self.assertIn("PSEUDO_GLOBAL_BATCH=8", text)
            self.assertIn("PSEUDO_LOSS_WEIGHT=0.25", text)
            self.assertIn("PSEUDO_LOSS_WARMUP_STEPS=3000", text)
            self.assertIn("PSEUDO_SAMPLER_SEED=43", text)
            self.assertIn("REAL_DATA_MODE=stage1", text)
            self.assertRegex(text, r"(?m)^REAL_DATA_ROOT=.*/stage1$")
            self.assertIn("TRAIN_BATCH_SIZE_PER_GPU=1", text)
            self.assertIn(f"GRADIENT_ACCUMULATION_STEPS={accumulation}", text)
            self.assertIn("TRAIN_GLOBAL_BATCH=64", text)
            self.assertIn("TRAIN_ACTIVATION_CHECKPOINTING=1", text)
            self.assertIn("ONE_SHOT_WARMUP_STEPS=10", text)
            self.assertRegex(text, r"(?m)^RFT_INITIAL_MODEL=.*/lingbot-va-base$")
            self.assertRegex(text, r"(?m)^BASE_MODEL=.*/lingbot-va-posttrain-robotwin$")
            self.assertRegex(text, r"(?m)^ACTION_GATE_POLICY=strict$")
            self.assertRegex(text, r"(?m)^ACTION_WORKSPACE_SCOPE=task$")
            self.assertNotIn("MAX_EPISODE_FRAMES=", text)

    def test_action_collection_defaults_are_strict_task_without_forcing(self) -> None:
        online = (ROOT / "script" / "run_robotwin_online_dual_rft.sh").read_text(
            encoding="utf-8"
        )
        pipeline = (
            ROOT / "script" / "run_robotwin_stage2_online_rft_pipeline.sh"
        ).read_text(encoding="utf-8")
        example = (ROOT / "script" / "robotwin_stage2_online_rft.env.example").read_text(
            encoding="utf-8"
        )
        for text in (online, pipeline):
            self.assertIn('ACTION_GATE_POLICY="${ACTION_GATE_POLICY:-strict}"', text)
            self.assertIn('ACTION_WORKSPACE_SCOPE="${ACTION_WORKSPACE_SCOPE:-task}"', text)
            self.assertNotRegex(text, r"export ACTION_GATE_POLICY=(?:score_with_safety_gates|strict)")
            self.assertNotRegex(text, r"export ACTION_WORKSPACE_SCOPE=(?:global|task)")
        self.assertRegex(example, r"(?m)^ACTION_GATE_POLICY=strict$")
        self.assertRegex(example, r"(?m)^ACTION_WORKSPACE_SCOPE=task$")

    def test_launcher_encodes_base_invariants_and_auxiliary_objective(self) -> None:
        launcher = (ROOT / "script" / "run_robotwin_joint_rft.sh").read_text(
            encoding="utf-8"
        )
        self.assertRegex(
            launcher,
            r'(?m)^SCHEDULE_MODE="\$\{RFT_SCHEDULE_MODE:-steps\}"$',
        )
        self.assertRegex(
            launcher, r'(?m)^NUM_STEPS="\$\{RFT_NUM_STEPS:-3000\}"$'
        )
        for assignment in (
            "TRAIN_BATCH_SIZE_PER_GPU=1", "TARGET_GLOBAL_BATCH=64",
            "REAL_CHUNK_MODE=full",
            "ACTIVATION_CHECKPOINTING=1", "WARMUP_STEPS=10",
            "LR_SCHEDULER=constant", "MAX_EPISODE_FRAMES=1000000000",
        ):
            self.assertIn(assignment, launcher)
        self.assertRegex(launcher, re.escape("GRADIENT_ACCUMULATION_STEPS=$((64 / WORLD_SIZE))"))
        online = (ROOT / "script" / "run_robotwin_online_dual_rft.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("ONE_SHOT_TRAIN_SCHEDULE:-base-auxiliary-pseudo", online)
        self.assertIn('RFT_SCHEDULE_MODE="$ONE_SHOT_TRAIN_SCHEDULE"', online)
        self.assertIn("Explicit REAL_DATA_ROOT=", online)
        self.assertIn("required_real_root", launcher)
        pipeline = (ROOT / "script" / "run_robotwin_stage2_online_rft_pipeline.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("REAL_DATA_ROOT_WAS_SET=1", pipeline)
        self.assertIn("Explicit REAL_DATA_ROOT=", pipeline)
        self.assertIn(
            'RFT_INITIAL_MODEL="${RFT_INITIAL_MODEL:-$LINGBOT_ROOT/models/lingbot-va-base}"',
            pipeline,
        )
        self.assertIn("base-auxiliary-pseudo requires ONE_SHOT_MODE=1 or --one-shot", pipeline)
        self.assertIn("ONE_SHOT_TRAIN_SCHEDULE=steps explicitly", pipeline)
        one_shot_function = online.split("run_one_shot_training() {", 1)[1]
        one_shot_call = one_shot_function.split("  if ! env \\\n", 1)[1].split(
            '  RUN_ID="one_shot_fraction_', 1
        )[0]
        self.assertIn('  MIXING_MODE="$effective_mixing_mode" \\\n', one_shot_call)
        self.assertIn('  PSEUDO_GLOBAL_BATCH="$PSEUDO_GLOBAL_BATCH" \\\n', one_shot_call)
        self.assertIn('  STAGE1_CHECKPOINT="$RFT_INITIAL_MODEL" \\\n', one_shot_call)
        self.assertNotIn('  STAGE1_CHECKPOINT="$INITIAL_MODEL" \\\n', one_shot_call)
        self.assertIn("wam_collection_model=$INITIAL_MODEL", online)
        self.assertIn("rft_initial_model=$RFT_INITIAL_MODEL", online)
        base_branch = online.split(
            'if [[ "$ONE_SHOT_TRAIN_SCHEDULE" == "base-auxiliary-pseudo" ]]; then',
            1,
        )[1].split("else", 1)[0]
        self.assertIn("effective_real_chunk_mode=full", base_branch)
        self.assertNotIn("all-transitions", base_branch)
        self.assertIn("effective_max_episode_frames=1000000000", online)
        self.assertIn("effective_num_steps=15000", online)
        self.assertIn("effective_save_interval=3000", online)
        self.assertIn("effective_mixing_mode=auxiliary", online)
        self.assertIn(
            "pseudo_loss_warmup_steps=$PSEUDO_LOSS_WARMUP_STEPS one_shot_full_target=",
            online,
        )
        self.assertIn('"exposure_report": report', online)
        self.assertIn('"base_reference_steps": int(sys.argv[7])', online)
        self.assertIn('"pseudo_loss_weight_target": float(sys.argv[17])', online)
        self.assertIn('"pseudo_loss_warmup_steps": int(sys.argv[18])', online)
        self.assertIn('report=json.load(open(sys.argv[19]))', online)
        self.assertIn(
            '"$PSEUDO_LOSS_WEIGHT" "$PSEUDO_LOSS_WARMUP_STEPS"', online
        )
        self.assertIn('if [[ "$SCHEDULE_MODE" != "steps" ]]', launcher)
        self.assertIn('["optimizer_steps"]', launcher)
        trainer = (
            ROOT / "robotwin_critic" / "two_stage_rft" / "train_joint_rft.py"
        ).read_text(encoding="utf-8")
        auxiliary_step = (
            ROOT
            / "robotwin_critic"
            / "two_stage_rft"
            / "production_auxiliary_step.py"
        ).read_text(encoding="utf-8")
        self.assertIn("DistributedSampler(\n                    pseudo", trainer)
        self.assertIn("next_sampler_batch(", trainer)
        self.assertIn(
            "return production_auxiliary_train_step(self, batch, batch_idx)", trainer
        )
        self.assertIn("torch.random.fork_rng", auxiliary_step)
        self.assertIn(
            'trainer.auxiliary_update_plan["real_sync_flags"][batch_idx]',
            auxiliary_step,
        )
        self.assertIn(
            'trainer.auxiliary_update_plan["pseudo_sync_flags"]', auxiliary_step
        )
        self.assertIn("* effective_pseudo_weight", auxiliary_step)
        self.assertIn("--pseudo-loss-warmup-steps", trainer)
        self.assertIn('PSEUDO_LOSS_WARMUP_STEPS="$PSEUDO_LOSS_WARMUP_STEPS"', online)
        schedule = (
            ROOT / "robotwin_critic" / "two_stage_rft" / "rft_schedule.py"
        ).read_text(encoding="utf-8")
        self.assertIn("base auxiliary optimizer steps must be positive", schedule)
        self.assertIn("resolve_base_auxiliary_steps(", trainer)
        self.assertIn("self.config.save_steps = auxiliary_save_steps", trainer)
        self.assertIn("self.config.save_interval = self.config.save_steps[0]", trainer)
        self.assertIn("SAVE_INTERVAL=3000", launcher)
        self.assertIn('echo "save_steps=$SAVE_STEPS"', launcher)
        self.assertIn(
            "gradient_objective=mean_64_real_plus_linear_warmup_to_"
            "${PSEUDO_LOSS_WEIGHT}_target_coefficient_times_mean_"
            "${PSEUDO_GLOBAL_BATCH}_pseudo",
            launcher,
        )

    def test_full_entrypoints_force_base_auxiliary_15000(self) -> None:
        for name in (
            "run_robotwin_stage2_full.sh",
            "run_robotwin_complete_experiment.sh",
        ):
            text = (ROOT / "script" / name).read_text(encoding="utf-8")
            self.assertIn("RFT_SCHEDULE_MODE=base-auxiliary-pseudo", text)
            self.assertIn("RFT_NUM_STEPS=15000", text)
            self.assertIn("RFT_SAVE_INTERVAL=3000", text)
            self.assertIn("REAL_DATA_MODE=stage1", text)
            self.assertIn('REAL_DATA_ROOT="$PREPARED_DATA_ROOT/stage1"', text)
            self.assertIn("TRAIN_BATCH_SIZE_PER_GPU=1", text)
            self.assertIn("TARGET_GLOBAL_BATCH=64", text)
            self.assertIn("GRADIENT_ACCUMULATION_STEPS=16", text)
            self.assertIn("ACTIVATION_CHECKPOINTING=1", text)
            self.assertIn("WARMUP_STEPS=10", text)
            self.assertNotIn("RFT_NUM_STEPS=3000", text)
            self.assertNotRegex(text, r"joint(?:_rft)?_3000")

        full = (ROOT / "script" / "run_robotwin_stage2_full.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'WAM_MODEL="${WAM_MODEL:-${STAGE1_CHECKPOINT:-$LINGBOT_ROOT/models/lingbot-va-posttrain-robotwin}}"',
            full,
        )
        self.assertIn(
            'RFT_INITIAL_MODEL="${RFT_INITIAL_MODEL:-$LINGBOT_ROOT/models/lingbot-va-base}"',
            full,
        )
        self.assertIn('STAGE1_CHECKPOINT="$WAM_MODEL"', full)
        self.assertIn('STAGE1_CHECKPOINT="$RFT_INITIAL_MODEL"', full)
        self.assertIn('test -s "$WAM_MODEL/transformer/config.json"', full)
        self.assertIn('test -s "$RFT_INITIAL_MODEL/transformer/config.json"', full)
        self.assertIn("wam_collection_model=$WAM_MODEL", full)
        self.assertIn("rft_training_initializer=$RFT_INITIAL_MODEL", full)

    def test_ten_percent_is_reported_as_pseudo_only(self) -> None:
        config = (
            ROOT / "script" / "configs" /
            "robotwin_stage2_one_shot_4xh200_10pct.env"
        ).read_text(encoding="utf-8")
        self.assertIn("Pseudo collection/finalization fraction only", config)
        online = (ROOT / "script" / "run_robotwin_online_dual_rft.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("real_data_fraction=1.0 pseudo_data_fraction=", online)
        self.assertIn(
            '[[ "$ONE_SHOT_TRAIN_SCHEDULE" == "base-auxiliary-pseudo" ]] && printf 1.0',
            online,
        )

    def test_base_auxiliary_uses_exact_stage1_checkpoint_cadence(self) -> None:
        trainer = (
            ROOT / "robotwin_critic" / "two_stage_rft" / "train_joint_rft.py"
        ).read_text(encoding="utf-8")
        auxiliary = trainer.split("if self.auxiliary_mode:", 2)[2].split(
            "else:", 1
        )[0]
        self.assertIn("self.config.save_steps = auxiliary_save_steps", auxiliary)
        self.assertIn("self.config.save_interval = self.config.save_steps[0]", auxiliary)
        self.assertNotIn("self.config.save_steps = [final_steps]", auxiliary)

        docs = (ROOT / "docs" / "robotwin_stage2_one_shot_rft.md").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "checkpoints at steps 3,000, 6,000, 9,000, 12,000, and 15,000",
            docs,
        )

    def test_report_outputs_are_not_duplicated(self) -> None:
        trainer = (
            ROOT / "robotwin_critic" / "two_stage_rft" / "train_joint_rft.py"
        ).read_text(encoding="utf-8")
        self.assertEqual(trainer.count('"rft_source_counts.json"'), 1)
        self.assertEqual(trainer.count('"RFT exposure: %s"'), 1)

    def test_external_pseudo_mode_has_precedence_and_no_collection_fallback(self) -> None:
        online = (ROOT / "script" / "run_robotwin_online_dual_rft.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('ONE_SHOT_PSEUDO_JSONL="${ONE_SHOT_PSEUDO_JSONL:-}"', online)
        self.assertIn(
            'ONE_SHOT_BUFFER="$(PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" '
            '"$WAM_PYTHON" \\\n'
            '    -m '
            'robotwin_critic.two_stage_rft.subset_external_pseudo \\\n'
            '    --source "$ONE_SHOT_PSEUDO_JSONL" --output-root "$ONLINE_ROOT" \\\n'
            '    --fraction "$ONE_SHOT_DATA_FRACTION" --seed "$BASE_SEED")"',
            online,
        )
        external_branch = online.split(
            'if [[ "$ONE_SHOT_SOURCE_MODE" == "external" ]]; then', 1
        )[1].split(
            'if [[ "$ONE_SHOT_MODE" == "1" && -d "$ONE_SHOT_COLLECT_ROOT" ]]',
            1,
        )[0]
        self.assertEqual(external_branch.count("run_one_shot_training"), 1)
        self.assertIn("exit 0", external_branch)
        for forbidden in (
            "refresh_one_shot_buffer",
            "update_one_shot_plateau",
            "online_iteration",
            "collect_index",
            "VLAC",
            "ACTION_PROFILE",
        ):
            self.assertNotIn(forbidden, external_branch)

        self.assertRegex(
            online,
            re.compile(
                r'if \[\[ "\$ONE_SHOT_SOURCE_MODE" == "collection" \]\]; then\n'
                r'(?:(?!\nfi\n).)*online_iteration init'
                r'(?:(?!\nfi\n).)*log_collection_swanlab\nfi',
                re.DOTALL,
            ),
        )

    def test_pipeline_passes_external_jsonl_without_finalizing_it(self) -> None:
        pipeline = (
            ROOT / "script" / "run_robotwin_stage2_online_rft_pipeline.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--one-shot-pseudo-jsonl", pipeline)
        self.assertIn('ONE_SHOT_PSEUDO_JSONL="$(realpath', pipeline)
        self.assertIn('parent_args+=(--one-shot-pseudo-jsonl', pipeline)
        self.assertIn(
            '[[ -z "$ONE_SHOT_PSEUDO_JSONL" && ! -s "$ACTION_PROFILE" ]]',
            pipeline,
        )
        self.assertIn(
            '[[ -z "$ONE_SHOT_PSEUDO_JSONL" && ! -s "$CONTEXTS" ]]', pipeline
        )
        self.assertIn(
            '[[ -z "$ONE_SHOT_PSEUDO_JSONL" && ! -s "$BUDGET" ]]', pipeline
        )

    def test_modelscope_pseudo_package_is_downloaded_as_dataset(self) -> None:
        docs = (ROOT / "docs" / "robotwin_stage2_one_shot_rft.md").read_text(
            encoding="utf-8"
        )
        package = "seamoke/robotwin-stage2-oneshot-pseudo-chunks"
        self.assertIn(f"--dataset {package}", docs)
        self.assertNotIn(f"--model {package}", docs)

    def test_data_scale_usage_has_no_stale_transition_counts(self) -> None:
        docs = (ROOT / "docs" / "robotwin_stage2_one_shot_rft.md").read_text(
            encoding="utf-8"
        )
        usage = docs.split("## Data Scale", 1)[1].split(
            "### Train from the published ModelScope pseudo package", 1
        )[0]
        self.assertNotIn("60,219", usage)
        self.assertNotIn("6,022", usage)
        self.assertIn("`ONE_SHOT_DATA_FRACTION=1.0` reuses the original", usage)
        self.assertIn("A fraction below `1.0` creates a deterministic subset", usage)
        self.assertIn("safely resolved against the source JSONL directory", usage)

    def test_low_level_schedule_logging_is_mode_specific(self) -> None:
        launcher = (ROOT / "script" / "run_robotwin_joint_rft.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('case "$SCHEDULE_MODE" in', launcher)
        reporting = launcher.split('case "$SCHEDULE_MODE" in', 2)[2].split(
            "  esac", 1
        )[0]
        base_report = reporting.split("base-auxiliary-pseudo)", 1)[1].split(
            "      ;;", 1
        )[0]
        steps_report = reporting.split("steps)", 1)[1].split("      ;;", 1)[0]
        epochs_report = reporting.split("epochs)", 1)[1].split("      ;;", 1)[0]
        self.assertIn("gradient_objective=mean_64_real", base_report)
        self.assertIn("base_real_boundary_then_separate_pseudo_boundary", base_report)
        self.assertNotIn("real_exposure_formula", steps_report)
        self.assertIn("configured_optimizer_steps_no_base_exposure_guarantee", steps_report)
        self.assertNotIn("real_exposure_formula", epochs_report)
        self.assertIn("exact_union_epochs_no_base_exposure_guarantee", epochs_report)

    def test_low_level_initializer_wording_is_model_role_neutral(self) -> None:
        launcher = (ROOT / "script" / "run_robotwin_joint_rft.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("Set STAGE1_CHECKPOINT to the RFT training initializer", launcher)
        self.assertIn('echo "rft_training_initializer=$STAGE1_CHECKPOINT"', launcher)
        self.assertNotIn("Set STAGE1_CHECKPOINT to M30 checkpoint_step_15000", launcher)
        self.assertNotIn('echo "stage1_checkpoint=$STAGE1_CHECKPOINT"', launcher)

    def test_runtime_counts_use_fixed_auxiliary_exposure(self) -> None:
        steps = resolve_optimizer_steps(
            schedule_mode="base-auxiliary-pseudo", configured_steps=15000,
            num_epochs=0, loader_microbatches=23465,
            gradient_accumulation_steps=8, real_items=2988,
            pseudo_items=20477,
        )
        self.assertEqual(steps, 15000)
        report = auxiliary_schedule_report(
            optimizer_steps=steps, world_size=8,
            pseudo_global_batch=8, pseudo_loss_weight=0.25,
        )
        self.assertEqual(report["expected_real_sample_draws"], 960000)
        self.assertEqual(report["expected_pseudo_sample_draws"], 120000)
        self.assertEqual(report["pseudo_backward_scale"], 2.0)
        self.assertEqual(
            report["fsdp_sync"],
            "final_real_backward_then_independent_final_pseudo_backward",
        )
        docs = (ROOT / "docs" / "robotwin_stage2_one_shot_rft.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("R=2988", docs)
        self.assertIn("P=20477", docs)
        self.assertIn("117,797 steps", docs)
        self.assertIn("87.3% pseudo draws", docs)

    def test_completion_marker_requires_artifacts_and_exact_exposure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final_model = root / "final_model"
            (final_model / "transformer").mkdir(parents=True)
            (final_model / "transformer" / "config.json").write_text("{}\n")
            weights = final_model / "transformer" / "diffusion_pytorch_model.safetensors"
            weights.write_bytes(b"official weights")
            pseudo = root / "pseudo.jsonl"
            pseudo.write_bytes(b'{"sample": 1}\n')
            initializer = root / "initializer"
            (initializer / "transformer").mkdir(parents=True)
            initializer_config = initializer / "transformer" / "config.json"
            initializer_config.write_bytes(b"{}\n")
            marker = root / "complete.json"
            payload = {
                "complete": True,
                "schedule": "base-auxiliary-pseudo",
                "optimizer_steps": 15000,
                "final_model": str(final_model),
                "pseudo_loss_weight_target": 0.25,
                "pseudo_loss_warmup_steps": 3000,
                "pseudo_sampler_seed": 43,
                "pseudo_artifact": {
                    "canonical_path": str(pseudo.resolve()),
                    "sha256": hashlib.sha256(pseudo.read_bytes()).hexdigest(),
                },
                "initializer_identity": {
                    "canonical_path": str(initializer.resolve()),
                    "transformer_config_sha256": hashlib.sha256(
                        initializer_config.read_bytes()
                    ).hexdigest(),
                    "transformer_weight_sha256": {},
                },
                "final_model_identity": model_identity(final_model),
                "exposure_report": {
                    "optimizer_steps": 15000,
                    "pseudo_global_batch": 8,
                    "observed_real_samples": 960000,
                    "observed_pseudo_samples": 120000,
                },
            }
            marker.write_text(json.dumps(payload), encoding="utf-8")
            expected = dict(
                pseudo_path=pseudo, pseudo_sampler_seed=43, target_weight=0.25,
                warmup_steps=3000, schedule="base-auxiliary-pseudo",
                initializer=initializer,
            )
            self.assertTrue(valid_base_auxiliary_completion(marker, **expected))
            smoke_payload = json.loads(marker.read_text(encoding="utf-8"))
            smoke_payload["optimizer_steps"] = 10
            smoke_payload["exposure_report"]["optimizer_steps"] = 10
            marker.write_text(json.dumps(smoke_payload), encoding="utf-8")
            self.assertFalse(valid_base_auxiliary_completion(marker, **expected))
            marker.write_text(json.dumps(payload), encoding="utf-8")
            weights.write_bytes(b"replaced but structurally complete weights")
            self.assertFalse(valid_base_auxiliary_completion(marker, **expected))
            weights.write_bytes(b"official weights")
            pseudo.write_bytes(b'{"sample": 2}\n')
            self.assertFalse(valid_base_auxiliary_completion(marker, **expected))
            pseudo.write_bytes(b'{"sample": 1}\n')
            self.assertFalse(valid_base_auxiliary_completion(
                marker, **{**expected, "pseudo_sampler_seed": 44}
            ))
            weights.write_bytes(b"")
            self.assertFalse(valid_base_auxiliary_completion(marker, **expected))
            weights.write_bytes(b"official weights")
            payload["exposure_report"]["observed_real_samples"] -= 1
            marker.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(valid_base_auxiliary_completion(marker, **expected))
            payload["exposure_report"]["observed_real_samples"] = 960000
            payload["complete"] = False
            marker.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(valid_base_auxiliary_completion(marker, **expected))

    def test_collection_resume_routes_completion_through_validator(self) -> None:
        launcher = (ROOT / "script" / "run_robotwin_online_dual_rft.sh").read_text(
            encoding="utf-8"
        )
        resume_branch = launcher.split("while true; do", 1)[1].split("fi", 1)[0]
        self.assertIn("run_one_shot_training", resume_branch)
        self.assertNotIn("ONE_SHOT_RFT_DONE marker=", resume_branch)

    def test_scheduler_rebuild_is_dataset_and_scheduler_dependent(self) -> None:
        self.assertFalse(should_rebuild_lr_scheduler(
            schedule_mode="base-auxiliary-pseudo", scheduler_type="constant",
            trainer_steps=15000, final_steps=15000,
        ))
        self.assertTrue(should_rebuild_lr_scheduler(
            schedule_mode="epochs", scheduler_type="cosine",
            trainer_steps=1, final_steps=20101,
        ))
        self.assertFalse(should_rebuild_lr_scheduler(
            schedule_mode="steps", scheduler_type="cosine",
            trainer_steps=15000, final_steps=15000,
        ))


if __name__ == "__main__":
    unittest.main()
