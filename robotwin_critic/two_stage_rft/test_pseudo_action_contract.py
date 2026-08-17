"""Standard-library tests for torch-free pseudo action contract validation."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from robotwin_critic.two_stage_rft.pseudo_action_contract import (
    EXECUTABLE_ACTION_SEMANTICS,
    PseudoActionContractError,
    main,
    validate_pseudo_action_contract,
    validate_pseudo_action_jsonl,
)
from robotwin_critic.two_stage_rft.pseudo_artifact_contract import (
    ARTIFACT_PATH_FIELDS,
    bind_artifact_hashes,
)


def strict_row(candidate_id: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "action_semantics": EXECUTABLE_ACTION_SEMANTICS,
        "latent_frames": 2,
        "executable_action_steps": 16,
        "action_critic": {
            "accepted": True,
            "hard_violations": [],
            "gate_violations": [],
            "gate_policy": "strict",
        },
    }


class PseudoActionContractTest(unittest.TestCase):
    def bound_strict_row(self, candidate_id: str) -> dict:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        parent = Path(directory.name)
        row = strict_row(candidate_id)
        for field in ARTIFACT_PATH_FIELDS:
            artifact = parent / field
            artifact.write_bytes((candidate_id + field).encode())
            row[field] = artifact.name
        return bind_artifact_hashes(row, jsonl_parent=parent)

    def write_jsonl(self, lines: list[str]) -> Path:
        temporary = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False)
        with temporary:
            temporary.write("".join(lines))
        self.addCleanup(Path(temporary.name).unlink)
        return Path(temporary.name)

    def validate(self, path: Path) -> int:
        return validate_pseudo_action_jsonl(
            path, expected_latent_frames=2, action_per_frame=16
        )

    def waiver_values(self, path: Path) -> dict:
        return {
            "legacy_pseudo_action_waiver_sha256": hashlib.sha256(
                path.read_bytes()
            ).hexdigest(),
            "legacy_pseudo_action_waiver_rows": sum(
                1 for line in path.read_bytes().splitlines() if line.strip()
            ),
        }

    def test_passing_strict_rows_and_blank_lines(self) -> None:
        path = self.write_jsonl([
            "\n", json.dumps(self.bound_strict_row("a")) + "\n", "   \n",
            json.dumps(self.bound_strict_row("b")) + "\n",
        ])
        self.assertEqual(self.validate(path), 2)

    def test_rejects_soft_gate(self) -> None:
        soft = strict_row("soft")
        soft["action_critic"].update({
            "accepted": False,
            "gate_policy": "score_with_safety_gates",
            "gate_violations": ["workspace"],
        })
        path = self.write_jsonl([json.dumps(soft) + "\n"])
        with self.assertRaises(PseudoActionContractError) as caught:
            self.validate(path)
        self.assertEqual(caught.exception.counts["action_critic.accepted"], 1)
        self.assertEqual(caught.exception.counts["action_critic.gate_policy"], 1)

    def test_rejects_hard_violations(self) -> None:
        hard = strict_row("hard")
        hard["action_critic"]["hard_violations"] = ["left.jerk"]
        path = self.write_jsonl([json.dumps(hard) + "\n"])
        with self.assertRaises(PseudoActionContractError) as caught:
            self.validate(path)
        self.assertEqual(caught.exception.counts["action_critic.hard_violations"], 1)

    def test_legacy_metadata_is_rejected_without_explicit_waiver(self) -> None:
        legacy = self.bound_strict_row("legacy-default-reject")
        legacy["action_critic"].update({
            "hard_violations": ["left.jerk"],
            "gate_policy": "score_with_safety_gates",
        })
        path = self.write_jsonl([json.dumps(legacy) + "\n"])
        with self.assertRaises(PseudoActionContractError):
            self.validate(path)

    def test_legacy_waiver_rejects_wrong_hash(self) -> None:
        path = self.write_jsonl([json.dumps(self.bound_strict_row("hash")) + "\n"])
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            validate_pseudo_action_jsonl(
                path, expected_latent_frames=2, action_per_frame=16,
                legacy_pseudo_action_waiver_sha256="0" * 64,
                legacy_pseudo_action_waiver_rows=1,
            )

    def test_legacy_waiver_rejects_wrong_row_count(self) -> None:
        path = self.write_jsonl([json.dumps(self.bound_strict_row("rows")) + "\n"])
        values = self.waiver_values(path)
        values["legacy_pseudo_action_waiver_rows"] = 2
        with self.assertRaisesRegex(ValueError, "row-count mismatch"):
            validate_pseudo_action_jsonl(
                path, expected_latent_frames=2, action_per_frame=16, **values
            )

    def test_legacy_waiver_rejects_forbidden_metadata(self) -> None:
        cases = (
            ({"accepted": False}, "action_critic.accepted"),
            ({"gate_violations": ["workspace"]}, "action_critic.gate_violations"),
            ({"gate_policy": "strict"}, "action_critic.gate_policy"),
        )
        for update, issue in cases:
            with self.subTest(issue=issue):
                legacy = self.bound_strict_row("forbidden-" + issue)
                legacy["action_critic"].update({
                    "hard_violations": ["left.jerk"],
                    "gate_policy": "score_with_safety_gates",
                })
                legacy["action_critic"].update(update)
                path = self.write_jsonl([json.dumps(legacy) + "\n"])
                with self.assertRaises(PseudoActionContractError) as caught:
                    validate_pseudo_action_jsonl(
                        path, expected_latent_frames=2, action_per_frame=16,
                        **self.waiver_values(path),
                    )
                self.assertEqual(caught.exception.counts[issue], 1)

    def test_exact_legacy_waiver_accepts_only_relaxed_metadata(self) -> None:
        legacy = self.bound_strict_row("exact")
        legacy["action_critic"].update({
            "hard_violations": ["left.jerk"],
            "gate_policy": "score_with_safety_gates",
        })
        path = self.write_jsonl([json.dumps(legacy) + "\n"])
        self.assertEqual(
            validate_pseudo_action_jsonl(
                path, expected_latent_frames=2, action_per_frame=16,
                **self.waiver_values(path),
            ),
            1,
        )

    def test_validates_all_rows_and_reports_physical_line_numbers(self) -> None:
        first = strict_row("first")
        first["latent_frames"] = 3
        last = strict_row("last")
        last["executable_action_steps"] = 15
        path = self.write_jsonl([
            json.dumps(first) + "\n", "\n", json.dumps(strict_row("middle")) + "\n",
            json.dumps(last) + "\n",
        ])
        with self.assertRaises(PseudoActionContractError) as caught:
            self.validate(path)
        message = str(caught.exception)
        self.assertIn("latent_frames=1 [line 1 (first)]", message)
        self.assertIn("executable_action_steps=1 [line 4 (last)]", message)

    def test_blank_only_file_is_rejected(self) -> None:
        path = self.write_jsonl(["\n", "  \t\n"])
        with self.assertRaises(PseudoActionContractError) as caught:
            self.validate(path)
        self.assertEqual(caught.exception.counts["no_nonblank_rows"], 1)

    def test_non_object_row_is_rejected_with_physical_line(self) -> None:
        path = self.write_jsonl(["\n", "[]\n"])
        with self.assertRaises(PseudoActionContractError) as caught:
            self.validate(path)
        self.assertIn("json_object=1 [line 2 (unknown)]", str(caught.exception))

    def test_malformed_json_is_aggregated_with_other_row_failures(self) -> None:
        invalid = strict_row("invalid")
        invalid["action_semantics"] = "absolute actions"
        path = self.write_jsonl(["{bad json}\n", json.dumps(invalid) + "\n"])
        with self.assertRaises(PseudoActionContractError) as caught:
            self.validate(path)
        self.assertEqual(caught.exception.row_count, 2)
        self.assertEqual(caught.exception.counts["malformed_json"], 1)
        self.assertEqual(caught.exception.counts["action_semantics"], 1)
        self.assertIn("malformed_json=1 [line 1 (unknown)]", str(caught.exception))

    def test_boolean_integer_fields_are_rejected(self) -> None:
        invalid = strict_row("bools")
        invalid["latent_frames"] = True
        invalid["executable_action_steps"] = True
        path = self.write_jsonl([json.dumps(invalid) + "\n"])
        with self.assertRaises(PseudoActionContractError) as caught:
            self.validate(path)
        self.assertEqual(caught.exception.counts["latent_frames"], 1)
        self.assertEqual(caught.exception.counts["executable_action_steps"], 1)

    def test_float_integer_fields_are_rejected(self) -> None:
        invalid = strict_row("floats")
        invalid["latent_frames"] = 2.0
        invalid["executable_action_steps"] = 16.0
        path = self.write_jsonl([json.dumps(invalid) + "\n"])
        with self.assertRaises(PseudoActionContractError) as caught:
            self.validate(path)
        self.assertEqual(caught.exception.counts["latent_frames"], 1)
        self.assertEqual(caught.exception.counts["executable_action_steps"], 1)

    def test_dimension_float_and_bool_are_rejected(self) -> None:
        for expected_latent_frames, action_per_frame in (
            (2.0, 16),
            (True, 16),
            (2, 16.0),
            (2, False),
        ):
            with self.subTest(
                expected_latent_frames=expected_latent_frames,
                action_per_frame=action_per_frame,
            ), self.assertRaisesRegex(
                ValueError, "Expected dimensions must be positive integers"
            ):
                validate_pseudo_action_contract(
                    [strict_row("dimensions")],
                    expected_latent_frames=expected_latent_frames,
                    action_per_frame=action_per_frame,
                )

    def test_dataset_passes_raw_config_dimensions_to_contract(self) -> None:
        dataset_path = Path(__file__).with_name("action_only_dataset.py")
        tree = ast.parse(dataset_path.read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "validate_pseudo_action_contract"
        ]
        self.assertEqual(len(calls), 1)
        keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
        for keyword, attribute in (
            ("expected_latent_frames", "frame_chunk_size"),
            ("action_per_frame", "action_per_frame"),
        ):
            with self.subTest(keyword=keyword):
                value = keywords[keyword]
                self.assertIsInstance(value, ast.Attribute)
                self.assertIsInstance(value.value, ast.Name)
                self.assertEqual(value.value.id, "config")
                self.assertEqual(value.attr, attribute)

    def test_cli_prints_concise_success_summary(self) -> None:
        path = self.write_jsonl([json.dumps(self.bound_strict_row("ok")) + "\n"])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main([
                str(path), "--expected-latent-frames", "2", "--action-per-frame", "16"
            ]), 0)
        self.assertEqual(
            output.getvalue().strip(),
            "Pseudo action contract valid: rows=1 latent_frames=2 executable_action_steps=16",
        )

    def test_cli_failure_exits_nonzero(self) -> None:
        path = self.write_jsonl(["null\n"])
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
            main([
                str(path), "--expected-latent-frames", "2", "--action-per-frame", "16"
            ])
        self.assertEqual(caught.exception.code, 1)
        self.assertIn("line 1", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
