"""Standard-library tests for pseudo artifact byte/path binding."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from robotwin_critic.two_stage_rft.pseudo_artifact_contract import (
    ARTIFACT_PATH_FIELDS,
    PseudoArtifactContractError,
    bind_artifact_hashes,
    validate_pseudo_artifact_jsonl,
)


class PseudoArtifactContractTest(unittest.TestCase):
    def make_artifacts(self, parent: Path) -> dict:
        parent.mkdir(parents=True, exist_ok=True)
        row = {}
        for index, field in enumerate(ARTIFACT_PATH_FIELDS):
            path = parent / f"{field}.bin"
            path.write_bytes(f"artifact-{index}".encode())
            row[field] = path.name
        return row

    def test_binding_emits_absolute_paths_and_validates_after_relocation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "different" / "selected.jsonl"
            output.parent.mkdir()
            bound = bind_artifact_hashes(self.make_artifacts(source), jsonl_parent=source)
            output.write_text(json.dumps(bound) + "\n", encoding="utf-8")
            self.assertTrue(all(Path(bound[field]).is_absolute() for field in ARTIFACT_PATH_FIELDS))
            self.assertEqual(validate_pseudo_artifact_jsonl(output), 1)

    def test_tampered_bytes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bound = bind_artifact_hashes(self.make_artifacts(root), jsonl_parent=root)
            Path(bound["action_path"]).write_bytes(b"tampered")
            path = root / "rows.jsonl"
            path.write_text(json.dumps(bound) + "\n", encoding="utf-8")
            with self.assertRaises(PseudoArtifactContractError) as caught:
                validate_pseudo_artifact_jsonl(path)
            self.assertEqual(caught.exception.counts["mismatched_action_path_sha256"], 1)

    def test_missing_and_invalid_hashes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bound = bind_artifact_hashes(self.make_artifacts(root), jsonl_parent=root)
            del bound["latent_path_sha256"]
            bound["text_emb_path_sha256"] = "not-a-sha256"
            path = root / "rows.jsonl"
            path.write_text(json.dumps(bound) + "\n", encoding="utf-8")
            with self.assertRaises(PseudoArtifactContractError) as caught:
                validate_pseudo_artifact_jsonl(path)
            self.assertEqual(caught.exception.counts["missing_or_invalid_latent_path_sha256"], 1)
            self.assertEqual(caught.exception.counts["missing_or_invalid_text_emb_path_sha256"], 1)

    def test_malformed_json_and_artifact_error_use_physical_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bound = bind_artifact_hashes(self.make_artifacts(root), jsonl_parent=root)
            bound["action_path_sha256"] = "0" * 64
            path = root / "rows.jsonl"
            path.write_text("\n{bad json}\n\n" + json.dumps(bound) + "\n", encoding="utf-8")
            with self.assertRaises(PseudoArtifactContractError) as caught:
                validate_pseudo_artifact_jsonl(path)
            self.assertEqual(caught.exception.examples["malformed_json"], [2])
            self.assertEqual(caught.exception.examples["mismatched_action_path_sha256"], [4])


if __name__ == "__main__":
    unittest.main()
