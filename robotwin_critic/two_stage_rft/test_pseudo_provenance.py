from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from robotwin_critic.two_stage_rft.pseudo_provenance import (
    validate_pseudo_split_provenance,
)


class PseudoProvenanceTest(unittest.TestCase):
    def _manifest(self, root: Path) -> tuple[Path, str]:
        path = root / "split_manifest.json"
        path.write_text(
            json.dumps(
                {
                    "tasks": [
                        {
                            "task": "pick",
                            "domains": {
                                "clean": {
                                    "source_repo": "/current/source/pick-demo-clean-50",
                                    "stage2_output_repo": "stage2/clean/pick",
                                    "stage2_source_episode_indices": [4, 7],
                                    "stage2_output": {
                                        "source_to_destination_index": {
                                            "4": 0,
                                            "7": 1,
                                        }
                                    },
                                }
                            },
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return path, digest

    def test_raw_hash_match_is_fast_path(self) -> None:
        report = validate_pseudo_split_provenance(
            [{"split_manifest_sha256": "same"}],
            expected_split_sha256="same",
            split_manifest_path=None,
        )
        self.assertEqual(report["validation_mode"], "raw_hash")

    def test_hash_mismatch_accepts_complete_stage2_membership(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, current_hash = self._manifest(Path(directory))
            row = {
                "split_manifest_sha256": "f95-old",
                "source_stage": "stage2",
                "task": "pick",
                "source_task": "pick",
                "domain": "clean",
                "source_episode_index": 7,
                "source_repo": "/published/redacted/pick",
                "output_episode_index": 1,
                "frame_index": 12,
                "source_context_id": "pick/clean/7/12",
                "source_parquet": "/published/redacted/pick/data/episode_000001.parquet",
            }
            report = validate_pseudo_split_provenance(
                [row], expected_split_sha256=current_hash,
                split_manifest_path=manifest,
            )
            self.assertEqual(report["validation_mode"], "semantic_membership")
            self.assertEqual(report["package_split_sha256"], "f95-old")
            self.assertEqual(report["current_split_sha256"], current_hash)

    def test_hash_mismatch_rejects_missing_or_invalid_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, current_hash = self._manifest(Path(directory))
            valid = {
                "split_manifest_sha256": "old",
                "source_stage": "stage2",
                "task": "pick",
                "domain": "clean",
                "source_episode_index": 7,
                "source_repo": "/published/pick",
                "output_episode_index": 1,
                "frame_index": 12,
                "source_context_id": "pick/clean/7/12",
                "source_parquet": "/published/pick/data/episode_000001.parquet",
            }
            for mutation, message in (
                ({"source_stage": "stage1"}, "source_stage"),
                ({"source_episode_index": 9}, "not in current stage2"),
                ({"source_repo": "/published/wrong"}, "basename"),
                ({"output_episode_index": 0}, "source/output episode mapping"),
                ({"source_context_id": "pick/clean/7/13"}, "source_context_id"),
                ({"source_parquet": "/published/episode_000000.parquet"}, "source_parquet"),
                ({"domain": None}, "missing provenance"),
            ):
                row = {**valid, **mutation}
                with self.assertRaisesRegex(ValueError, message):
                    validate_pseudo_split_provenance(
                        [row], expected_split_sha256=current_hash,
                        split_manifest_path=manifest,
                    )


if __name__ == "__main__":
    unittest.main()
