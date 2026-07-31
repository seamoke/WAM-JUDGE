from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from robotwin_critic.vlac_finetune.two_stage_protocol import audit_protocol


class ProtocolTest(unittest.TestCase):
    def test_exact_30_20_partition_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = {"task": "task-a", "domains": {}}
            for domain in ("clean", "randomized"):
                domain_row = {
                    "selected_source_episode_indices_ranked": list(range(50)),
                    "stage1_source_episode_indices": list(range(30)),
                    "stage2_source_episode_indices": list(range(30, 50)),
                }
                for stage, indices in (
                    ("stage1", range(30)),
                    ("stage2", range(30, 50)),
                ):
                    repo = Path(stage) / domain / "task-a"
                    (root / repo / "meta").mkdir(parents=True)
                    (root / repo / "meta" / "episodes.jsonl").write_text("")
                    domain_row[f"{stage}_output_repo"] = str(repo)
                    domain_row[f"{stage}_output"] = {
                        "source_to_destination_index": {
                            str(source): destination
                            for destination, source in enumerate(indices)
                        }
                    }
                task["domains"][domain] = domain_row
            manifest = {
                "schema_version": 1,
                "split": {
                    "per_domain_total": 50,
                    "stage1_per_domain": 30,
                    "stage2_per_domain": 20,
                },
                "tasks": [task],
            }
            manifest_path = root / "split_manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            (root / "PREPARATION_COMPLETE.json").write_text(
                json.dumps({"manifest_sha256": digest})
            )
            summary = audit_protocol(root, expected_tasks=1)
            self.assertEqual(summary["episodes_total"], 100)
            self.assertEqual(summary["counts"]["stage1"]["clean"], 30)
            self.assertEqual(summary["counts"]["stage2"]["randomized"], 20)


if __name__ == "__main__":
    unittest.main()
