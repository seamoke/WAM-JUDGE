from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from robotwin_critic.two_stage_rft.summarize_online_task_retention import (
    summarize_collect_root,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class TaskRetentionSummaryTest(unittest.TestCase):
    def test_counts_q_and_qa_pairs_by_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collect = root / "collect_000000"
            generated = [
                {"task": "pick", "context_id": "q0", "candidate_id": "q0/0"},
                {"task": "pick", "context_id": "q0", "candidate_id": "q0/1"},
                {"task": "place", "context_id": "q1", "candidate_id": "q1/0"},
            ]
            retained = [generated[1]]
            write_jsonl(collect / "dual_scored.jsonl", generated)
            write_jsonl(collect / "selected_winners.jsonl", retained)

            summary = summarize_collect_root(root)

            self.assertEqual(summary["generated_q"], 2)
            self.assertEqual(summary["generated_qa_pairs"], 3)
            self.assertEqual(summary["retained_q"], 1)
            self.assertEqual(summary["retained_qa_pairs"], 1)
            self.assertEqual(summary["tasks"]["pick"]["generated_q"], 1)
            self.assertEqual(summary["tasks"]["pick"]["retained_qa_pairs"], 1)
            self.assertEqual(summary["tasks"]["place"]["retained_q"], 0)


if __name__ == "__main__":
    unittest.main()
