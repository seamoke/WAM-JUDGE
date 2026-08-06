from __future__ import annotations

import unittest

from robotwin_critic.two_stage_rft.select_rft_candidates import select_candidates


def candidate(context: str, score: float, action: float, domain: str = "clean"):
    return {
        "context_id": context,
        "task": "task-a",
        "domain": domain,
        "process_score": score,
        "process_critic": {"numeric_parsed": True},
        "action_critic": {"accepted": action > 0.5, "action_score": action},
    }


class CandidateSelectionTest(unittest.TestCase):
    def test_action_is_gate_and_process_is_ranking_reward(self) -> None:
        rows = [
            candidate("c1", 90.0, 0.4),
            candidate("c1", 20.0, 0.9),
            candidate("c2", 50.0, 0.8),
        ]
        selected, summary = select_candidates(
            rows,
            {"task-a/clean": 2},
            min_action_score=0.5,
        )
        self.assertEqual([row["process_score"] for row in selected], [20.0, 50.0])
        self.assertEqual(summary["action_rejected"], 1)
        self.assertEqual(
            {row["rft_selection"]["mode"] for row in selected}, {"dual"}
        )

    def test_exact_budget_shortfall_fails_closed(self) -> None:
        with self.assertRaises(RuntimeError):
            select_candidates(
                [candidate("c1", 20.0, 0.9)],
                {"task-a/clean": 2},
                min_action_score=0.5,
            )

    def test_process_only_does_not_require_action_acceptance(self) -> None:
        selected, summary = select_candidates(
            [candidate("c1", 90.0, 0.1)],
            {"task-a/clean": 1},
            mode="process",
            min_action_score=0.5,
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(summary["mode"], "process")


if __name__ == "__main__":
    unittest.main()
