import unittest

from robotwin_critic.two_stage_rft.rollout_provenance import (
    canonical_source_stage,
    with_rollout_provenance,
)


class RolloutProvenanceTest(unittest.TestCase):
    def test_normalizes_stage2_context(self):
        row = with_rollout_provenance(
            {"task": "pick", "stage": "stage2_video_only", "candidate_id": "c0"}
        )
        self.assertEqual(row["source_task"], "pick")
        self.assertEqual(row["source_stage"], "stage2")
        self.assertEqual(row["stage"], "stage2_video_only")

    def test_preserves_explicit_provenance(self):
        row = with_rollout_provenance(
            {"task": "legacy", "source_task": "place", "source_stage": "stage1"}
        )
        self.assertEqual(row["source_task"], "place")
        self.assertEqual(row["source_stage"], "stage1")

    def test_unknown_source_is_explicit(self):
        self.assertEqual(canonical_source_stage({}), "unknown")


if __name__ == "__main__":
    unittest.main()
