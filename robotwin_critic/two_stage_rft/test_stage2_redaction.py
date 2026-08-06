from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class Stage2RedactionTest(unittest.TestCase):
    def test_stage2_stats_drop_action_distribution(self) -> None:
        from script.prepare_robotwin_two_stage_dataset import (
            redact_episode_action_stats,
        )

        rows = [
            {
                "episode_index": 0,
                "stats": {
                    "action": {"mean": [1.0]},
                    "observation.state": {"mean": [2.0]},
                },
            }
        ]
        redacted = redact_episode_action_stats(rows)
        self.assertNotIn("action", redacted[0]["stats"])
        self.assertIn("observation.state", redacted[0]["stats"])
        self.assertIn("action", rows[0]["stats"])

    def test_stage2_parquet_physically_drops_action(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        from script.prepare_robotwin_two_stage_dataset import materialize_parquet

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.parquet"
            destination = root / "stage2.parquet"
            pq.write_table(
                pa.table(
                    {
                        "action": [[0.0] * 16, [1.0] * 16],
                        "observation.state": [[0.0] * 16, [1.0] * 16],
                    }
                ),
                source,
            )
            materialize_parquet(source, destination, redact_action=True)
            self.assertEqual(
                pq.read_schema(destination).names,
                ["observation.state"],
            )
            self.assertIn("action", pq.read_schema(source).names)


if __name__ == "__main__":
    unittest.main()
