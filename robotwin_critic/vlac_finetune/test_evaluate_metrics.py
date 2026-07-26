import unittest

from .evaluate_vlac import pair_classification_metrics, summarize_prediction_records


class PairClassificationMetricsTest(unittest.TestCase):
    def test_perfect_three_class_predictions(self):
        metrics = pair_classification_metrics(
            targets=[-30.0, -10.0, 0.0, 0.0, 10.0, 30.0],
            predictions=[-25.0, -8.0, 0.0, 0.0, 9.0, 28.0],
            neutral_threshold=5.0,
        )
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["macro_f1"], 1.0)
        self.assertAlmostEqual(metrics["macro_ovr_auc"], 1.0)
        self.assertAlmostEqual(metrics["target_spearman"], 1.0)

    def test_direction_swap_penalizes_positive_and_negative(self):
        metrics = pair_classification_metrics(
            targets=[-20.0, 0.0, 20.0],
            predictions=[20.0, 0.0, -20.0],
            neutral_threshold=5.0,
        )
        self.assertAlmostEqual(metrics["accuracy"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["macro_f1"], 1.0 / 3.0)
        self.assertLess(metrics["macro_ovr_auc"], 0.5)
        self.assertAlmostEqual(metrics["target_spearman"], -1.0)

    def test_sign_accuracy_and_strict_accuracy_are_distinct(self):
        pairs = [
            {
                "task": "task_a",
                "target": -20.0,
                "prediction": -2.0,
                "parsed": True,
            },
            {
                "task": "task_a",
                "target": 0.0,
                "prediction": 0.0,
                "parsed": True,
            },
            {
                "task": "task_a",
                "target": 20.0,
                "prediction": 2.0,
                "parsed": True,
            },
        ]
        summary, per_task = summarize_prediction_records(
            pairs,
            [
                {
                    "task": "task_a",
                    "voc": 1.0,
                    "vroc": 1.0,
                    "voc_f1": 1.0,
                    "antisymmetry_mae": 0.0,
                    "all_numeric": True,
                }
            ],
            neutral_threshold=5.0,
        )
        self.assertAlmostEqual(summary["pair_sign_accuracy"], 1.0)
        self.assertAlmostEqual(summary["pair_accuracy"], 1.0 / 3.0)
        self.assertEqual(per_task["task_count"], 1)
        self.assertAlmostEqual(
            per_task["tasks"]["task_a"]["pair_accuracy"],
            1.0 / 3.0,
        )


if __name__ == "__main__":
    unittest.main()
