import unittest

from robotwin_critic.two_stage_rft.one_shot_plateau import update_plateau


class OneShotPlateauTest(unittest.TestCase):
    def test_first_observation_is_only_a_baseline(self):
        state = update_plateau(
            None, collect_index=155, selected=19800, min_delta=50, patience=10
        )
        self.assertEqual(state["consecutive_low_growth"], 0)
        self.assertIsNone(state["selected_delta"])
        self.assertFalse(state["stopped"])

    def test_stops_after_ten_new_low_growth_rounds(self):
        state = update_plateau(
            None, collect_index=100, selected=19000, min_delta=50, patience=10
        )
        for offset in range(1, 11):
            state = update_plateau(
                state,
                collect_index=100 + offset,
                selected=19000 + offset * 49,
                min_delta=50,
                patience=10,
            )
        self.assertEqual(state["consecutive_low_growth"], 10)
        self.assertTrue(state["stopped"])

    def test_large_growth_resets_patience_and_duplicate_round_is_ignored(self):
        state = update_plateau(
            None, collect_index=4, selected=1000, min_delta=50, patience=10
        )
        state = update_plateau(
            state, collect_index=5, selected=1020, min_delta=50, patience=10
        )
        duplicate = update_plateau(
            state, collect_index=5, selected=1020, min_delta=50, patience=10
        )
        self.assertEqual(duplicate, state)
        state = update_plateau(
            state, collect_index=6, selected=1100, min_delta=50, patience=10
        )
        self.assertEqual(state["consecutive_low_growth"], 0)


if __name__ == "__main__":
    unittest.main()
