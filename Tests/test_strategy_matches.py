import unittest

from src.Utils.StrategyMatches import (
    STRATEGIES, matched_keys, pick_side, side_line, signals_for_game, _result_for,
)


def _prof(wr_1m=50, ats_3m=50, season_wr=50):
    return {
        "last_1m": {"overall": {"win_rate": wr_1m}},
        "last_3m": {"overall": {"ats_rate": ats_3m}},
        "season": {"overall": {"win_rate": season_wr}},
    }


class StrategyMatchesTest(unittest.TestCase):
    def test_keys_are_unique_and_named(self):
        keys = [s["key"] for s in STRATEGIES]
        self.assertEqual(len(keys), len(set(keys)))
        for s in STRATEGIES:
            self.assertIn(s["pick_rule"], ("home", "away", "dog"))

    def test_home_weak_big_spread_fires_and_picks_away(self):
        keys = matched_keys(7.5, _prof(wr_1m=30), _prof())
        self.assertIn("home_weak_big_spread", keys)
        self.assertIn("mid_spread_home_weak", keys)
        self.assertIn("mid_spread_dog", keys)
        self.assertEqual(pick_side("home_weak_big_spread", 7.5), "away")
        # dog = away when home is favoured, home when away is favoured
        self.assertEqual(pick_side("mid_spread_dog", 7.5), "away")
        self.assertEqual(pick_side("mid_spread_dog", -7.5), "home")

    def test_home_dog_strong_needs_home_underdog(self):
        self.assertIn("home_dog_strong", matched_keys(-2.5, _prof(season_wr=60), _prof()))
        self.assertNotIn("home_dog_strong", matched_keys(2.5, _prof(season_wr=60), _prof()))

    def test_zero_win_rate_is_not_treated_as_fifty(self):
        # 0-5 in the last month is "weak"; None (no games) is neutral.
        self.assertIn("home_weak_big_spread", matched_keys(6, _prof(wr_1m=0.0), _prof()))
        self.assertNotIn("home_weak_big_spread", matched_keys(6, _prof(wr_1m=None), _prof()))

    def test_side_line_is_from_the_picked_side(self):
        self.assertEqual(side_line("home", 7.5), "−7.5")
        self.assertEqual(side_line("away", 7.5), "+7.5")
        self.assertEqual(side_line("home", -3.5), "+3.5")
        self.assertEqual(side_line("away", 0), "PK")

    def test_signal_text_names_the_team_and_line(self):
        game = {"spread": 7.5, "home_team": "Washington Wizards", "away_team": "Memphis Grizzlies",
                "home_team_zh": "巫師", "away_team_zh": "灰熊",
                "home_profile": _prof(wr_1m=20), "away_profile": _prof()}
        sigs = signals_for_game(game)
        first = sigs[0]
        self.assertEqual(first["pick_team_zh"], "灰熊")
        self.assertEqual(first["pick_line"], "+7.5")
        self.assertIn("押 灰熊 +7.5", first["text"])
        self.assertEqual(signals_for_game({**game, "is_playoff": True}), [])

    def test_result_grading(self):
        # home −7.5, home wins by 10 → home covers; away pick loses
        self.assertEqual(_result_for("away", 7.5, 10)[0], "loss")
        self.assertEqual(_result_for("home", 7.5, 10)[0], "win")
        self.assertEqual(_result_for("away", 7.5, 7)[0], "win")
        self.assertEqual(_result_for("away", 7, 7)[0], "push")
        self.assertEqual(_result_for("away", 7.5, None)[0], "pending")


if __name__ == "__main__":
    unittest.main()
