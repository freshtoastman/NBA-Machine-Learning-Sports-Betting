import unittest

from src.Utils.StrategyMatches import (
    STRATEGIES, matched_keys, pick_side, side_line, signals_for_game, _result_for,
    rule_weight, verdict_for, payout_profile, verdict_tier,
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

    def _sigs(self, spread=-5.5, **kw):
        game = {"spread": spread, "home_team": "Brooklyn Nets", "away_team": "Philadelphia 76ers",
                "home_team_zh": "籃網", "away_team_zh": "七六人",
                "home_profile": _prof(**kw.get("home", {})), "away_profile": _prof(**kw.get("away", {}))}
        return signals_for_game(game)

    def test_weight_pools_history_with_season_and_floors_at_zero(self):
        self.assertEqual(rule_weight("mid_spread_dog", 0, 0), 8.9)       # history only
        self.assertEqual(rule_weight("home_dog_strong", 3, 10), 0.0)     # (3+12.3)/33 < 50%
        self.assertGreater(rule_weight("away_hot_1m", 41, 30), 0)

    def test_conflict_gives_one_side_and_nested_dog_rules_vote_once(self):
        sigs = self._sigs(home={"wr_1m": 25}, away={"wr_1m": 80})
        self.assertEqual({s["pick_side"] for s in sigs}, {"home", "away"})
        v = verdict_for(sigs, {})
        self.assertTrue(v["conflict"])
        nested = [x for x in v["votes"] if x.get("nested")]
        self.assertEqual(len(nested), 2)                    # 3 dog rules, 1 votes
        self.assertEqual(v["score_home"], 20.1)             # 70.1 − 50
        self.assertEqual(v["score_away"], 10.1)             # 60.1 − 50
        self.assertEqual(v["side"], "home")
        self.assertIn("買 籃網 +5.5", v["text"])

    def test_season_record_can_flip_the_verdict(self):
        sigs = self._sigs(home={"wr_1m": 25}, away={"wr_1m": 80})
        bad_dog = {k: (0, 40) for k in ("home_weak_big_spread", "mid_spread_home_weak", "mid_spread_dog")}
        v = verdict_for(sigs, bad_dog)
        self.assertEqual(v["side"], "away")
        self.assertEqual(v["team_zh"], "七六人")

    def test_all_zero_weights_mean_no_bet(self):
        sigs = self._sigs(spread=-2.5, home={"season_wr": 60})
        v = verdict_for(sigs, {"home_dog_strong": (3, 10)})
        self.assertIsNone(v["side"])
        self.assertIn("不下注", v["text"])

    def test_payout_profile_flags_losers_and_near_breakeven(self):
        self.assertEqual(payout_profile(36, 34)["risk"], "lose")      # 51.4% < 53.5%
        self.assertLess(payout_profile(36, 34)["roi_high"], 0)
        self.assertEqual(payout_profile(32, 28)["risk"], "lose")      # 53.3%: wins at 0.90, loses at 0.87
        self.assertGreater(payout_profile(32, 28)["roi_high"], 0)
        self.assertLess(payout_profile(32, 28)["roi_low"], 0)
        self.assertEqual(payout_profile(41, 30)["risk"], "edge")      # 57.7%
        self.assertEqual(payout_profile(26, 16)["risk"], "ok")        # 61.9%
        self.assertEqual(payout_profile(0, 0)["risk"], "na")
        self.assertEqual(payout_profile(10, 10)["roi_low"], -6.5)     # (10*0.87-10)/20

    def test_verdict_tiers(self):
        self.assertEqual([verdict_tier(m) for m in (3.1, 6.0, 14.9, 22.8)], [0, 1, 1, 2])


if __name__ == "__main__":
    unittest.main()
