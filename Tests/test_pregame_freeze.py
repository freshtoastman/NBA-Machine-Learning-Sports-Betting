import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.Utils import FrozenRecord, Policy, PregameFreeze


def _pregame_game(**over):
    g = {
        "home_team": "Boston Celtics", "away_team": "New York Knicks",
        "winner": "home", "home_confidence": 61.0, "away_confidence": 39.0,
        "home_team_odds": -150, "away_team_odds": 130,
        "ou_pick": "OVER", "ou_value": 220.5, "ou_confidence": 55.0,
        "spread": 3.5, "ats_model_pick": "home", "ats_model_home_prob": 54.0,
        "ats_model_confidence": 54.0, "ats_value_edge": 4.0, "ats_is_value": False,
        "ats_shadow_value": False, "ats_reference_only": True,
        "value_side": "home", "is_value": False, "ml_shadow_value": True, "ml_shadow_golden": False,
        "home_score": None, "away_score": None, "actual_winner": None,
        "ml_correct": None, "ats_winner": None, "ats_cover_margin": None, "ou_correct": None,
    }
    g.update(over)
    return g


class TestApplyFreeze(unittest.TestCase):
    def test_first_pregame_export_writes_unlocked_block(self):
        games = {"K": _pregame_game()}
        PregameFreeze.apply_freeze(None, games, "2026-10-20T06:00:00")
        block = games["K"]["pregame"]
        self.assertFalse(block["locked"])
        self.assertEqual(block["revisions"], 1)
        self.assertEqual(block["frozen_at"], "2026-10-20T06:00:00")
        self.assertEqual(block["spread"], 3.5)
        self.assertFalse(games["K"]["pregame_missing"])

    def test_later_pregame_export_refreshes_block(self):
        prev = {"K": _pregame_game()}
        PregameFreeze.apply_freeze(None, prev, "2026-10-20T06:00:00")
        fresh = {"K": _pregame_game(spread=4.5, ats_model_pick="away")}
        PregameFreeze.apply_freeze(prev, fresh, "2026-10-21T06:00:00")
        block = fresh["K"]["pregame"]
        self.assertFalse(block["locked"])
        self.assertEqual(block["revisions"], 2)
        self.assertEqual(block["spread"], 4.5)
        self.assertEqual(block["ats_model_pick"], "away")

    def test_result_locks_block_and_grades_against_frozen_lines(self):
        prev = {"K": _pregame_game()}
        PregameFreeze.apply_freeze(None, prev, "2026-10-21T06:00:00")
        # Post-game re-run: closing spread moved to 6.5, model now says away.
        fresh = {"K": _pregame_game(spread=6.5, winner="away", ats_model_pick="away",
                                    ou_value=225.5, home_score=110, away_score=105,
                                    actual_winner="home", ml_correct=False,
                                    ats_winner="away", ats_cover_margin=-1.5)}
        PregameFreeze.apply_freeze(prev, fresh, "2026-10-22T06:00:00")
        g = fresh["K"]
        self.assertTrue(g["pregame"]["locked"])
        self.assertEqual(g["pregame"]["locked_at"], "2026-10-22T06:00:00")
        # Frozen prediction is what is shown…
        self.assertEqual(g["winner"], "home")
        self.assertEqual(g["ats_model_pick"], "home")
        self.assertEqual(g["spread"], 3.5)
        self.assertEqual(g["ou_value"], 220.5)
        # …and graded against the frozen spread (home by 5 covers 3.5, not 6.5).
        self.assertTrue(g["ml_correct"])
        self.assertEqual(g["ats_winner"], "home")
        self.assertEqual(g["ats_cover_margin"], 1.5)
        self.assertEqual(g["actual_ou_result"], "UNDER")
        self.assertFalse(g["ou_correct"])
        # The closing numbers stay visible for comparison.
        self.assertEqual(g["closing"]["spread"], 6.5)
        self.assertEqual(g["closing"]["ats_winner"], "away")

    def test_locked_block_survives_further_exports(self):
        prev = {"K": _pregame_game()}
        PregameFreeze.apply_freeze(None, prev, "t1")
        done = {"K": _pregame_game(home_score=100, away_score=90, actual_winner="home")}
        PregameFreeze.apply_freeze(prev, done, "t2")
        again = {"K": _pregame_game(spread=9.5, winner="away", home_score=100, away_score=90,
                                    actual_winner="home")}
        PregameFreeze.apply_freeze(done, again, "t3")
        self.assertEqual(again["K"]["pregame"]["locked_at"], "t2")
        self.assertEqual(again["K"]["spread"], 3.5)
        self.assertEqual(again["K"]["winner"], "home")

    def test_live_game_locks_block(self):
        prev = {"K": _pregame_game()}
        PregameFreeze.apply_freeze(None, prev, "t1")
        live = {"K": _pregame_game(spread=5.5, live_status="Q2 3:12", live_home_score=40, live_away_score=38)}
        PregameFreeze.apply_freeze(prev, live, "t2")
        self.assertTrue(live["K"]["pregame"]["locked"])
        self.assertEqual(live["K"]["spread"], 3.5)
        self.assertIsNone(live["K"]["ml_correct"])  # not graded until a final score exists

    def test_game_first_seen_with_result_has_no_snapshot(self):
        fresh = {"K": _pregame_game(home_score=100, away_score=90, actual_winner="home")}
        PregameFreeze.apply_freeze(None, fresh, "t1")
        self.assertIsNone(fresh["K"]["pregame"])
        self.assertTrue(fresh["K"]["pregame_missing"])
        self.assertEqual(PregameFreeze.frozen_games(fresh.items()), [])

    def test_push_on_frozen_spread(self):
        prev = {"K": _pregame_game(spread=5.0)}
        PregameFreeze.apply_freeze(None, prev, "t1")
        done = {"K": _pregame_game(spread=6.0, home_score=105, away_score=100, actual_winner="home")}
        PregameFreeze.apply_freeze(prev, done, "t2")
        self.assertEqual(done["K"]["ats_winner"], "push")


class TestFrozenRecord(unittest.TestCase):
    def setUp(self):
        Policy.reload()
        self.addCleanup(Policy.reload)
        patcher = mock.patch.object(Policy, "_config", return_value={
            "ats-policy": {"value_flag_enabled": False, "effective_from": "2026-10-02",
                           "reopen_min_frozen_picks": 2, "reopen_ci_lower_pct": 52.4},
            "ml-value-policy": {"value_flag_enabled": False, "effective_from": "2026-10-04",
                                "reopen_min_frozen_picks": 2, "reopen_roi_ci_lower_pct": 0.0},
        })
        patcher.start()
        self.addCleanup(patcher.stop)

    def _write_day(self, d: Path, iso: str, games: dict):
        (d / f"{iso}.json").write_text(json.dumps({"date": iso, "games": games}), encoding="utf-8")

    def test_only_locked_games_are_graded(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            # Day 1: two frozen + graded games, one legacy game without a block.
            prev = {"A": _pregame_game(ats_shadow_value=True), "B": _pregame_game(winner="away", value_side="away")}
            PregameFreeze.apply_freeze(None, prev, "t1")
            done = {"A": _pregame_game(home_score=110, away_score=100, actual_winner="home"),
                    "B": _pregame_game(home_score=110, away_score=100, actual_winner="home"),
                    "C": _pregame_game(home_score=90, away_score=100, actual_winner="away")}
            PregameFreeze.apply_freeze(prev, done, "t2")
            self._write_day(d, "2026-10-21", done)
            # Day 2: a pre-game (unlocked) game.
            pend = {"D": _pregame_game()}
            PregameFreeze.apply_freeze(None, pend, "t3")
            self._write_day(d, "2026-10-22", pend)
            # A file before `since` is ignored.
            self._write_day(d, "2026-04-10", {"Z": done["A"]})

            out = FrozenRecord.write_frozen_record(d, since="2026-10-02")
            self.assertEqual(out["frozen_games"], 2)
            self.assertEqual(out["graded_games"], 2)
            self.assertEqual(out["pending_games"], 1)
            self.assertEqual(out["legacy_graded_games_without_block"], 0)
            self.assertEqual([m["game"] for m in out["games_without_pregame_snapshot"]], ["C"])
            self.assertEqual(out["ml"]["wins"], 1)
            self.assertEqual(out["ml"]["losses"], 1)
            self.assertEqual(out["ats_model"]["n"], 2)           # home covers 3.5 twice
            self.assertEqual(out["ats_shadow_value"]["n"], 1)
            self.assertFalse(out["reopen"]["ats"]["met"])       # n=1 < 2
            # ML shadow: A on home at -150 won (+0.667), B on away at +130 lost (-1)
            r = out["ml_shadow_value_roi"]
            self.assertEqual(r["n"], 2)
            self.assertAlmostEqual(r["roi_pct"], round(100 * (0.6667 - 1) / 2, 1), places=0)
            self.assertFalse(out["reopen"]["ml_value"]["met"])
            self.assertIn("2026-27", out["by_season"])
            self.assertTrue((d / "frozen_record.json").exists())

    def test_american_profit(self):
        self.assertAlmostEqual(FrozenRecord.american_profit(130, True), 1.3)
        self.assertAlmostEqual(FrozenRecord.american_profit(-150, True), 0.6667, places=3)
        self.assertEqual(FrozenRecord.american_profit(-150, False), -1.0)
        self.assertIsNone(FrozenRecord.american_profit(None, True))


if __name__ == "__main__":
    unittest.main()
