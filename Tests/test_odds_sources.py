"""odds_sources: preferred SBR book → other SBR books → ESPN (2026-10-23 blank-board incident)."""
import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.Utils import odds_sources as os_  # noqa: E402


def sbr_game(home="Boston Celtics", away="New York Knicks", books=None):
    books = books if books is not None else {"fanduel": (-2.5, 223.5, -130, 110)}
    g = {"home_team": home, "away_team": away, "home_score": 0, "away_score": 0,
         "home_spread": {}, "away_spread": {}, "total": {}, "home_ml": {}, "away_ml": {}}
    for book, (home_spread, total, ml_home, ml_away) in books.items():
        g["home_spread"][book] = home_spread
        g["away_spread"][book] = -home_spread
        g["total"][book] = total
        g["home_ml"][book] = ml_home
        g["away_ml"][book] = ml_away
    return g


class TestSbrLine(unittest.TestCase):
    def test_preferred_book_wins(self):
        line = os_.sbr_line(sbr_game(books={"draftkings": (-3.5, 220.5, -150, 130),
                                            "fanduel": (-2.5, 223.5, -130, 110)}))
        self.assertEqual(line["source"], "sbr_fanduel")
        self.assertEqual(line["spread"], 2.5)   # away spread: positive = home favoured
        self.assertEqual(line["ou"], 223.5)

    def test_falls_back_to_another_book(self):
        line = os_.sbr_line(sbr_game(books={"caesars": (-7, 225.5, -260, 210)}))
        self.assertEqual(line["source"], "sbr_caesars")
        self.assertEqual(line["spread"], 7)

    def test_unpriced_game_is_none(self):
        self.assertIsNone(os_.sbr_line(sbr_game(books={})))


class TestLinesForGames(unittest.TestCase):
    TODAY = date(2026, 10, 23)

    def espn(self, _d):
        self.espn_calls += 1
        return [
            {"home_team": "Milwaukee Bucks", "away_team": "Chicago Bulls", "ou": 240.5, "spread": 1.5,
             "ml_home": -115, "ml_away": -105, "source": "espn_draftkings", "home_score": 0, "away_score": 0,
             "espn_id": "1"},
            {"home_team": "Boston Celtics", "away_team": "New York Knicks", "ou": 221.5, "spread": 2.5,
             "ml_home": -148, "ml_away": 124, "source": "espn_draftkings", "home_score": 0, "away_score": 0,
             "espn_id": "2"},
        ]

    def setUp(self):
        self.espn_calls = 0

    def test_all_priced_never_touches_espn(self):
        lines = os_.lines_for_games([sbr_game()], self.TODAY, espn_fetch=self.espn, today=self.TODAY)
        self.assertEqual([l["source"] for l in lines], ["sbr_fanduel"])
        self.assertEqual(self.espn_calls, 0)

    def test_missing_games_filled_from_espn_keeping_sbr_names(self):
        games = [sbr_game(), sbr_game(home="Milwaukee Bucks", away="Chicago Bulls", books={})]
        lines = os_.lines_for_games(games, self.TODAY, espn_fetch=self.espn, today=self.TODAY, log=lambda *_: None)
        self.assertEqual(self.espn_calls, 1)
        by_home = {l["home_team"]: l for l in lines}
        self.assertEqual(by_home["Boston Celtics"]["source"], "sbr_fanduel")
        self.assertEqual(by_home["Milwaukee Bucks"]["source"], "espn_draftkings")
        self.assertEqual(by_home["Milwaukee Bucks"]["spread"], 1.5)
        self.assertNotIn("espn_id", by_home["Milwaukee Bucks"])

    def test_empty_sbr_board_uses_espn_slate(self):
        lines = os_.lines_for_games([], self.TODAY, espn_fetch=self.espn, today=self.TODAY, log=lambda *_: None)
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(l["source"] == "espn_draftkings" for l in lines))

    def test_old_dates_never_hit_espn(self):
        lines = os_.lines_for_games([sbr_game(books={})], date(2025, 1, 1), espn_fetch=self.espn, today=self.TODAY)
        self.assertEqual(lines, [])
        self.assertEqual(self.espn_calls, 0)

    def test_espn_failure_keeps_sbr_lines(self):
        def boom(_d):
            raise RuntimeError("down")
        games = [sbr_game(), sbr_game(home="Milwaukee Bucks", away="Chicago Bulls", books={})]
        lines = os_.lines_for_games(games, self.TODAY, espn_fetch=boom, today=self.TODAY, log=lambda *_: None)
        self.assertEqual([l["home_team"] for l in lines], ["Boston Celtics"])


class TestEspnParsing(unittest.TestCase):
    def test_home_spread_is_negated_and_labels(self):
        self.assertEqual(os_.source_label("sbr_fanduel_open"), "FanDuel")
        self.assertEqual(os_.source_label("espn_draftkings"), "DraftKings (ESPN)")
        self.assertIsNone(os_.source_label(None))
        self.assertEqual(os_._num("-2.5"), -2.5)
        self.assertEqual(os_._num("240.5"), 240.5)
        self.assertEqual(os_._int("-115"), -115)


if __name__ == "__main__":
    unittest.main()
