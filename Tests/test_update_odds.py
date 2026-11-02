"""update_odds.update_odds_for_date must not treat a post-tip board as a line.

2026-11-01: BOS @ ORL tipped 20:30Z; the 22:01Z `update_odds --db-only` snapshot
overwrote the season row (-6.5 → -7.5, an in-game line) and logged it to
odds_history. 2026-10-31: POR @ UTA was already final (Points filled) so the
UPDATE matched nothing, yet odds_history still got a stray 22:03Z row.
"""
import importlib.util
import sqlite3
import unittest
from datetime import date
from pathlib import Path

_UPD = Path(__file__).resolve().parents[1] / "scripts" / "update_odds.py"
_spec = importlib.util.spec_from_file_location("update_odds", _UPD)
update_odds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(update_odds)

D = date(2026, 11, 1)
SEASON = "2026-27"


def _db():
    con = sqlite3.connect(":memory:")
    con.execute(
        f'CREATE TABLE "{SEASON}" (Date TEXT, Home TEXT, Away TEXT, OU REAL, Spread REAL, '
        "ML_Home INTEGER, ML_Away INTEGER, Points INTEGER, Win_Margin INTEGER)"
    )
    con.execute(f'CREATE TABLE "odds_{SEASON}" (Date TEXT, Home TEXT, Away TEXT, OU REAL, Spread REAL, ML_Home INTEGER, ML_Away INTEGER)')
    update_odds._ensure_history_table(con)
    rows = [
        ("2026-11-01", "Orlando Magic", "Boston Celtics", 222.5, -6.5, 205, -250, None, None),   # tipped, no final yet
        ("2026-11-01", "Utah Jazz", "Portland Trail Blazers", 231.5, -8.5, 260, -320, 202, -22),  # final
        ("2026-11-01", "Golden State Warriors", "LA Clippers", 218.5, 1.5, -130, 108, None, None),  # not tipped
    ]
    con.executemany(f'INSERT INTO "{SEASON}" VALUES (?,?,?,?,?,?,?,?,?)', rows)
    return con


def _line(home, away, spread, ou=220.5, mlh=-150, mla=130):
    return {"home_team": home, "away_team": away, "spread": spread, "ou": ou,
            "ml_home": mlh, "ml_away": mla, "source": "sbr_fanduel"}


class UpdateOddsPostTipTest(unittest.TestCase):
    def setUp(self):
        self._orig = update_odds.lines_for_games
        update_odds.lines_for_games = lambda games, d, book: [
            _line("Orlando Magic", "Boston Celtics", -7.5),
            _line("Utah Jazz", "Portland Trail Blazers", -6.5),
            _line("Golden State Warriors", "LA Clippers", 2.5),
        ]

    def tearDown(self):
        update_odds.lines_for_games = self._orig

    def test_started_game_is_not_refreshed_or_logged(self):
        con = _db()
        n = update_odds.update_odds_for_date(
            con, SEASON, D, games=[], fetched_at_iso="2026-11-01T22:01:28+00:00",
            started={("Orlando Magic", "Boston Celtics")},
        )
        self.assertEqual(n, 1)  # only GSW
        spread = con.execute(f'SELECT Spread FROM "{SEASON}" WHERE Home="Orlando Magic"').fetchone()[0]
        self.assertEqual(spread, -6.5)
        hist = con.execute("SELECT Home FROM odds_history ORDER BY Home").fetchall()
        self.assertEqual(hist, [("Golden State Warriors",)])

    def test_final_game_gets_no_history_row(self):
        con = _db()
        update_odds.update_odds_for_date(con, SEASON, D, games=[], fetched_at_iso="2026-11-01T22:03:09+00:00")
        homes = {r[0] for r in con.execute("SELECT Home FROM odds_history")}
        self.assertNotIn("Utah Jazz", homes)
        self.assertIn("Golden State Warriors", homes)

    def test_started_games_uses_schedule_tipoff(self):
        import src.Utils.nba_cdn as cdn
        orig = cdn.games_for_date
        cdn.games_for_date = lambda d, season_only=True: [
            {"home": "Orlando Magic", "away": "Boston Celtics", "tipoff_utc": "2026-11-01T20:30:00Z"},
            {"home": "Golden State Warriors", "away": "LA Clippers", "tipoff_utc": "2026-11-02T01:30:00Z"},
        ]
        try:
            s = update_odds.started_games(D, "2026-11-01T22:01:28+00:00")
        finally:
            cdn.games_for_date = orig
        self.assertEqual(s, {("Orlando Magic", "Boston Celtics")})

    def test_started_games_empty_when_schedule_unreachable(self):
        import src.Utils.nba_cdn as cdn
        orig = cdn.games_for_date

        def boom(d, season_only=True):
            raise OSError("offline")
        cdn.games_for_date = boom
        try:
            self.assertEqual(update_odds.started_games(D, "2026-11-01T22:01:28+00:00"), set())
        finally:
            cdn.games_for_date = orig


if __name__ == "__main__":
    unittest.main()
