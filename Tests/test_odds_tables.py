import sqlite3
import unittest

import pandas as pd

from src.Utils.OddsTables import load_season_odds, load_seasons_odds, season_odds_tables

COLUMNS = ["Date", "Home", "Away", "OU", "Spread", "ML_Home", "ML_Away",
           "Points", "Win_Margin", "Days_Rest_Home", "Days_Rest_Away"]


def _game(date, home, away, points=200.0, margin=5.0, spread=3.5):
    return {"Date": date, "Home": home, "Away": away, "OU": 220.5, "Spread": spread,
            "ML_Home": -150, "ML_Away": 130, "Points": points, "Win_Margin": margin,
            "Days_Rest_Home": 2, "Days_Rest_Away": 1}


def _write(con, table, games):
    pd.DataFrame(games, columns=COLUMNS).to_sql(table, con, if_exists="replace", index=False)


class TestOddsTables(unittest.TestCase):

    def test_missing_season_returns_empty_frame(self):
        with sqlite3.connect(":memory:") as con:
            self.assertEqual(season_odds_tables(con, "2030-31"), [])
            self.assertTrue(load_season_odds(con, "2030-31").empty)

    def test_partial_table_with_latest_date_does_not_replace_full_table(self):
        full = [_game(f"2025-11-{d:02d}", "Team A", "Team B") for d in range(1, 21)]
        # Stale copy: first three games plus a hand-inserted later game.
        partial = full[:3] + [_game("2026-06-13", "Team C", "Team D")]
        with sqlite3.connect(":memory:") as con:
            _write(con, "2025-26", full)
            _write(con, "odds_2025-26", partial)
            merged = load_season_odds(con, "2025-26")
        self.assertEqual(len(merged), 21)
        self.assertEqual(merged["Date"].iloc[-1], "2026-06-13")
        self.assertFalse(merged.duplicated(["Date", "Home", "Away"]).any())

    def test_scored_row_wins_over_unscored_duplicate(self):
        with sqlite3.connect(":memory:") as con:
            _write(con, "2025-26", [_game("2025-11-01", "Team A", "Team B", points=None, margin=None),
                                    _game("2025-11-02", "Team C", "Team D")])
            _write(con, "odds_2025-26", [_game("2025-11-01", "Team A", "Team B", points=211.0, margin=-7.0)])
            merged = load_season_odds(con, "2025-26")
        row = merged[merged["Date"] == "2025-11-01"].iloc[0]
        self.assertEqual(len(merged), 2)
        self.assertEqual(row["Points"], 211.0)
        self.assertEqual(row["Win_Margin"], -7.0)

    def test_team_alias_and_shifted_dates_do_not_duplicate_games(self):
        base = [_game("2023-11-01", "LA Clippers", "Team B"), _game("2023-11-05", "Team C", "Team D"),
                _game("2023-11-07", "Team E", "Team F")]
        other = [_game("2023-11-01", "Los Angeles Clippers", "Team B"),
                 _game("2023-11-06", "Team C", "Team D"),      # same game, date off by one
                 _game("2023-11-20", "Team G", "Team H")]      # genuinely missing from base
        with sqlite3.connect(":memory:") as con:
            _write(con, "2023-24", base + [_game("2023-11-09", "Team I", "Team J")])
            _write(con, "odds_2023-24_new", other)
            merged = load_season_odds(con, "2023-24")
        self.assertEqual(len(merged), 5)
        self.assertEqual(sorted(merged["Home"][merged["Home"].str.contains("Clippers")]), ["LA Clippers"])
        self.assertIn("2023-11-20", merged["Date"].tolist())
        self.assertNotIn("2023-11-06", merged["Date"].tolist())

    def test_unusable_legacy_dates_are_dropped(self):
        with sqlite3.connect(":memory:") as con:
            _write(con, "odds_2007-08", [_game("2007-08-0102", "Team A", "Team B")])
            _write(con, "odds_2007-08_new", [_game("2007-10-30 00:00:00", "Team A", "Team B")])
            merged = load_seasons_odds(con, ["2007-08", "2008-09"], ["Date", "Home", "Away"])
        self.assertEqual(merged.values.tolist(), [["2007-10-30", "Team A", "Team B"]])

    def test_unsigned_legacy_spreads_are_signed_from_the_moneylines(self):
        games = []
        for d in range(60):
            home_fav = d % 3 != 0
            g = _game(f"2016-{11 + d // 30:02d}-{d % 30 + 1:02d}", f"Team {d}", "Team X", spread=4.5)
            g["ML_Home"], g["ML_Away"] = (-180, 160) if home_fav else (160, -180)
            games.append(g)
        games[1]["ML_Home"] = games[1]["ML_Away"] = -110          # side unknown
        games[2]["Spread"] = 0.0                                   # pick'em stays 0
        with sqlite3.connect(":memory:") as con:
            _write(con, "odds_2016-17_new", games)
            merged = load_season_odds(con, "2016-17")
        self.assertEqual(merged["Spread"].iloc[0], -4.5)           # home underdog
        self.assertTrue(pd.isna(merged["Spread"].iloc[1]))
        self.assertEqual(merged["Spread"].iloc[2], 0.0)
        self.assertEqual(merged["Spread"].iloc[4], 4.5)            # home favourite
        self.assertEqual(int((merged["Spread"] < 0).sum()), 20)    # every third game

    def test_home_referenced_tables_keep_their_spreads(self):
        games = []
        for d in range(60):
            g = _game(f"2025-{11 + d // 30:02d}-{d % 30 + 1:02d}", f"Team {d}", "Team X",
                      spread=-3.5 if d % 3 == 0 else 6.0)
            # Moneylines that disagree with the spread must not flip a signed table.
            g["ML_Home"], g["ML_Away"] = 150, -170
            games.append(g)
        with sqlite3.connect(":memory:") as con:
            _write(con, "2025-26", games)
            merged = load_season_odds(con, "2025-26")
        self.assertEqual(sorted(merged["Spread"].unique()), [-3.5, 6.0])
        self.assertEqual(int((merged["Spread"] < 0).sum()), 20)

    def test_short_table_of_home_favourites_is_left_alone(self):
        games = [_game(f"2026-10-{20 + d}", f"Team {d}", "Team X", spread=5.5) for d in range(8)]
        for g in games:
            g["ML_Home"], g["ML_Away"] = 170, -200
        with sqlite3.connect(":memory:") as con:
            _write(con, "2026-27", games)
            merged = load_season_odds(con, "2026-27")
        self.assertTrue((merged["Spread"] == 5.5).all())


if __name__ == "__main__":
    unittest.main()
