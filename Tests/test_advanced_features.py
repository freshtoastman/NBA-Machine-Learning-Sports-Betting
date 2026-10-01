import os
import sqlite3
import tempfile
import unittest

import numpy as np
import pandas as pd

import src.Utils.AdvancedFeatures as AF

COLUMNS = ["Date", "Home", "Away", "OU", "Spread", "ML_Home", "ML_Away",
           "Points", "Win_Margin", "Days_Rest_Home", "Days_Rest_Away"]
TEAMS = ["Team A", "Team B", "Team C", "Team D"]


def _schedule():
    """Twelve game days of a four-team round robin with varied margins."""
    pairings = [(0, 1, 2, 3), (2, 0, 3, 1), (0, 3, 1, 2)]
    games = []
    for day in range(12):
        h1, a1, h2, a2 = pairings[day % 3]
        date = f"2025-11-{2 * day + 1:02d}"
        for n, (h, a) in enumerate(((h1, a1), (h2, a2))):
            margin = float((-1) ** (day + n) * (3 + (day * 5 + n * 7) % 11))
            games.append({"Date": date, "Home": TEAMS[h], "Away": TEAMS[a], "OU": 220.5,
                          "Spread": float((day + n) % 7 - 3) + 0.5, "ML_Home": -150, "ML_Away": 130,
                          "Points": 210.0 + day + n, "Win_Margin": margin,
                          "Days_Rest_Home": 2, "Days_Rest_Away": 2})
    return games


class TestPendingGames(unittest.TestCase):

    def setUp(self):
        self._db, self._cache = AF.ODDS_DB, AF._FEATURE_TABLE_CACHE
        self._dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        AF.ODDS_DB, AF._FEATURE_TABLE_CACHE = self._db, self._cache
        self._dir.cleanup()

    def _table(self, name, games):
        path = os.path.join(self._dir.name, name)
        with sqlite3.connect(path) as con:
            pd.DataFrame(games, columns=COLUMNS).to_sql("2025-26", con, index=False)
        AF.ODDS_DB = path
        AF.reset_cache()
        return AF.build_feature_table().copy()

    def test_unplayed_game_gets_the_features_it_has_once_played(self):
        games = _schedule()
        last = games[-1]["Date"]
        unplayed = [dict(g, Points=0.0, Win_Margin=0.0) if g["Date"] == last else g for g in games]
        before = self._table("before.sqlite", unplayed)
        after = self._table("after.sqlite", games)
        cols = [c for c in after.columns if c not in ("Team", "Date")]
        a = before[before["Date"] == last].set_index("Team")[cols].sort_index().astype(float)
        b = after[after["Date"] == last].set_index("Team")[cols].sort_index().astype(float)
        self.assertEqual(len(a), 4)
        self.assertFalse(a["form_w_pct_10"].isna().any())
        np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)

    def test_second_unplayed_day_counts_the_first_in_schedule_density(self):
        games = _schedule()
        days = sorted({g["Date"] for g in games})[-2:]
        unplayed = [dict(g, Points=0.0, Win_Margin=0.0) if g["Date"] in days else g for g in games]
        before = self._table("before.sqlite", unplayed)
        after = self._table("after.sqlite", games)
        a = before[before["Date"] == days[1]].set_index("Team").sort_index()
        b = after[after["Date"] == days[1]].set_index("Team").sort_index()
        for col in ("days_since_prev", "b2b", "games_last_4d", "games_last_7d", "road_trip_len"):
            np.testing.assert_allclose(a[col].astype(float), b[col].astype(float))
        # Results of the first unplayed day are unknown, so form stops one game earlier.
        prior = after[after["Date"] == days[0]].set_index("Team").sort_index()
        np.testing.assert_allclose(a["form_w_pct_10"].astype(float), prior["form_w_pct_10"].astype(float))

    def test_stale_unscored_rows_are_not_treated_as_upcoming(self):
        games = _schedule()
        games[0] = dict(games[0], Points=0.0, Win_Margin=0.0)      # postponed in week one
        table = self._table("stale.sqlite", games)
        first = table[table["Date"] == games[0]["Date"]]
        self.assertEqual(set(first["Team"]), {games[1]["Home"], games[1]["Away"]})


if __name__ == "__main__":
    unittest.main()
