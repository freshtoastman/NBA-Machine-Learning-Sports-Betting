import importlib.util
import sqlite3
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

CREATE_GAMES_PATH = Path(__file__).resolve().parents[1] / "src" / "Process-Data" / "Create_Games.py"
SPEC = importlib.util.spec_from_file_location("create_games", CREATE_GAMES_PATH)
create_games = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(create_games)


class TestCreateGames(unittest.TestCase):

    def test_normalize_date(self):
        self.assertEqual(create_games.normalize_date("2025-01-02"), "2025-01-02")
        self.assertEqual(create_games.normalize_date(datetime(2025, 1, 2, 12, 0, 0)), "2025-01-02")

    def test_get_team_index_map(self):
        self.assertIs(create_games.get_team_index_map("2023-24"), create_games.team_index_current)
        self.assertIs(create_games.get_team_index_map("2015-16"), create_games.team_index_14)
        self.assertIs(create_games.get_team_index_map("bad-key"), create_games.team_index_current)

    @staticmethod
    def _stats_table(con, name, games_played):
        """30-team stats table; teams 0 and 1 are 'Team A' and 'Team B'."""
        gp = [0] * 30
        for idx, value in games_played.items():
            gp[idx] = value
        pd.DataFrame(
            {"TEAM_ID": list(range(30)), "TEAM_NAME": [f"Team {i}" for i in range(30)], "GP": gp}
        ).to_sql(name, con, if_exists="replace", index=False)

    @staticmethod
    def _odds(rows):
        return pd.DataFrame(rows, columns=["Date", "Home", "Away", "Points"])

    def test_pregame_snapshot_skips_backfilled_same_day_table(self):
        index_map = {"Team A": 0, "Team B": 1}
        odds = self._odds([
            ("2025-01-01", "Team A", "Team B", 200.0),
            ("2025-01-03", "Team A", "Team B", 210.0),
        ])
        with sqlite3.connect(":memory:") as con:
            self._stats_table(con, "2025-01-02", {0: 1, 1: 1})
            # Backfilled: already counts the 01-03 game.
            self._stats_table(con, "2025-01-03", {0: 2, 1: 2})
            self._stats_table(con, "2025-01-04", {0: 2, 1: 2})
            picker = create_games.PregameSnapshots(con, "2024-10-22", odds, index_map)
            self.assertEqual(picker.pick("2025-01-03"), "2025-01-02")

    def test_pregame_snapshot_uses_newest_table_without_the_game(self):
        index_map = {"Team A": 0, "Team B": 1}
        odds = self._odds([
            ("2025-01-01", "Team A", "Team B", 200.0),
            ("2025-01-03", "Team A", "Team B", 210.0),
        ])
        with sqlite3.connect(":memory:") as con:
            self._stats_table(con, "2025-01-02", {0: 1, 1: 1})
            # Fetched before tip-off: still one game each.
            self._stats_table(con, "2025-01-03", {0: 1, 1: 1})
            picker = create_games.PregameSnapshots(con, "2024-10-22", odds, index_map)
            self.assertEqual(picker.pick("2025-01-03"), "2025-01-03")
            # A next-day table fetched before tip-off is fresher still.
            self._stats_table(con, "2025-01-04", {0: 1, 1: 1})
            picker = create_games.PregameSnapshots(con, "2024-10-22", odds, index_map)
            self.assertEqual(picker.pick("2025-01-03"), "2025-01-04")

    def test_pregame_snapshot_survives_game_missing_from_odds(self):
        # Team A's opener is absent from the odds table; the anchor table (idle on
        # 01-04 and 01-05) still tells us it had played twice before 01-06.
        index_map = {"Team A": 0, "Team B": 1}
        odds = self._odds([
            ("2025-01-03", "Team A", "Team B", 200.0),
            ("2025-01-06", "Team A", "Team B", 210.0),
        ])
        with sqlite3.connect(":memory:") as con:
            self._stats_table(con, "2025-01-05", {0: 2, 1: 1})
            self._stats_table(con, "2025-01-06", {0: 2, 1: 1})
            picker = create_games.PregameSnapshots(con, "2024-10-22", odds, index_map)
            self.assertEqual(picker.pick("2025-01-06"), "2025-01-06")

    def test_pregame_snapshot_stays_inside_season(self):
        odds = self._odds([("2025-10-21", "Team A", "Team B", None)])
        with sqlite3.connect(":memory:") as con:
            self._stats_table(con, "2025-06-01", {0: 82, 1: 82})
            picker = create_games.PregameSnapshots(con, "2025-10-21", odds, {"Team A": 0, "Team B": 1})
            self.assertIsNone(picker.pick("2025-10-21"))

    def test_build_game_features(self):
        team_df = pd.DataFrame(
            {
                "TEAM_ID": list(range(30)),
                "TEAM_NAME": [f"Team {i}" for i in range(30)],
                "STAT_A": list(range(30)),
            }
        )
        index_map = {"Team A": 0, "Team B": 1}
        features = create_games.build_game_features(team_df, "Team A", "Team B", index_map)
        self.assertIsNotNone(features)
        self.assertIn("TEAM_NAME", features.index)
        self.assertIn("TEAM_NAME.1", features.index)

    def test_fetch_team_table_missing(self):
        with sqlite3.connect(":memory:") as con:
            result = create_games.fetch_team_table(con, "2025-01-02")
        self.assertIsNone(result)

    def test_main_writes_dataset(self):
        config = {
            "create-games": {
                "2023-24": {
                    "start_date": "2023-10-23",
                    "end_date": "2024-04-28",
                }
            }
        }
        odds_rows = [
            {
                "Date": "2025-01-02",
                "Home": "Team A",
                "Away": "Team B",
                "OU": 210.5,
                "Points": 205.0,
                "Win_Margin": 5,
                "Days_Rest_Home": 2,
                "Days_Rest_Away": 4,
            }
        ]
        team_df = pd.DataFrame(
            {
                "TEAM_ID": list(range(30)),
                "TEAM_NAME": [f"Team {i}" for i in range(30)],
                "GP": [0] * 30,
                "STAT_A": list(range(30)),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            odds_path = Path(tmpdir) / "odds.sqlite"
            teams_path = Path(tmpdir) / "teams.sqlite"
            out_path = Path(tmpdir) / "out.sqlite"

            with sqlite3.connect(odds_path) as con:
                pd.DataFrame(odds_rows).to_sql("2023-24", con, if_exists="replace", index=False)
            with sqlite3.connect(teams_path) as con:
                team_df.to_sql("2025-01-02", con, if_exists="replace", index=False)

            original_config_path = create_games.CONFIG_PATH
            original_odds_path = create_games.ODDS_DB_PATH
            original_teams_path = create_games.TEAMS_DB_PATH
            original_output_path = create_games.OUTPUT_DB_PATH
            original_output_table = create_games.OUTPUT_TABLE
            original_advanced_path = create_games.ADVANCED_DB_PATH
            original_team_map = create_games.TEAM_INDEX_BY_SEASON

            try:
                create_games.CONFIG_PATH = Path(tmpdir) / "config.toml"
                create_games.ODDS_DB_PATH = odds_path
                create_games.TEAMS_DB_PATH = teams_path
                create_games.OUTPUT_DB_PATH = out_path
                create_games.ADVANCED_DB_PATH = Path(tmpdir) / "missing_advanced.sqlite"
                create_games.OUTPUT_TABLE = "dataset_test"
                create_games.TEAM_INDEX_BY_SEASON = {"2023-24": {"Team A": 0, "Team B": 1}}

                create_games.toml.dump(config, create_games.CONFIG_PATH.open("w"))
                create_games.main()

                with sqlite3.connect(out_path) as con:
                    df = pd.read_sql_query('SELECT * FROM "dataset_test"', con)
                    used = pd.read_sql_query('SELECT * FROM "dataset_test_snapshots"', con)
            finally:
                create_games.CONFIG_PATH = original_config_path
                create_games.ODDS_DB_PATH = original_odds_path
                create_games.TEAMS_DB_PATH = original_teams_path
                create_games.OUTPUT_DB_PATH = original_output_path
                create_games.OUTPUT_TABLE = original_output_table
                create_games.ADVANCED_DB_PATH = original_advanced_path
                create_games.TEAM_INDEX_BY_SEASON = original_team_map

        self.assertEqual(len(df.index), 1)
        self.assertIn("Score", df.columns)
        self.assertIn("OU-Cover", df.columns)
        self.assertEqual(used["team_table"].tolist(), ["2025-01-02"])
