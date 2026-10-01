import bisect
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import toml

sys.path.insert(1, str(Path(__file__).resolve().parents[2]))

from src.Utils.Dictionaries import (
    team_index_07,
    team_index_08,
    team_index_12,
    team_index_13,
    team_index_14,
    team_index_current,
)
from src.Utils.OddsTables import canonical_team, load_season_odds

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "config.toml"
ODDS_DB_PATH = BASE_DIR / "Data" / "OddsData.sqlite"
TEAMS_DB_PATH = BASE_DIR / "Data" / "TeamData.sqlite"
ADVANCED_DB_PATH = BASE_DIR / "Data" / "AdvancedTeamData.sqlite"
OUTPUT_DB_PATH = BASE_DIR / "Data" / "dataset.sqlite"
OUTPUT_TABLE = "dataset_2012-26"
SNAPSHOT_TABLE_SUFFIX = "_snapshots"

# "pregame": each game is built from the newest stats snapshot that does not yet
# count any game of that day (what a live pre-game run can see).
# "same_day": legacy behaviour — the table named after the game date, which for
# backfilled tables already includes the game being predicted. Kept only so
# scripts/audit_feature_leak.py can reproduce the leaky dataset.
SNAPSHOT_MODE = "pregame"

# NBA Cup championship games are not counted in regular-season GP, so they are
# left out when counting how many games a team has played before a date.
NBA_CUP_FINAL_DATES = {"2023-12-09", "2024-12-17", "2025-12-16"}

# Advanced stat columns we want to merge (subset of MeasureType=Advanced).
# These are the most predictive efficiency metrics not already in Base stats.
ADVANCED_COLS = [
    "TEAM_ID",
    "OFF_RATING", "DEF_RATING", "NET_RATING",
    "PACE", "TS_PCT", "EFG_PCT",
    "OREB_PCT", "DREB_PCT", "REB_PCT",
    "TM_TOV_PCT", "AST_PCT", "AST_TO", "AST_RATIO",
    "PIE",
]

TEAM_INDEX_BY_SEASON = {
    "2007-08": team_index_07,
    "2008-09": team_index_08,
    "2009-10": team_index_08,
    "2010-11": team_index_08,
    "2011-12": team_index_08,
    "2012-13": team_index_12,
    "2013-14": team_index_13,
    "2014-15": team_index_14,
    "2015-16": team_index_14,
    "2016-17": team_index_14,
    "2017-18": team_index_14,
    "2018-19": team_index_14,
    "2019-20": team_index_14,
    "2020-21": team_index_14,
    "2021-22": team_index_14,
    "2022-23": team_index_current,
    "2023-24": team_index_current,
    "2024-25": team_index_current,
    "2025-26": team_index_current,
    "2026-27": team_index_current,
}


def table_exists(con, table_name):
    cursor = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    return cursor.fetchone() is not None


def normalize_date(value):
    if isinstance(value, datetime):
        return value.date().isoformat()
    if hasattr(value, "date"):
        try:
            return value.date().isoformat()
        except Exception:
            pass
    return str(value)


def get_team_index_map(season_key):
    if season_key in TEAM_INDEX_BY_SEASON:
        return TEAM_INDEX_BY_SEASON[season_key]
    try:
        start_year = int(season_key.split("-")[0])
    except (ValueError, IndexError):
        return team_index_current
    return team_index_current if start_year >= 2022 else team_index_14


def fetch_team_table(teams_con, date_str):
    if table_exists(teams_con, date_str):
        return pd.read_sql_query(f'SELECT * FROM "{date_str}"', teams_con)
    # For future dates (upcoming scheduled games), fall back to the most recent snapshot.
    # Only consider plain YYYY-MM-DD tables (not _playoff suffix) so features match training.
    try:
        cursor = teams_con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name <= ? AND name GLOB '????-??-??' ORDER BY name DESC LIMIT 1",
            (date_str,),
        )
        row = cursor.fetchone()
        if row:
            return pd.read_sql_query(f'SELECT * FROM "{row[0]}"', teams_con)
    except Exception:
        pass
    return None


def fetch_advanced_table(adv_con, date_str):
    """Return advanced stats DataFrame for a date, or None if unavailable."""
    if adv_con is None:
        return None
    if not table_exists(adv_con, date_str):
        return None
    df = pd.read_sql_query(f'SELECT * FROM "{date_str}"', adv_con)
    # Keep only the columns we care about (some API versions may differ).
    available = [c for c in ADVANCED_COLS if c in df.columns]
    return df[available] if available else None


def _merge_advanced_into_game(game_series, adv_df, home_team, away_team, index_map):
    """Append advanced stat columns (prefixed A_) for home and away teams."""
    if adv_df is None:
        return game_series
    home_idx = index_map.get(home_team)
    away_idx = index_map.get(away_team)
    if home_idx is None or away_idx is None:
        return game_series
    if len(adv_df) != 30:
        return game_series

    cols = [c for c in adv_df.columns if c != "TEAM_ID"]
    home_adv = adv_df.iloc[home_idx]
    away_adv = adv_df.iloc[away_idx]

    for col in cols:
        game_series[f"ADV_{col}"] = home_adv[col]
        game_series[f"ADV_{col}.1"] = away_adv[col]
    return game_series


def build_game_features(team_df, home_team, away_team, index_map):
    home_index = index_map.get(home_team)
    away_index = index_map.get(away_team)
    if home_index is None or away_index is None:
        return None
    if len(team_df.index) != 30:
        return None

    home_team_series = team_df.iloc[home_index]
    away_team_series = team_df.iloc[away_index]
    return pd.concat([
        home_team_series,
        away_team_series.rename(index={col: f"{col}.1" for col in team_df.columns.values}),
    ])


class PregameSnapshots:
    """Pick, per game date, the newest stats table that excludes that day's games.

    A table named D is fetched with DateTo=D. Fetched live before tip-off it holds
    results through D-1; backfilled later it already contains D's games, which
    leaks the outcome into the features. Table names alone cannot tell the two
    apart, so each candidate is checked by content: no team playing on D may show
    more GP than the games it had completed before D. Candidates are the tables
    named D+1 and D (newest first); the first table named before D is accepted
    as-is because its DateTo bound already excludes D.
    """

    ANCHOR_LOOKBACK = 30

    def __init__(self, con, season_start, season_odds, index_map=None):
        self.con = con
        self.season_start = season_start
        self.index_map = index_map or {}
        self.tables = []
        if con is not None:
            self.tables = sorted(
                row[0] for row in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name GLOB '????-??-??' AND name >= ?",
                    (season_start,),
                )
            )
        self._frames = {}
        self._games_played = {}
        self._picked = {}
        self._played_dates = {}
        self._scheduled = {}
        self._teams_on = {}
        for row in season_odds.itertuples(index=False):
            date_str = normalize_date(row.Date)
            teams = (row.Home, row.Away)
            self._teams_on.setdefault(date_str, set()).update(teams)
            for team in map(canonical_team, teams):
                self._scheduled.setdefault(team, set()).add(date_str)
            points = getattr(row, "Points", None)
            if points is None or not points > 0 or date_str in NBA_CUP_FINAL_DATES:
                continue
            for team in map(canonical_team, teams):
                self._played_dates.setdefault(team, []).append(date_str)
        for dates in self._played_dates.values():
            dates.sort()

    def frame(self, table):
        if table not in self._frames:
            self._frames[table] = pd.read_sql_query(f'SELECT * FROM "{table}"', self.con)
        return self._frames[table]

    def games_played(self, table, team):
        """GP of `team` in `table`, or None when the table cannot tell."""
        if table not in self._games_played:
            df = self.frame(table)
            if "GP" not in df.columns:
                self._games_played[table] = None
            else:
                by_name = {}
                if "TEAM_NAME" in df.columns:
                    by_name = dict(zip(df["TEAM_NAME"].map(canonical_team), df["GP"]))
                self._games_played[table] = (df["GP"].tolist(), by_name)
        cached = self._games_played[table]
        if cached is None:
            return None
        by_row, by_name = cached
        # Same positional lookup the features use; survives franchise renames.
        if len(by_row) == 30 and team in self.index_map:
            return by_row[self.index_map[team]]
        return by_name.get(canonical_team(team))

    def prior_games(self, team, date_str):
        """Regular-season games `team` had completed before date_str.

        Counted from an anchor: the newest earlier table on whose date and eve the
        team was idle states its GP unambiguously, however that table was fetched.
        Games listed after the anchor are added on top. Anchoring keeps a game
        missing from the odds table from skewing the count for the rest of the season.
        """
        played = self._played_dates.get(canonical_team(team), [])
        scheduled = self._scheduled.get(canonical_team(team), ())
        stop = bisect.bisect_left(self.tables, date_str)
        for table in reversed(self.tables[max(0, stop - self.ANCHOR_LOOKBACK):stop]):
            eve = (datetime.strptime(table, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
            if table in scheduled or eve in scheduled:
                continue
            anchored = self.games_played(table, team)
            if anchored is None:
                continue
            return anchored + bisect.bisect_left(played, date_str) - bisect.bisect_right(played, table)
        return bisect.bisect_left(played, date_str)

    def _excludes_games_on(self, table, date_str):
        self.games_played(table, None)  # fills the cache
        if self._games_played[table] is None:
            return False
        for team in self._teams_on.get(date_str, ()):
            shown = self.games_played(table, team)
            if shown is not None and shown > self.prior_games(team, date_str):
                return False
        return True

    def pick(self, date_str):
        """Name of the snapshot table to use for games on date_str, or None."""
        if date_str in self._picked:
            return self._picked[date_str]
        picked = None
        if date_str >= self.season_start:
            next_day = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            upto = bisect.bisect_right(self.tables, next_day)
            for table in reversed(self.tables[:upto]):
                if table < date_str or self._excludes_games_on(table, date_str):
                    picked = table
                    break
        self._picked[date_str] = picked
        return picked


def _iso_date(value):
    return datetime.strptime(str(value), "%Y-%m-%d").strftime("%Y-%m-%d")


def _load_playoff_windows(config):
    """Return dict {season_key: (start_date, end_date)} for playoff date ranges."""
    out = {}
    for season_key, value in config.get("get-playoffs", {}).items():
        out[season_key] = (
            datetime.strptime(value["start_date"], "%Y-%m-%d").date(),
            datetime.strptime(value["end_date"], "%Y-%m-%d").date(),
        )
    return out


def _build_per_game_series_state(odds_df, playoff_window):
    """Walk playoff games chronologically and compute the live series state at
    the moment each game tipped off (i.e., based only on PRIOR games in the same
    series). Returns dict {(date_str, home, away): (game_num, home_wins_so_far,
    away_wins_so_far, is_elimination)}.
    """
    if playoff_window is None or odds_df.empty:
        return {}
    start, end = playoff_window
    out = {}
    # Filter to playoff window and sort by date.
    rows = []
    for r in odds_df.itertuples(index=False):
        date_str = normalize_date(r.Date)
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        if start <= d <= end:
            rows.append((date_str, r.Home, r.Away, getattr(r, "Win_Margin", 0), getattr(r, "Points", 0)))
    rows.sort(key=lambda x: x[0])

    # Per-matchup running win counters keyed by (team_a, team_b) where team_a
    # alphabetically precedes team_b. The values track accumulated wins for
    # each side based on Win_Margin sign.
    counts = {}  # frozenset → {team_a: int, team_b: int}
    for date_str, home, away, margin, points in rows:
        key = frozenset({home, away})
        teams = sorted([home, away])
        team_a, team_b = teams[0], teams[1]
        prior = counts.get(key, {team_a: 0, team_b: 0})
        # Snapshot BEFORE this game.
        home_wins_so_far = prior[home]
        away_wins_so_far = prior[away]
        game_num = home_wins_so_far + away_wins_so_far + 1
        is_elim = 1 if (home_wins_so_far == 3 and away_wins_so_far < 3) or (away_wins_so_far == 3 and home_wins_so_far < 3) else 0
        out[(date_str, home, away)] = (game_num, home_wins_so_far, away_wins_so_far, is_elim)

        # Update for next iteration: only credit a win if the game was played.
        if pd.notna(margin) and pd.notna(points) and points != 0:
            winner = home if margin > 0 else away
            prior[winner] = prior.get(winner, 0) + 1
        counts[key] = prior
    return out


def main():
    config = toml.load(CONFIG_PATH)
    playoff_windows = _load_playoff_windows(config)

    scores = []
    win_margin = []
    ou_values = []
    ou_cover = []
    games = []
    days_rest_away = []
    days_rest_home = []
    is_playoff_list = []
    series_game_num_list = []
    series_lead_for_home_list = []
    is_elimination_list = []
    snapshots = []

    # Open advanced stats DB if it exists; gracefully degrade if missing.
    adv_con = None
    if ADVANCED_DB_PATH.exists():
        adv_con = sqlite3.connect(ADVANCED_DB_PATH)

    with sqlite3.connect(ODDS_DB_PATH) as odds_con, sqlite3.connect(TEAMS_DB_PATH) as teams_con:
        for season_key in config["create-games"].keys():
            print(season_key)
            odds_df = load_season_odds(odds_con, season_key)
            if odds_df.empty:
                print(f"No odds data for {season_key}.")
                continue

            season_start = _iso_date(config["create-games"][season_key]["start_date"])
            index_map = get_team_index_map(season_key)
            team_snapshots = PregameSnapshots(teams_con, season_start, odds_df, index_map)
            adv_snapshots = PregameSnapshots(adv_con, season_start, odds_df, index_map)
            playoff_window = playoff_windows.get(season_key)
            per_game_series = _build_per_game_series_state(odds_df, playoff_window)

            for row in odds_df.itertuples(index=False):
                date_str = normalize_date(row.Date)
                if SNAPSHOT_MODE == "same_day":
                    team_table = adv_table = date_str
                    team_df = fetch_team_table(teams_con, date_str)
                else:
                    team_table = team_snapshots.pick(date_str)
                    adv_table = adv_snapshots.pick(date_str)
                    team_df = team_snapshots.frame(team_table) if team_table else None
                if team_df is None:
                    continue

                game = build_game_features(team_df, row.Home, row.Away, index_map)
                if game is None:
                    continue

                # Ensure game row carries the actual game date (not the stats snapshot date).
                # Needed when the stats fallback returns an earlier snapshot for future games.
                if "Date" in game.index:
                    game["Date"] = date_str
                if "Date.1" in game.index:
                    game["Date.1"] = date_str

                # Merge advanced efficiency stats if available for this date.
                adv_df = fetch_advanced_table(adv_con, adv_table) if adv_table else None
                game = _merge_advanced_into_game(game, adv_df, row.Home, row.Away, index_map)

                scores.append(row.Points)
                ou_values.append(row.OU)
                days_rest_home.append(row.Days_Rest_Home)
                days_rest_away.append(row.Days_Rest_Away)
                win_margin.append(1 if row.Win_Margin > 0 else 0)

                points = getattr(row, "Points", None)
                ou = getattr(row, "OU", None)
                if points is None or ou is None or points == 0:
                    ou_cover.append(2)  # unplayed / unknown
                elif points < ou:
                    ou_cover.append(0)
                elif points > ou:
                    ou_cover.append(1)
                else:
                    ou_cover.append(2)

                # Playoff flagging.
                row_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                is_po = bool(playoff_window and playoff_window[0] <= row_date <= playoff_window[1])
                is_playoff_list.append(1 if is_po else 0)

                # Per-game series snapshot (state BEFORE this game tipped off).
                snapshot = per_game_series.get((date_str, row.Home, row.Away)) if is_po else None
                if snapshot:
                    game_num, home_wins, away_wins, is_elim = snapshot
                    series_game_num_list.append(game_num)
                    series_lead_for_home_list.append(home_wins - away_wins)
                    is_elimination_list.append(is_elim)
                else:
                    series_game_num_list.append(0)
                    series_lead_for_home_list.append(0)
                    is_elimination_list.append(0)

                games.append(game)
                snapshots.append((date_str, row.Home, row.Away, team_table, adv_table if adv_df is not None else None))

    if not games:
        print("No game rows produced. Check odds and team tables.")
        return

    season = pd.concat(games, ignore_index=True, axis=1).T
    frame = season.drop(columns=["TEAM_ID", "TEAM_ID.1"], errors="ignore")
    frame["Score"] = np.asarray(scores)
    frame["Home-Team-Win"] = np.asarray(win_margin)
    frame["OU"] = np.asarray(ou_values)
    frame["OU-Cover"] = np.asarray(ou_cover)
    frame["Days-Rest-Home"] = np.asarray(days_rest_home)
    frame["Days-Rest-Away"] = np.asarray(days_rest_away)
    frame["is_playoff"] = np.asarray(is_playoff_list)
    frame["series_game_num"] = np.asarray(series_game_num_list)
    frame["series_lead_for_home"] = np.asarray(series_lead_for_home_list)
    frame["is_elimination_game"] = np.asarray(is_elimination_list)

    for field in frame.columns.values:
        if "TEAM_" in field or "Date" in field:
            continue
        frame[field] = frame[field].astype(float)

    if adv_con is not None:
        adv_con.close()

    with sqlite3.connect(OUTPUT_DB_PATH) as con:
        frame.to_sql(OUTPUT_TABLE, con, if_exists="replace", index=False)
        # Which stats snapshot fed each row — lets audits prove rows are pre-game.
        pd.DataFrame(
            snapshots, columns=["Date", "Home", "Away", "team_table", "adv_table"]
        ).to_sql(OUTPUT_TABLE + SNAPSHOT_TABLE_SUFFIX, con, if_exists="replace", index=False)
    n_adv = sum(1 for c in frame.columns if c.startswith("ADV_"))
    n_playoff = sum(is_playoff_list)
    print(f"Wrote {len(games)} rows to {OUTPUT_TABLE} (regular={len(games) - n_playoff}, playoff={n_playoff}, adv_cols={n_adv})")


if __name__ == "__main__":
    main()
