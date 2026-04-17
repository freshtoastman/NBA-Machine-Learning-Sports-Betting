import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import toml

from src.Utils.Dictionaries import (
    team_index_07,
    team_index_08,
    team_index_12,
    team_index_13,
    team_index_14,
    team_index_current,
)

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "config.toml"
ODDS_DB_PATH = BASE_DIR / "Data" / "OddsData.sqlite"
TEAMS_DB_PATH = BASE_DIR / "Data" / "TeamData.sqlite"
ADVANCED_DB_PATH = BASE_DIR / "Data" / "AdvancedTeamData.sqlite"
OUTPUT_DB_PATH = BASE_DIR / "Data" / "dataset.sqlite"
OUTPUT_TABLE = "dataset_2012-26"

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
    if not table_exists(teams_con, date_str):
        return None
    return pd.read_sql_query(f'SELECT * FROM "{date_str}"', teams_con)


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


def select_odds_table(odds_con, season_key):
    """Pick the freshest odds table for a season.

    Multiple legacy table-name conventions exist (e.g. `odds_2025-26`,
    `2025-26`, `odds_2025-26_new`). When more than one is present we choose
    the one with the most recent Date so we don't accidentally pick a
    stale snapshot from an earlier ingest.
    """
    candidates = [
        f"odds_{season_key}_new",
        f"odds_{season_key}",
        f"{season_key}_new",
        f"{season_key}",
    ]
    existing = [t for t in candidates if table_exists(odds_con, t)]
    if not existing:
        return None
    if len(existing) == 1:
        return existing[0]

    def freshness(table_name):
        try:
            row = odds_con.execute(f'SELECT MAX(Date) FROM "{table_name}"').fetchone()
            return row[0] or ""
        except Exception:
            return ""

    return max(existing, key=freshness)


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
        if margin is not None and points not in (None, 0):
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

    # Open advanced stats DB if it exists; gracefully degrade if missing.
    adv_con = None
    if ADVANCED_DB_PATH.exists():
        adv_con = sqlite3.connect(ADVANCED_DB_PATH)

    with sqlite3.connect(ODDS_DB_PATH) as odds_con, sqlite3.connect(TEAMS_DB_PATH) as teams_con:
        for season_key in config["create-games"].keys():
            print(season_key)
            odds_table = select_odds_table(odds_con, season_key)
            if not odds_table:
                print(f"Missing odds tables for {season_key}.")
                continue

            odds_df = pd.read_sql_query(f'SELECT * FROM "{odds_table}"', odds_con)
            if odds_df.empty:
                print(f"No odds data for {season_key}.")
                continue

            index_map = get_team_index_map(season_key)
            playoff_window = playoff_windows.get(season_key)
            per_game_series = _build_per_game_series_state(odds_df, playoff_window)

            for row in odds_df.itertuples(index=False):
                date_str = normalize_date(row.Date)
                team_df = fetch_team_table(teams_con, date_str)
                if team_df is None:
                    continue

                game = build_game_features(team_df, row.Home, row.Away, index_map)
                if game is None:
                    continue

                # Merge advanced efficiency stats if available for this date.
                adv_df = fetch_advanced_table(adv_con, date_str)
                game = _merge_advanced_into_game(game, adv_df, row.Home, row.Away, index_map)

                scores.append(row.Points)
                ou_values.append(row.OU)
                days_rest_home.append(row.Days_Rest_Home)
                days_rest_away.append(row.Days_Rest_Away)
                win_margin.append(1 if row.Win_Margin > 0 else 0)

                if row.Points < row.OU:
                    ou_cover.append(0)
                elif row.Points > row.OU:
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
    n_adv = sum(1 for c in frame.columns if c.startswith("ADV_"))
    n_playoff = sum(is_playoff_list)
    print(f"Wrote {len(games)} rows to {OUTPUT_TABLE} (regular={len(games) - n_playoff}, playoff={n_playoff}, adv_cols={n_adv})")


if __name__ == "__main__":
    main()
