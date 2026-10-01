"""Per-team rolling form & schedule-density features.

Computes features the box-score / season-average view doesn't see:
    - Recent form (last 5 / 10 games W%, ATS%, point diff)
    - Schedule density (back-to-back, 3-in-4, road trip length)
    - Hot/cold streaks (current win streak, current ATS streak)

All features are derivable from OddsData.sqlite alone — no external API.
The output is a DataFrame indexed by (Team, Date) with feature columns
that can be merged into any trainer's input.
"""

from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.Utils.OddsTables import load_seasons_odds

ODDS_DB = Path(__file__).resolve().parents[2] / "Data" / "OddsData.sqlite"

SEASON_KEYS = [
    "2007-08", "2008-09", "2009-10", "2010-11", "2011-12",
    "2012-13", "2013-14", "2014-15", "2015-16", "2016-17",
    "2017-18", "2018-19", "2019-20", "2020-21", "2021-22",
    "2022-23", "2023-24", "2024-25", "2025-26", "2026-27",
]


# Feature stems added after the 175-feature ATS model was trained. They must sit
# after every other H_/A_/D_ column so the runner's trailing-column trim removes
# them; left in place they shift the A_/D_ positions and the model reads the
# wrong columns. Any new stem goes here until a model is trained with it.
POST_MODEL_STEMS = {
    "game_num_season", "month_sin", "month_cos",                    # temporal
    "form_ats_pct_home_10", "form_ats_pct_away_10",                 # home/away ATS split
    "form_pts_for_5", "form_pts_against_5", "form_pts_diff_10",     # off/def split
}


def ats_model_columns(columns):
    """H_/A_/D_ columns in the order the production ATS model expects."""
    adv = [c for c in columns if c.startswith(("H_", "A_", "D_"))]
    return ([c for c in adv if c[2:] not in POST_MODEL_STEMS]
            + [c for c in adv if c[2:] in POST_MODEL_STEMS])


# Scheduled games without a score are kept this many days behind the newest result;
# anything older is a postponement, not an upcoming game.
PENDING_LOOKBACK_DAYS = 3
# Completed games per team fed into the pre-game rows (longest window: 10 home games).
PENDING_HISTORY_GAMES = 200
_RESULT_COLUMNS = ["Spread", "Win_Margin", "Points", "TeamScore", "OppScore", "Won", "CoverMargin", "Covered"]
_DENSITY_COLUMNS = ["days_since_prev", "b2b", "games_last_4d", "games_last_7d", "road_trip_len"]


def _load_odds_games():
    """Every season's games split into (completed, scheduled without a result yet)."""
    with sqlite3.connect(ODDS_DB) as con:
        df = load_seasons_odds(
            con, SEASON_KEYS, ["Date", "Home", "Away", "Spread", "Win_Margin", "Points"]
        )
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    done = df["Points"].fillna(0) > 0
    played = df[done].reset_index(drop=True)
    if played.empty:
        return played, played
    cutoff = played["Date"].max() - pd.Timedelta(days=PENDING_LOOKBACK_DAYS)
    pending = df[~done & (df["Date"] > cutoff)].reset_index(drop=True)
    return played, pending


def _load_all_games() -> pd.DataFrame:
    """Concatenate every season's completed games. Used as the foundation for features."""
    return _load_odds_games()[0]


def _team_perspective_long(games: pd.DataFrame) -> pd.DataFrame:
    """Reshape games into one row per (team, game) — both teams' perspective."""
    home = games[["Date", "Home", "Away", "Spread", "Win_Margin", "Points"]].copy()
    home.columns = ["Date", "Team", "Opp", "Spread", "Win_Margin", "Points"]
    home["IsHome"] = 1
    home["TeamScore"] = (home["Points"] + home["Win_Margin"]) / 2
    home["OppScore"] = (home["Points"] - home["Win_Margin"]) / 2
    home["Won"] = (home["Win_Margin"] > 0).astype(int)
    # Home-perspective cover margin = Win_Margin - Spread.
    home["CoverMargin"] = home["Win_Margin"] - home["Spread"]
    # No usable spread -> no cover result (NaN), not a failed cover.
    home["Covered"] = (home["CoverMargin"] > 0).astype(float).where(home["CoverMargin"].notna())

    away = games[["Date", "Home", "Away", "Spread", "Win_Margin", "Points"]].copy()
    away.columns = ["Date", "Opp", "Team", "Spread", "Win_Margin", "Points"]
    away["IsHome"] = 0
    away["TeamScore"] = (away["Points"] - away["Win_Margin"]) / 2
    away["OppScore"] = (away["Points"] + away["Win_Margin"]) / 2
    away["Won"] = (away["Win_Margin"] < 0).astype(int)
    # Away-perspective cover margin: away covers when home_cover < 0 → -home_cover.
    away["CoverMargin"] = -(away["Win_Margin"] - away["Spread"])
    away["Covered"] = (away["CoverMargin"] > 0).astype(float).where(away["CoverMargin"].notna())

    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["Team", "Date"]).reset_index(drop=True)
    return long


def _add_rolling_features(long: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling-window features per team. Uses .shift(1) so the row's
    own outcome doesn't leak into its own features."""
    long = long.copy()
    grp = long.groupby("Team", sort=False)

    # Shift first so the current game's result isn't part of "last N games".
    won_prev = grp["Won"].shift(1)
    cov_prev = grp["Covered"].shift(1)
    pts_for_prev = grp["TeamScore"].shift(1)
    pts_against_prev = grp["OppScore"].shift(1)

    long["form_w_pct_5"] = won_prev.groupby(long["Team"]).transform(
        lambda s: s.rolling(5, min_periods=2).mean()
    )
    long["form_w_pct_10"] = won_prev.groupby(long["Team"]).transform(
        lambda s: s.rolling(10, min_periods=3).mean()
    )
    long["form_ats_pct_10"] = cov_prev.groupby(long["Team"]).transform(
        lambda s: s.rolling(10, min_periods=3).mean()
    )
    long["form_pts_diff_5"] = (pts_for_prev - pts_against_prev).groupby(long["Team"]).transform(
        lambda s: s.rolling(5, min_periods=2).mean()
    )
    long["form_pts_for_5"] = pts_for_prev.groupby(long["Team"]).transform(
        lambda s: s.rolling(5, min_periods=2).mean()
    )
    long["form_pts_against_5"] = pts_against_prev.groupby(long["Team"]).transform(
        lambda s: s.rolling(5, min_periods=2).mean()
    )
    long["form_pts_diff_10"] = (pts_for_prev - pts_against_prev).groupby(long["Team"]).transform(
        lambda s: s.rolling(10, min_periods=3).mean()
    )

    # Streaks: count of consecutive prior games with same outcome.
    def _streak(series, value):
        out = []
        c = 0
        for v in series:
            if pd.isna(v):
                c = 0
                out.append(0)
                continue
            if v == value:
                c += 1
            else:
                c = 0
            out.append(c)
        return pd.Series(out, index=series.index)

    long["streak_w"] = grp["Won"].shift(1).groupby(long["Team"], sort=False).transform(
        lambda s: _streak(s, 1.0)
    )
    long["streak_l"] = grp["Won"].shift(1).groupby(long["Team"], sort=False).transform(
        lambda s: _streak(s, 0.0)
    )
    long["streak_ats_cover"] = grp["Covered"].shift(1).groupby(long["Team"], sort=False).transform(
        lambda s: _streak(s, 1.0)
    )

    # Home/away split ATS cover rates.
    # Tracks how each team covers the spread specifically as the home team
    # vs. as the away team. Model's away picks fail when home teams are hot
    # at home — this splits the form_ats_pct_10 signal by venue context.
    home_long = long[long["IsHome"] == 1].copy().sort_values(["Team", "Date"])
    away_long = long[long["IsHome"] == 0].copy().sort_values(["Team", "Date"])

    home_cov_prev = home_long.groupby("Team")["Covered"].shift(1)
    away_cov_prev = away_long.groupby("Team")["Covered"].shift(1)

    home_long["form_ats_pct_home_10"] = home_cov_prev.groupby(home_long["Team"]).transform(
        lambda s: s.rolling(10, min_periods=3).mean()
    )
    away_long["form_ats_pct_away_10"] = away_cov_prev.groupby(away_long["Team"]).transform(
        lambda s: s.rolling(10, min_periods=3).mean()
    )

    long = long.merge(
        home_long[["Team", "Date", "form_ats_pct_home_10"]],
        on=["Team", "Date"], how="left"
    )
    long = long.merge(
        away_long[["Team", "Date", "form_ats_pct_away_10"]],
        on=["Team", "Date"], how="left"
    )

    # Season position features.
    # game_num_season: how many games this team has already played in this
    # calendar season (0 = first game). Uses the NBA season year (Oct-Sep)
    # so game 70+ = end-of-season where tanking/rest dynamics differ.
    def _season_year(date):
        # Season year is the start calendar year: Oct 2025 -> 2025.
        return date.year if date.month >= 10 else date.year - 1

    long["_season_year"] = long["Date"].apply(_season_year)
    long["game_num_season"] = long.groupby(["Team", "_season_year"]).cumcount()
    long = long.drop(columns=["_season_year"])

    # Month of year (1-12). Sinusoidal encoding to preserve cyclical structure.
    long["month_num"] = long["Date"].dt.month
    long["month_sin"] = np.sin(2 * np.pi * long["month_num"] / 12)
    long["month_cos"] = np.cos(2 * np.pi * long["month_num"] / 12)

    return long


def _add_schedule_density(long: pd.DataFrame) -> pd.DataFrame:
    """Add back-to-back and 3-in-4 schedule indicators per team."""
    long = long.copy()
    long["prev_date"] = long.groupby("Team")["Date"].shift(1)
    long["days_since_prev"] = (long["Date"] - long["prev_date"]).dt.days

    long["b2b"] = (long["days_since_prev"] == 1).astype(int)

    # 3 in 4 nights: count games in trailing 4-day window (excluding today).
    def _count_in_window(group, window_days):
        dates = group["Date"].values
        out = np.zeros(len(dates), dtype=int)
        for i in range(len(dates)):
            cutoff = dates[i] - np.timedelta64(window_days, "D")
            out[i] = int(((dates[:i] >= cutoff) & (dates[:i] < dates[i])).sum())
        return pd.Series(out, index=group.index)

    long["games_last_4d"] = long.groupby("Team", group_keys=False).apply(
        lambda g: _count_in_window(g, 4)
    )
    long["games_last_7d"] = long.groupby("Team", group_keys=False).apply(
        lambda g: _count_in_window(g, 7)
    )

    # Road trip length: consecutive prior away games.
    def _road_trip(group):
        is_away = (group["IsHome"] == 0).astype(int).values
        out = np.zeros(len(is_away), dtype=int)
        c = 0
        for i, v in enumerate(is_away):
            out[i] = c  # length of trip *before* this game
            c = c + 1 if v else 0
        return pd.Series(out, index=group.index)

    long["road_trip_len"] = long.groupby("Team", group_keys=False).apply(_road_trip)

    long = long.drop(columns=["prev_date"], errors="ignore")
    return long


def _pending_feature_rows(played_long: pd.DataFrame, pending: pd.DataFrame) -> pd.DataFrame:
    """Feature rows for scheduled games that have no result yet — what a pre-game
    prediction sees.

    Form and streak features read the team's completed games only. Schedule density
    also counts earlier scheduled-but-unplayed games: the team still plays them.
    """
    sched = _team_perspective_long(pending)
    sched[_RESULT_COLUMNS] = np.nan
    history = played_long[played_long["Team"].isin(sched["Team"].unique())]
    history = history.groupby("Team", sort=False).tail(PENDING_HISTORY_GAMES)

    calendar = pd.concat([history, sched.assign(_pending=True)], ignore_index=True)
    calendar = calendar.sort_values(["Team", "Date"], kind="stable").reset_index(drop=True)
    density = _add_schedule_density(calendar)
    density = density[density["_pending"].eq(True)].set_index(["Team", "Date"])[_DENSITY_COLUMNS]

    out = []
    for team, team_sched in sched.groupby("Team", sort=False):
        team_history = history[history["Team"] == team]
        form_cache = {}
        for i in range(len(team_sched)):
            target = team_sched.iloc[[i]]
            date = target["Date"].iloc[0]
            before = team_history[team_history["Date"] < date]
            # Same completed games and same venue -> same form row (the home/away
            # ATS splits depend on which side the team is on).
            key = (len(before), int(target["IsHome"].iloc[0]))
            if key not in form_cache:
                form_cache[key] = _add_rolling_features(
                    pd.concat([before, target], ignore_index=True)
                ).iloc[[-1]]
            row = form_cache[key].copy()
            row["Date"] = date
            row["month_num"] = date.month
            row["month_sin"] = np.sin(2 * np.pi * date.month / 12)
            row["month_cos"] = np.cos(2 * np.pi * date.month / 12)
            for col in _DENSITY_COLUMNS:
                row[col] = density.loc[(team, date), col]
            out.append(row)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


_FEATURE_TABLE_CACHE: pd.DataFrame | None = None


def build_feature_table() -> pd.DataFrame:
    """Return a (Team, Date)-indexed table with all advanced features.

    Cached at module level — call `reset_cache()` after data refresh.
    """
    global _FEATURE_TABLE_CACHE
    if _FEATURE_TABLE_CACHE is not None:
        return _FEATURE_TABLE_CACHE

    games, pending = _load_odds_games()
    if games.empty:
        _FEATURE_TABLE_CACHE = pd.DataFrame()
        return _FEATURE_TABLE_CACHE

    played_long = _team_perspective_long(games)
    long = _add_rolling_features(played_long)
    long = _add_schedule_density(long)
    if not pending.empty:
        # Without these rows an upcoming game finds no match in merge_into and every
        # rolling feature reaches the model as a zero-filled NaN.
        upcoming = _pending_feature_rows(played_long, pending)
        long = pd.concat([long, upcoming], ignore_index=True)
        long = long.sort_values(["Team", "Date"], kind="stable").reset_index(drop=True)

    feature_cols = [
        "Team", "Date",
        # Legacy 12 features (positions 0-11): preserved for 175-feature model compat.
        "form_w_pct_5", "form_w_pct_10", "form_ats_pct_10", "form_pts_diff_5",
        "streak_w", "streak_l", "streak_ats_cover",
        "days_since_prev", "b2b", "games_last_4d", "games_last_7d",
        "road_trip_len",
        # Home/away split ATS features (positions 12-13): added 2026-04-17.
        # Not in _NEW_ADV_STEMS so they go into adv_old → appear at positions
        # 175-180 for the 175-feature model (safely truncated) and 175-180 for
        # any 181-feature model trained to include them.
        "form_ats_pct_home_10", "form_ats_pct_away_10",
        # Temporal features (positions 14-16): excluded from training via DROP_COLUMNS.
        "game_num_season", "month_sin", "month_cos",
        # Offense/defense split + longer-window diff (positions 17-19): added 2026-04-17.
        # Included in retrain experiments; excluded from 175-feature production model
        # via DROP_COLUMNS (safely truncated by _align_features()).
        "form_pts_for_5", "form_pts_against_5", "form_pts_diff_10",
    ]
    table = long[feature_cols].copy()
    _FEATURE_TABLE_CACHE = table
    return table


def reset_cache():
    global _FEATURE_TABLE_CACHE
    _FEATURE_TABLE_CACHE = None


def merge_into(df: pd.DataFrame, home_col: str = "TEAM_NAME",
               away_col: str = "TEAM_NAME.1", date_col: str = "Date") -> pd.DataFrame:
    """Merge advanced features into a games DataFrame for both home and away teams.

    Output adds columns prefixed with `H_` and `A_`.
    """
    table = build_feature_table()
    if table.empty:
        return df

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    feature_cols = [c for c in table.columns if c not in ("Team", "Date")]
    home_features = table.rename(
        columns={c: f"H_{c}" for c in feature_cols} | {"Team": home_col, "Date": date_col}
    )
    away_features = table.rename(
        columns={c: f"A_{c}" for c in feature_cols} | {"Team": away_col, "Date": date_col}
    )
    df = df.merge(home_features, on=[home_col, date_col], how="left")
    df = df.merge(away_features, on=[away_col, date_col], how="left")

    # Diff features (home minus away) — often more predictive than raw values.
    for c in feature_cols:
        h, a = f"H_{c}", f"A_{c}"
        if h in df.columns and a in df.columns:
            df[f"D_{c}"] = df[h] - df[a]

    return df
