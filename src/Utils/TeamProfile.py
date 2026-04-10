"""Team historical splits: by weekday, vs strong opponents, ATS record.

Used by Flask + main.py to attach context to game cards so users can see
patterns like "湖人週五打得很好" or "勇士對戰強隊頹勢".

Conventions in OddsData:
    Spread > 0  → home favored by Spread points
    Spread < 0  → home underdog by |Spread|
    Win_Margin = home_score - away_score
    home_cover_margin = Win_Margin - Spread
        > 0 → home covered the spread
        < 0 → away covered
        = 0 → push
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ODDS_DB = Path(__file__).resolve().parents[2] / "Data" / "OddsData.sqlite"

WEEKDAY_LABELS_ZH = ["週一", "週二", "週三", "週四", "週五", "週六", "週日"]
SEASON_KEYS = [
    "2007-08", "2008-09", "2009-10", "2010-11", "2011-12",
    "2012-13", "2013-14", "2014-15", "2015-16", "2016-17",
    "2017-18", "2018-19", "2019-20", "2020-21", "2021-22",
    "2022-23", "2023-24", "2024-25", "2025-26",
]


def _table_exists(con, name):
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _freshest_season_table(con, season_key):
    """Mirror Create_Games' freshness rule so we read the latest snapshot."""
    candidates = [
        f"odds_{season_key}_new",
        f"odds_{season_key}",
        f"{season_key}_new",
        season_key,
    ]
    existing = [t for t in candidates if _table_exists(con, t)]
    if not existing:
        return None
    if len(existing) == 1:
        return existing[0]

    def freshness(t):
        try:
            row = con.execute(f'SELECT MAX(Date) FROM "{t}"').fetchone()
            return row[0] or ""
        except Exception:
            return ""

    return max(existing, key=freshness)


_ALL_GAMES_CACHE: pd.DataFrame | None = None


def _load_all_games() -> pd.DataFrame:
    """Load every season's games once and cache as a single DataFrame."""
    global _ALL_GAMES_CACHE
    if _ALL_GAMES_CACHE is not None:
        return _ALL_GAMES_CACHE

    frames = []
    with sqlite3.connect(ODDS_DB) as con:
        for season in SEASON_KEYS:
            tbl = _freshest_season_table(con, season)
            if not tbl:
                continue
            try:
                df = pd.read_sql_query(
                    f'SELECT Date, Home, Away, OU, Spread, ML_Home, ML_Away, '
                    f'Points, Win_Margin FROM "{tbl}"',
                    con,
                )
                frames.append(df)
            except Exception:
                continue
    if not frames:
        _ALL_GAMES_CACHE = pd.DataFrame()
        return _ALL_GAMES_CACHE

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.dropna(subset=["Date", "Home", "Away"])
    all_df = all_df.drop_duplicates(subset=["Date", "Home", "Away"], keep="last")
    all_df["Date"] = pd.to_datetime(all_df["Date"], errors="coerce")
    all_df = all_df.dropna(subset=["Date"])
    # Coerce numeric columns — historical seasons sometimes contain strings.
    for col in ("OU", "Spread", "ML_Home", "ML_Away", "Points", "Win_Margin"):
        if col in all_df.columns:
            all_df[col] = pd.to_numeric(all_df[col], errors="coerce")
    # Only keep games with final scores so we can compute team-perspective stats.
    all_df = all_df[all_df["Points"].fillna(0) > 0].reset_index(drop=True)
    all_df["weekday"] = all_df["Date"].dt.dayofweek
    _ALL_GAMES_CACHE = all_df
    return _ALL_GAMES_CACHE


def reset_cache():
    """Clear the in-memory game cache; call after data refresh."""
    global _ALL_GAMES_CACHE
    _ALL_GAMES_CACHE = None
    team_profile_for_date.cache_clear()


def _implied_prob(american_odds):
    if pd.isna(american_odds):
        return np.nan
    if american_odds >= 100:
        return 100.0 / (american_odds + 100.0)
    return abs(american_odds) / (abs(american_odds) + 100.0)


def _team_perspective(games: pd.DataFrame, team: str) -> pd.DataFrame:
    """Filter games involving `team` and add team-perspective columns."""
    is_home = games["Home"] == team
    is_away = games["Away"] == team
    sub = games[is_home | is_away].copy()
    if sub.empty:
        return sub

    sub["is_home"] = sub["Home"] == team
    home_score = (sub["Points"] + sub["Win_Margin"]) / 2.0
    away_score = (sub["Points"] - sub["Win_Margin"]) / 2.0
    sub["team_score"] = np.where(sub["is_home"], home_score, away_score)
    sub["opp_score"] = np.where(sub["is_home"], away_score, home_score)
    sub["won"] = sub["team_score"] > sub["opp_score"]

    # ATS: home_cover_margin > 0 means home covered. Translate to team perspective.
    home_cover_margin = sub["Win_Margin"] - sub["Spread"]
    sub["team_cover_margin"] = np.where(sub["is_home"], home_cover_margin, -home_cover_margin)
    sub["team_covered"] = sub["team_cover_margin"] > 0
    sub["ats_push"] = sub["team_cover_margin"] == 0

    # Opponent implied probability from their money line; > 0.55 = "strong".
    opp_ml = np.where(sub["is_home"], sub["ML_Away"], sub["ML_Home"])
    sub["opp_implied_prob"] = pd.Series(opp_ml).apply(_implied_prob).values
    sub["vs_strong"] = sub["opp_implied_prob"] > 0.55
    return sub


def _aggregate(sub: pd.DataFrame) -> dict:
    if sub.empty:
        return {
            "games": 0, "wins": 0, "losses": 0, "win_rate": None,
            "avg_for": None, "avg_against": None, "avg_total": None,
            "ats_wins": 0, "ats_losses": 0, "ats_pushes": 0, "ats_rate": None,
        }
    games = len(sub)
    wins = int(sub["won"].sum())
    ats_pushes = int(sub["ats_push"].sum())
    ats_wins = int(sub["team_covered"].sum())
    ats_losses = games - ats_wins - ats_pushes
    ats_decided = ats_wins + ats_losses
    return {
        "games": games,
        "wins": wins,
        "losses": games - wins,
        "win_rate": round(wins / games * 100, 1),
        "avg_for": round(float(sub["team_score"].mean()), 1),
        "avg_against": round(float(sub["opp_score"].mean()), 1),
        "avg_total": round(float(sub["team_score"].mean() + sub["opp_score"].mean()), 1),
        "ats_wins": ats_wins,
        "ats_losses": ats_losses,
        "ats_pushes": ats_pushes,
        "ats_rate": round(ats_wins / ats_decided * 100, 1) if ats_decided else None,
    }


def _season_start(target: pd.Timestamp) -> pd.Timestamp:
    """Return Oct 1 of the NBA season that contains `target`."""
    if target.month >= 10:
        return pd.Timestamp(year=target.year, month=10, day=1)
    return pd.Timestamp(year=target.year - 1, month=10, day=1)


def _window_splits(sub: pd.DataFrame, weekday: int) -> dict:
    """Compute overall / weekday / vs_strong / vs_weak aggregates for a window."""
    return {
        "overall": _aggregate(sub),
        "weekday": _aggregate(sub[sub["weekday"] == weekday]),
        "vs_strong": _aggregate(sub[sub["vs_strong"]]),
        "vs_weak": _aggregate(sub[~sub["vs_strong"]]),
    }


@lru_cache(maxsize=2048)
def team_profile_for_date(team: str, target_iso: str) -> dict:
    """Return time-windowed historical splits for `team` (strictly before
    `target_iso`).

    Each time window contains the same four splits:
        overall / weekday / vs_strong / vs_weak

    Windows: season / last_3m / last_2m / last_1m.
    """
    target = pd.Timestamp(target_iso)
    weekday = target.dayofweek
    weekday_name = WEEKDAY_LABELS_ZH[weekday]

    all_games = _load_all_games()
    if all_games.empty:
        empty_window = _window_splits(all_games.iloc[0:0], weekday)
        return {
            "weekday_name": weekday_name,
            "season": empty_window, "last_3m": empty_window,
            "last_2m": empty_window, "last_1m": empty_window,
        }

    history = all_games[all_games["Date"] < target]
    sub = _team_perspective(history, team)

    season_start = _season_start(target)
    cut_3m = target - pd.Timedelta(days=90)
    cut_2m = target - pd.Timedelta(days=60)
    cut_1m = target - pd.Timedelta(days=30)

    return {
        "weekday_name": weekday_name,
        "season":  _window_splits(sub[sub["Date"] >= season_start], weekday),
        "last_3m": _window_splits(sub[sub["Date"] >= cut_3m], weekday),
        "last_2m": _window_splits(sub[sub["Date"] >= cut_2m], weekday),
        "last_1m": _window_splits(sub[sub["Date"] >= cut_1m], weekday),
    }


def grade_spread(spread, win_margin):
    """Return ('home'|'away'|'push'|None, cover_margin) for a finished game.

    spread: positive = home favored by N (home -N).
    win_margin: home_score - away_score (positive = home won by N).
    """
    if spread is None or win_margin is None:
        return None, None
    if pd.isna(spread) or pd.isna(win_margin):
        return None, None
    diff = float(win_margin) - float(spread)
    if diff > 0:
        return "home", diff
    if diff < 0:
        return "away", diff
    return "push", 0.0
