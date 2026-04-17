"""ATS (against-the-spread) 回測：找出提升讓分/受讓勝率的篩選條件。

從 OddsData 載入所有歷史比賽，計算進階特徵（近期戰績、賽程密度、
連勝連敗等），然後按多種維度切片分析 ATS 勝率，找出有利可圖的情境。

分析維度：
  1. 讓分大小 (spread size)
  2. 主/客場
  3. 近期戰績 (L5/L10 勝率)
  4. 近期 ATS 表現 (L10 cover%)
  5. 連勝/連敗/ATS 連勝
  6. 背靠背 (back-to-back)
  7. 賽程密度 (3-in-4, 4-in-7)
  8. 客場連續作戰
  9. 組合篩選 (多條件交叉)

也支援使用 ATS 模型預測進行回測，按模型信心度篩選。

Usage:
  PYTHONPATH=. python scripts/backtest_ats.py
  PYTHONPATH=. python scripts/backtest_ats.py --seasons 2024-25,2025-26
  PYTHONPATH=. python scripts/backtest_ats.py --use-model
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
# Use main repo Data directory (worktrees may not have full data).
_MAIN_REPO = Path(__file__).resolve().parents[1]
_DATA_CANDIDATES = [
    _MAIN_REPO / "Data" / "OddsData.sqlite",
    _MAIN_REPO.parent.parent.parent / "Data" / "OddsData.sqlite",  # worktree fallback
]
ODDS_DB = next((p for p in _DATA_CANDIDATES if p.exists() and p.stat().st_size > 1000), _DATA_CANDIDATES[0])

ALL_SEASONS = [
    "2007-08", "2008-09", "2009-10", "2010-11", "2011-12",
    "2012-13", "2013-14", "2014-15", "2015-16", "2016-17",
    "2017-18", "2018-19", "2019-20", "2020-21", "2021-22",
    "2022-23", "2023-24", "2024-25", "2025-26",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _table_exists(con, name):
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _freshest_odds_table(con, season_key):
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
    return max(
        existing,
        key=lambda t: con.execute(f'SELECT MAX(Date) FROM "{t}"').fetchone()[0] or "",
    )


def load_all_games(seasons: list[str] | None = None) -> pd.DataFrame:
    """Load completed games with spread outcomes from OddsData."""
    if seasons is None:
        seasons = ALL_SEASONS
    frames = []
    with sqlite3.connect(ODDS_DB) as con:
        for season in seasons:
            tbl = _freshest_odds_table(con, season)
            if not tbl:
                continue
            try:
                df = pd.read_sql_query(
                    f'SELECT Date, Home, Away, Spread, Win_Margin, Points, '
                    f'ML_Home, ML_Away FROM "{tbl}"',
                    con,
                )
                df["Season"] = season
                frames.append(df)
            except Exception as exc:
                print(f"warn: {season}: {exc}")
    if not frames:
        raise RuntimeError("No odds data found.")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["Date", "Home", "Away"], keep="last")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ("Spread", "Win_Margin", "Points", "ML_Home", "ML_Away"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Only completed games
    df = df.dropna(subset=["Date", "Spread", "Win_Margin"])
    df = df[df["Points"].fillna(0) > 0].reset_index(drop=True)
    df = df.sort_values("Date").reset_index(drop=True)

    # Core ATS columns
    df["Home_Cover_Margin"] = df["Win_Margin"] - df["Spread"]
    df["Home_Covered"] = (df["Home_Cover_Margin"] > 0).astype(int)
    df["Away_Covered"] = (df["Home_Cover_Margin"] < 0).astype(int)
    df["Push"] = (df["Home_Cover_Margin"] == 0).astype(int)

    # Spread direction
    df["Home_Fav"] = (df["Spread"] > 0).astype(int)   # home is favored (gives points)
    df["Away_Fav"] = (df["Spread"] < 0).astype(int)   # away is favored
    df["Abs_Spread"] = df["Spread"].abs()

    return df


# ---------------------------------------------------------------------------
# Advanced features (per-team rolling)
# ---------------------------------------------------------------------------

def add_team_features(games: pd.DataFrame) -> pd.DataFrame:
    """Add rolling form, schedule density, and streak features."""
    games = games.copy()

    # Build per-team game log (long format)
    home = games[["Date", "Home", "Away", "Spread", "Win_Margin", "Points"]].copy()
    home.columns = ["Date", "Team", "Opp", "Spread", "Win_Margin", "Points"]
    home["IsHome"] = 1
    home["Won"] = (home["Win_Margin"] > 0).astype(int)
    home["CoverMargin"] = home["Win_Margin"] - home["Spread"]
    home["Covered"] = (home["CoverMargin"] > 0).astype(int)

    away = games[["Date", "Home", "Away", "Spread", "Win_Margin", "Points"]].copy()
    away.columns = ["Date", "Opp", "Team", "Spread", "Win_Margin", "Points"]
    away["IsHome"] = 0
    away["Won"] = (away["Win_Margin"] < 0).astype(int)
    away["CoverMargin"] = -(away["Win_Margin"] - away["Spread"])
    away["Covered"] = (away["CoverMargin"] > 0).astype(int)

    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["Team", "Date"]).reset_index(drop=True)
    grp = long.groupby("Team", sort=False)

    # Rolling form (shifted so current game not included)
    won_prev = grp["Won"].shift(1)
    cov_prev = grp["Covered"].shift(1)

    long["form_w5"] = won_prev.groupby(long["Team"]).transform(
        lambda s: s.rolling(5, min_periods=3).mean()
    )
    long["form_w10"] = won_prev.groupby(long["Team"]).transform(
        lambda s: s.rolling(10, min_periods=5).mean()
    )
    long["form_ats10"] = cov_prev.groupby(long["Team"]).transform(
        lambda s: s.rolling(10, min_periods=5).mean()
    )

    # Streaks
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
    long["streak_ats"] = grp["Covered"].shift(1).groupby(long["Team"], sort=False).transform(
        lambda s: _streak(s, 1.0)
    )

    # Schedule density
    long["prev_date"] = grp["Date"].shift(1)
    long["days_rest"] = (long["Date"] - long["prev_date"]).dt.days
    long["b2b"] = (long["days_rest"] == 1).astype(int)

    def _count_in_window(group, window_days):
        dates = group["Date"].values
        out = np.zeros(len(dates), dtype=int)
        for i in range(len(dates)):
            cutoff = dates[i] - np.timedelta64(window_days, "D")
            out[i] = int(((dates[:i] >= cutoff) & (dates[:i] < dates[i])).sum())
        return pd.Series(out, index=group.index)

    long["games_4d"] = long.groupby("Team", group_keys=False).apply(
        lambda g: _count_in_window(g, 4)
    )
    long["games_7d"] = long.groupby("Team", group_keys=False).apply(
        lambda g: _count_in_window(g, 7)
    )

    # Road trip length
    def _road_trip(group):
        is_away = (group["IsHome"] == 0).astype(int).values
        out = np.zeros(len(is_away), dtype=int)
        c = 0
        for i, v in enumerate(is_away):
            out[i] = c
            c = c + 1 if v else 0
        return pd.Series(out, index=group.index)

    long["road_trip"] = long.groupby("Team", group_keys=False).apply(_road_trip)

    # Split back into home / away features and merge onto games
    feature_cols = [
        "form_w5", "form_w10", "form_ats10",
        "streak_w", "streak_l", "streak_ats",
        "days_rest", "b2b", "games_4d", "games_7d", "road_trip",
    ]

    home_features = long[long["IsHome"] == 1][["Team", "Date"] + feature_cols].rename(
        columns={c: f"H_{c}" for c in feature_cols} | {"Team": "Home"}
    )
    away_features = long[long["IsHome"] == 0][["Team", "Date"] + feature_cols].rename(
        columns={c: f"A_{c}" for c in feature_cols} | {"Team": "Away"}
    )

    games = games.merge(home_features, on=["Home", "Date"], how="left")
    games = games.merge(away_features, on=["Away", "Date"], how="left")

    return games


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def ats_stats(mask: pd.Series, games: pd.DataFrame, side: str = "home") -> dict:
    """Compute ATS stats for a subset of games.

    side: 'home' = bet home covers, 'away' = bet away covers,
          'fav' = bet favorite covers, 'dog' = bet underdog covers.
    """
    subset = games[mask].copy()
    n = len(subset)
    if n == 0:
        return {"n": 0, "wins": 0, "pushes": 0, "wr": 0.0, "wr_no_push": 0.0}

    if side == "home":
        wins = int(subset["Home_Covered"].sum())
    elif side == "away":
        wins = int(subset["Away_Covered"].sum())
    elif side == "fav":
        # Favorite covers = home covers when home is fav, away covers when away is fav
        wins = int(
            (subset["Home_Fav"] * subset["Home_Covered"]).sum()
            + (subset["Away_Fav"] * subset["Away_Covered"]).sum()
        )
    elif side == "dog":
        wins = int(
            (subset["Home_Fav"] * subset["Away_Covered"]).sum()
            + (subset["Away_Fav"] * subset["Home_Covered"]).sum()
        )
    else:
        raise ValueError(f"Unknown side: {side}")

    pushes = int(subset["Push"].sum())
    effective = n - pushes
    wr = wins / n if n else 0.0
    wr_no_push = wins / effective if effective else 0.0
    return {"n": n, "wins": wins, "pushes": pushes, "wr": wr, "wr_no_push": wr_no_push}


def print_section(title: str):
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")


def print_row(label: str, stats: dict, highlight_threshold: float = 0.53):
    """Print a stats row. Highlight if win rate exceeds threshold."""
    if stats["n"] < 30:
        return
    marker = " ★" if stats["wr_no_push"] >= highlight_threshold else ""
    print(
        f"  {label:<45s}  {stats['n']:>5d}  "
        f"{stats['wins']:>5d}  {stats['pushes']:>3d}  "
        f"{stats['wr']*100:>5.1f}%  {stats['wr_no_push']*100:>5.1f}%{marker}"
    )


def print_header():
    print(
        f"  {'Filter':<45s}  {'Games':>5s}  "
        f"{'Wins':>5s}  {'P':>3s}  "
        f"{'WR%':>6s}  {'WR-NP%':>7s}"
    )
    print(f"  {'-'*45}  {'-'*5}  {'-'*5}  {'-'*3}  {'-'*6}  {'-'*7}")


# ---------------------------------------------------------------------------
# Dimension analyses
# ---------------------------------------------------------------------------

def analyze_baseline(games: pd.DataFrame):
    print_section("基準線 (Baseline)")
    print_header()
    all_mask = pd.Series(True, index=games.index)
    print_row("全部比賽 → 押主場 cover", ats_stats(all_mask, games, "home"))
    print_row("全部比賽 → 押客場 cover", ats_stats(all_mask, games, "away"))
    print_row("全部比賽 → 押熱門 cover", ats_stats(all_mask, games, "fav"))
    print_row("全部比賽 → 押冷門 cover", ats_stats(all_mask, games, "dog"))


def analyze_spread_size(games: pd.DataFrame):
    print_section("維度 1: 讓分大小 (Spread Size)")
    bins = [(0, 2.5), (2.5, 5), (5, 7.5), (7.5, 10), (10, 15), (15, 30)]
    print_header()
    for lo, hi in bins:
        mask = (games["Abs_Spread"] >= lo) & (games["Abs_Spread"] < hi)
        label_range = f"讓分 {lo}-{hi}"
        print_row(f"{label_range} → 押熱門", ats_stats(mask, games, "fav"))
        print_row(f"{label_range} → 押冷門", ats_stats(mask, games, "dog"))


def analyze_home_away_fav(games: pd.DataFrame):
    print_section("維度 2: 主客場 × 讓受讓")
    print_header()
    # Home favorite
    hf = games["Home_Fav"] == 1
    print_row("主場讓分 → 押主場 cover", ats_stats(hf, games, "home"))
    print_row("主場讓分 → 押客場 cover", ats_stats(hf, games, "away"))
    # Away favorite
    af = games["Away_Fav"] == 1
    print_row("客場讓分 → 押主場 cover (受讓)", ats_stats(af, games, "home"))
    print_row("客場讓分 → 押客場 cover (讓分)", ats_stats(af, games, "away"))


def analyze_form(games: pd.DataFrame):
    print_section("維度 3: 近期戰績 (Recent Form)")
    print_header()
    # Home team form
    for threshold in [0.7, 0.6, 0.4, 0.3]:
        mask_hot = games["H_form_w10"].fillna(0.5) >= threshold
        mask_cold = games["H_form_w10"].fillna(0.5) <= (1 - threshold)
        if threshold >= 0.5:
            print_row(f"主場 L10 勝率 ≥{threshold:.0%} → 押主場", ats_stats(mask_hot, games, "home"))
            print_row(f"主場 L10 勝率 ≥{threshold:.0%} → 押客場", ats_stats(mask_hot, games, "away"))
        else:
            print_row(f"主場 L10 勝率 ≤{threshold:.0%} → 押主場", ats_stats(mask_cold, games, "home"))
            print_row(f"主場 L10 勝率 ≤{threshold:.0%} → 押客場", ats_stats(mask_cold, games, "away"))

    # Away team form
    for threshold in [0.7, 0.3]:
        mask_hot = games["A_form_w10"].fillna(0.5) >= threshold
        mask_cold = games["A_form_w10"].fillna(0.5) <= threshold
        if threshold >= 0.5:
            print_row(f"客場 L10 勝率 ≥{threshold:.0%} → 押客場", ats_stats(mask_hot, games, "away"))
            print_row(f"客場 L10 勝率 ≥{threshold:.0%} → 押主場", ats_stats(mask_hot, games, "home"))
        else:
            print_row(f"客場 L10 勝率 ≤{threshold:.0%} → 押客場", ats_stats(mask_cold, games, "away"))
            print_row(f"客場 L10 勝率 ≤{threshold:.0%} → 押主場", ats_stats(mask_cold, games, "home"))


def analyze_ats_form(games: pd.DataFrame):
    print_section("維度 4: 近期 ATS 表現 (L10 Cover %)")
    print_header()
    for threshold in [0.7, 0.6, 0.4, 0.3]:
        h_hot = games["H_form_ats10"].fillna(0.5) >= threshold
        h_cold = games["H_form_ats10"].fillna(0.5) <= threshold
        a_hot = games["A_form_ats10"].fillna(0.5) >= threshold
        a_cold = games["A_form_ats10"].fillna(0.5) <= threshold

        if threshold >= 0.5:
            print_row(f"主場 ATS L10 ≥{threshold:.0%} → 押主場", ats_stats(h_hot, games, "home"))
            print_row(f"主場 ATS L10 ≥{threshold:.0%} → 押客場", ats_stats(h_hot, games, "away"))
            print_row(f"客場 ATS L10 ≥{threshold:.0%} → 押客場", ats_stats(a_hot, games, "away"))
            print_row(f"客場 ATS L10 ≥{threshold:.0%} → 押主場", ats_stats(a_hot, games, "home"))
        else:
            print_row(f"主場 ATS L10 ≤{threshold:.0%} → 押主場 (反向)", ats_stats(h_cold, games, "home"))
            print_row(f"主場 ATS L10 ≤{threshold:.0%} → 押客場", ats_stats(h_cold, games, "away"))
            print_row(f"客場 ATS L10 ≤{threshold:.0%} → 押客場 (反向)", ats_stats(a_cold, games, "away"))
            print_row(f"客場 ATS L10 ≤{threshold:.0%} → 押主場", ats_stats(a_cold, games, "home"))


def analyze_streaks(games: pd.DataFrame):
    print_section("維度 5: 連勝/連敗/ATS 連勝")
    print_header()
    for streak_len in [3, 4, 5]:
        # Home team on win streak
        mask = games["H_streak_w"].fillna(0) >= streak_len
        print_row(f"主場連勝 ≥{streak_len} → 押主場", ats_stats(mask, games, "home"))
        print_row(f"主場連勝 ≥{streak_len} → 押客場", ats_stats(mask, games, "away"))

        # Home team on losing streak
        mask = games["H_streak_l"].fillna(0) >= streak_len
        print_row(f"主場連敗 ≥{streak_len} → 押主場 (反彈)", ats_stats(mask, games, "home"))
        print_row(f"主場連敗 ≥{streak_len} → 押客場", ats_stats(mask, games, "away"))

        # Away team on win streak
        mask = games["A_streak_w"].fillna(0) >= streak_len
        print_row(f"客場連勝 ≥{streak_len} → 押客場", ats_stats(mask, games, "away"))

        # Away team on losing streak
        mask = games["A_streak_l"].fillna(0) >= streak_len
        print_row(f"客場連敗 ≥{streak_len} → 押主場", ats_stats(mask, games, "home"))

    # ATS streaks
    for streak_len in [3, 4, 5]:
        mask = games["H_streak_ats"].fillna(0) >= streak_len
        print_row(f"主場 ATS 連勝 ≥{streak_len} → 押主場", ats_stats(mask, games, "home"))
        print_row(f"主場 ATS 連勝 ≥{streak_len} → 押客場 (反向)", ats_stats(mask, games, "away"))

        mask = games["A_streak_ats"].fillna(0) >= streak_len
        print_row(f"客場 ATS 連勝 ≥{streak_len} → 押客場", ats_stats(mask, games, "away"))
        print_row(f"客場 ATS 連勝 ≥{streak_len} → 押主場 (反向)", ats_stats(mask, games, "home"))


def analyze_schedule(games: pd.DataFrame):
    print_section("維度 6: 賽程密度 (B2B, 3-in-4, 客場連續)")
    print_header()

    # Home team B2B
    h_b2b = games["H_b2b"].fillna(0) == 1
    print_row("主場背靠背 → 押主場", ats_stats(h_b2b, games, "home"))
    print_row("主場背靠背 → 押客場", ats_stats(h_b2b, games, "away"))

    # Away team B2B
    a_b2b = games["A_b2b"].fillna(0) == 1
    print_row("客場背靠背 → 押客場", ats_stats(a_b2b, games, "away"))
    print_row("客場背靠背 → 押主場", ats_stats(a_b2b, games, "home"))

    # Both B2B
    both_b2b = h_b2b & a_b2b
    print_row("雙方都背靠背 → 押主場", ats_stats(both_b2b, games, "home"))

    # Only one side B2B
    h_only_b2b = h_b2b & ~a_b2b
    a_only_b2b = a_b2b & ~h_b2b
    print_row("僅主場背靠背 → 押客場", ats_stats(h_only_b2b, games, "away"))
    print_row("僅客場背靠背 → 押主場", ats_stats(a_only_b2b, games, "home"))

    # 3-in-4
    h_dense = games["H_games_4d"].fillna(0) >= 2
    a_dense = games["A_games_4d"].fillna(0) >= 2
    print_row("主場 4天3場 → 押客場", ats_stats(h_dense, games, "away"))
    print_row("客場 4天3場 → 押主場", ats_stats(a_dense, games, "home"))

    # Road trip
    for trip_len in [3, 4, 5]:
        mask = games["A_road_trip"].fillna(0) >= trip_len
        print_row(f"客場連續 ≥{trip_len} 場客場 → 押主場", ats_stats(mask, games, "home"))
        print_row(f"客場連續 ≥{trip_len} 場客場 → 押客場", ats_stats(mask, games, "away"))


def analyze_rest_advantage(games: pd.DataFrame):
    print_section("維度 7: 休息天數差異")
    print_header()

    h_rest = games["H_days_rest"].fillna(2)
    a_rest = games["A_days_rest"].fillna(2)
    rest_diff = h_rest - a_rest  # positive = home more rested

    for lo, hi, label in [
        (2, 99, "主場多休 ≥2天"),
        (1, 1, "主場多休 1天"),
        (0, 0, "雙方同休"),
        (-1, -1, "客場多休 1天"),
        (-99, -2, "客場多休 ≥2天"),
    ]:
        mask = (rest_diff >= lo) & (rest_diff <= hi)
        print_row(f"{label} → 押主場", ats_stats(mask, games, "home"))
        print_row(f"{label} → 押客場", ats_stats(mask, games, "away"))


def analyze_combined(games: pd.DataFrame):
    """Test multi-condition filters to find high-value spots."""
    print_section("維度 8: 組合篩選 (高勝率策略搜尋)")
    print_header()

    results = []

    # Sweep combinations
    for side_label, side in [("押冷門", "dog"), ("押熱門", "fav")]:
        for spread_lo, spread_hi in [(0, 5), (5, 10), (10, 30), (0, 30)]:
            spread_mask = (games["Abs_Spread"] >= spread_lo) & (games["Abs_Spread"] < spread_hi)
            spread_str = f"讓分{spread_lo}-{spread_hi}"

            # Base spread + side
            s = ats_stats(spread_mask, games, side)
            if s["n"] >= 30:
                results.append((f"{spread_str} {side_label}", s))

            # + Home form
            for form_op, form_val, form_label in [
                (">=", 0.7, "主場強隊"),
                ("<=", 0.3, "主場弱隊"),
            ]:
                if form_op == ">=":
                    form_mask = spread_mask & (games["H_form_w10"].fillna(0.5) >= form_val)
                else:
                    form_mask = spread_mask & (games["H_form_w10"].fillna(0.5) <= form_val)
                s = ats_stats(form_mask, games, side)
                if s["n"] >= 30:
                    results.append((f"{spread_str} + {form_label} {side_label}", s))

            # + B2B disadvantage
            h_b2b_only = spread_mask & (games["H_b2b"].fillna(0) == 1) & (games["A_b2b"].fillna(0) == 0)
            s = ats_stats(h_b2b_only, games, "away")
            if s["n"] >= 30:
                results.append((f"{spread_str} + 僅主場B2B → 押客場", s))

            a_b2b_only = spread_mask & (games["A_b2b"].fillna(0) == 1) & (games["H_b2b"].fillna(0) == 0)
            s = ats_stats(a_b2b_only, games, "home")
            if s["n"] >= 30:
                results.append((f"{spread_str} + 僅客場B2B → 押主場", s))

            # + ATS form regression
            for ats_thresh, ats_label in [(0.3, "ATS冷"), (0.7, "ATS熱")]:
                h_ats_mask = spread_mask & (games["H_form_ats10"].fillna(0.5) <= ats_thresh)
                s = ats_stats(h_ats_mask, games, "home")
                if s["n"] >= 30 and ats_thresh <= 0.4:
                    results.append((f"{spread_str} + 主場{ats_label} → 押主場(反彈)", s))

                a_ats_mask = spread_mask & (games["A_form_ats10"].fillna(0.5) <= ats_thresh)
                s = ats_stats(a_ats_mask, games, "away")
                if s["n"] >= 30 and ats_thresh <= 0.4:
                    results.append((f"{spread_str} + 客場{ats_label} → 押客場(反彈)", s))

            # + Losing streak (bounce-back)
            for streak_len in [3, 4]:
                h_lose = spread_mask & (games["H_streak_l"].fillna(0) >= streak_len)
                s = ats_stats(h_lose, games, "home")
                if s["n"] >= 30:
                    results.append((f"{spread_str} + 主場連敗≥{streak_len} → 押主場(反彈)", s))

                a_lose = spread_mask & (games["A_streak_l"].fillna(0) >= streak_len)
                s = ats_stats(a_lose, games, "home")
                if s["n"] >= 30:
                    results.append((f"{spread_str} + 客場連敗≥{streak_len} → 押主場", s))

    # Also test: strong form + dog
    for form_thresh in [0.7, 0.6]:
        # Home team strong but is underdog (receiving points)
        mask = (games["Away_Fav"] == 1) & (games["H_form_w10"].fillna(0.5) >= form_thresh)
        s = ats_stats(mask, games, "home")
        if s["n"] >= 30:
            results.append((f"主場受讓 + L10≥{form_thresh:.0%} → 押主場", s))

        # Away team strong and is underdog
        mask = (games["Home_Fav"] == 1) & (games["A_form_w10"].fillna(0.5) >= form_thresh)
        s = ats_stats(mask, games, "away")
        if s["n"] >= 30:
            results.append((f"客場受讓 + L10≥{form_thresh:.0%} → 押客場", s))

    # Road trip fatigue
    for trip_len in [3, 4]:
        mask = (games["A_road_trip"].fillna(0) >= trip_len) & (games["Abs_Spread"] <= 7)
        s = ats_stats(mask, games, "home")
        if s["n"] >= 30:
            results.append((f"小讓分 + 客場連客≥{trip_len} → 押主場", s))

    # Sort by win rate (no push) descending
    results.sort(key=lambda x: -x[1]["wr_no_push"])

    print("\n  Top 30 組合篩選 (依 WR-NP% 排序, 至少 30 場):\n")
    print_header()
    for label, stats in results[:30]:
        print_row(label, stats)


# ---------------------------------------------------------------------------
# Model-based backtest
# ---------------------------------------------------------------------------

def analyze_with_model(games: pd.DataFrame):
    """Use the trained ATS model for predictions and sweep confidence thresholds."""
    print_section("模型回測: ATS XGBoost 模型信心度篩選")

    try:
        import xgboost as xgb
        import joblib
    except ImportError:
        print("  xgboost not installed, skipping model analysis.")
        return

    model_dir = BASE_DIR / "Models" / "XGBoost_Models"
    ats_models = list(model_dir.glob("*ATS*.json"))
    if not ats_models:
        print("  No ATS model found, skipping.")
        return

    import re
    pat = re.compile(r"XGBoost_(\d+(?:\.\d+)?)%_")
    best_model_path = max(
        ats_models,
        key=lambda p: (float(pat.search(p.name).group(1)) if pat.search(p.name) else 0, p.stat().st_mtime),
    )
    print(f"  Using model: {best_model_path.name}")

    booster = xgb.Booster()
    booster.load_model(str(best_model_path))

    # Load the dataset the same way ATS trainer does
    dataset_db = BASE_DIR / "Data" / "dataset.sqlite"
    with sqlite3.connect(dataset_db) as con:
        ds = pd.read_sql_query('SELECT * FROM "dataset_2012-26"', con)

    # Merge odds
    odds_frames = []
    with sqlite3.connect(ODDS_DB) as con:
        for season in ALL_SEASONS:
            tbl = _freshest_odds_table(con, season)
            if not tbl:
                continue
            try:
                o = pd.read_sql_query(
                    f'SELECT Date, Home, Away, Spread, Win_Margin, Points, ML_Home, ML_Away FROM "{tbl}"',
                    con,
                )
                odds_frames.append(o)
            except Exception:
                continue
    if not odds_frames:
        print("  No odds data for model merge.")
        return

    odds = pd.concat(odds_frames, ignore_index=True)
    odds = odds.drop_duplicates(subset=["Date", "Home", "Away"], keep="last")
    for col in ("Spread", "Win_Margin", "Points", "ML_Home", "ML_Away"):
        odds[col] = pd.to_numeric(odds[col], errors="coerce")

    ds = ds.merge(
        odds, how="left",
        left_on=["Date", "TEAM_NAME", "TEAM_NAME.1"],
        right_on=["Date", "Home", "Away"],
    )
    ds = ds.dropna(subset=["Spread", "Win_Margin"]).reset_index(drop=True)
    ds = ds[ds["Points"].fillna(0) > 0].reset_index(drop=True)
    ds["ATS_Cover"] = ((ds["Win_Margin"] - ds["Spread"]) > 0).astype(int)

    # Merge advanced features
    try:
        sys.path.insert(0, str(BASE_DIR))
        from src.Utils.AdvancedFeatures import merge_into
        ds = merge_into(ds, home_col="TEAM_NAME", away_col="TEAM_NAME.1", date_col="Date")
        ds = ds.dropna(subset=["H_form_w_pct_10", "A_form_w_pct_10"]).reset_index(drop=True)
    except Exception as exc:
        print(f"  warn: advanced features failed: {exc}")

    # Use last 10% as test set (same as trainer)
    n = len(ds)
    test_start = int(n * 0.9)
    test_df = ds.iloc[test_start:].reset_index(drop=True)
    print(f"  Test set: {len(test_df)} games")

    # Prepare features (same drops as ATS trainer — must stay in sync with
    # src/Train-Models/XGBoost_Model_ATS.py DROP_COLUMNS)
    drop_cols = [
        "index", "Score", "Home-Team-Win", "TEAM_NAME", "Date",
        "index.1", "TEAM_NAME.1", "Date.1", "OU-Cover", "OU",
        "Win_Margin", "ATS_Cover",
        "Home", "Away", "Points", "ML_Home", "ML_Away",
        # Temporal features excluded from 175-feature production model.
        "H_game_num_season", "A_game_num_season", "D_game_num_season",
        "H_month_sin", "A_month_sin", "D_month_sin",
        "H_month_cos", "A_month_cos", "D_month_cos",
        # Home/away split ATS features excluded from production model.
        "H_form_ats_pct_home_10", "A_form_ats_pct_home_10", "D_form_ats_pct_home_10",
        "H_form_ats_pct_away_10", "A_form_ats_pct_away_10", "D_form_ats_pct_away_10",
        # Offense/defense split + 10-game diff: excluded from 175-feature production model.
        "H_form_pts_for_5", "A_form_pts_for_5", "D_form_pts_for_5",
        "H_form_pts_against_5", "A_form_pts_against_5", "D_form_pts_against_5",
        "H_form_pts_diff_10", "A_form_pts_diff_10", "D_form_pts_diff_10",
    ]
    X_test = test_df.drop(columns=drop_cols, errors="ignore").astype(float).to_numpy()
    y_test = test_df["ATS_Cover"].astype(int).to_numpy()
    spreads = test_df["Spread"].to_numpy()
    margins = test_df["Win_Margin"].to_numpy()

    # Predict
    dtest = xgb.DMatrix(X_test)
    probs = booster.predict(dtest)
    if probs.ndim == 1:
        # Binary: prob of class 1
        prob_home_cover = probs
    else:
        prob_home_cover = probs[:, 1]

    raw_pred = (prob_home_cover >= 0.5).astype(int)
    raw_acc = (raw_pred == y_test).mean()
    print(f"  Raw accuracy: {raw_acc:.4f}")

    # Sweep confidence thresholds
    print(f"\n  信心度篩選 (押主場 cover):")
    print(f"  {'Threshold':>10s}  {'Games':>6s}  {'Wins':>5s}  {'WR%':>6s}  {'ROI@-110':>9s}")
    print(f"  {'-'*10}  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*9}")

    for thresh in [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.65, 0.70]:
        mask = prob_home_cover >= thresh
        n_bets = mask.sum()
        if n_bets < 20:
            continue
        wins = y_test[mask].sum()
        wr = wins / n_bets
        # ROI at -110 juice
        roi = (wins * 100 / 110 - (n_bets - wins)) / n_bets * 100
        print(f"  {thresh:>10.2f}  {n_bets:>6d}  {wins:>5d}  {wr*100:>5.1f}%  {roi:>+8.1f}%")

    print(f"\n  信心度篩選 (押客場 cover):")
    print(f"  {'Threshold':>10s}  {'Games':>6s}  {'Wins':>5s}  {'WR%':>6s}  {'ROI@-110':>9s}")
    print(f"  {'-'*10}  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*9}")

    for thresh in [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.65, 0.70]:
        mask = prob_home_cover <= (1 - thresh)  # away cover confidence
        n_bets = mask.sum()
        if n_bets < 20:
            continue
        wins = (1 - y_test[mask]).sum()  # away covered
        wr = wins / n_bets
        roi = (wins * 100 / 110 - (n_bets - wins)) / n_bets * 100
        print(f"  {thresh:>10.2f}  {n_bets:>6d}  {wins:>5.0f}  {wr*100:>5.1f}%  {roi:>+8.1f}%")


# ---------------------------------------------------------------------------
# Season breakdown
# ---------------------------------------------------------------------------

def analyze_by_season(games: pd.DataFrame):
    print_section("各賽季 ATS 基準線")
    print(f"  {'Season':<12s}  {'Games':>6s}  {'Home%':>6s}  {'Away%':>6s}  {'Fav%':>6s}  {'Dog%':>6s}")
    print(f"  {'-'*12}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
    for season in games["Season"].unique():
        mask = games["Season"] == season
        subset = games[mask]
        n = len(subset)
        non_push = n - int(subset["Push"].sum())
        if non_push == 0:
            continue
        h_wr = subset["Home_Covered"].sum() / non_push * 100
        a_wr = subset["Away_Covered"].sum() / non_push * 100
        fav_wins = (subset["Home_Fav"] * subset["Home_Covered"]).sum() + (subset["Away_Fav"] * subset["Away_Covered"]).sum()
        dog_wins = (subset["Home_Fav"] * subset["Away_Covered"]).sum() + (subset["Away_Fav"] * subset["Home_Covered"]).sum()
        fav_wr = fav_wins / non_push * 100
        dog_wr = dog_wins / non_push * 100
        print(f"  {season:<12s}  {n:>6d}  {h_wr:>5.1f}%  {a_wr:>5.1f}%  {fav_wr:>5.1f}%  {dog_wr:>5.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ATS 回測：找出提升讓分勝率的篩選條件")
    parser.add_argument("--seasons", default=None, help="Comma-separated seasons (default: all)")
    parser.add_argument("--eval-seasons", default=None,
                        help="Only show combined analysis for these seasons (e.g. 2024-25,2025-26)")
    parser.add_argument("--use-model", action="store_true", help="Also run ATS model backtest")
    parser.add_argument("--min-games", type=int, default=30, help="Minimum games for a filter row")
    args = parser.parse_args()

    seasons = [s.strip() for s in args.seasons.split(",")] if args.seasons else None

    print("載入比賽資料...")
    games = load_all_games(seasons)
    print(f"  共 {len(games)} 場已完成比賽 (含讓分資料)")

    print("計算進階特徵...")
    games = add_team_features(games)
    # Drop early-season games where features are NaN
    feature_cols = ["H_form_w10", "A_form_w10"]
    games_with_features = games.dropna(subset=feature_cols).reset_index(drop=True)
    print(f"  有進階特徵的比賽: {len(games_with_features)} 場")

    # If eval-seasons specified, filter for combined analysis
    if args.eval_seasons:
        eval_szns = [s.strip() for s in args.eval_seasons.split(",")]
        eval_mask = games_with_features["Season"].isin(eval_szns)
        eval_games = games_with_features[eval_mask].reset_index(drop=True)
        print(f"  評估賽季 ({args.eval_seasons}): {len(eval_games)} 場")
    else:
        eval_games = games_with_features

    # Run all analyses
    analyze_by_season(games)
    analyze_baseline(eval_games)
    analyze_spread_size(eval_games)
    analyze_home_away_fav(eval_games)
    analyze_form(eval_games)
    analyze_ats_form(eval_games)
    analyze_streaks(eval_games)
    analyze_schedule(eval_games)
    analyze_rest_advantage(eval_games)
    analyze_combined(eval_games)

    if args.use_model:
        analyze_with_model(eval_games)

    print("\n" + "=" * 70)
    print(" 完成！★ 標記表示 WR-NP% ≥ 53% (盈利門檻，假設 -110 juice)")
    print("=" * 70)


if __name__ == "__main__":
    main()
