"""用例行賽資料訓練模型 → 預測季後賽 ATS 回測。

策略：
  1. 僅用例行賽資料訓練 XGBoost (ML + ATS)
  2. 在季後賽上評估準確率、ATS 勝率
  3. 按多維度切片分析：系列賽局數、淘汰局、讓分大小、
     例行賽戰績、休息天數等
  4. Walk-forward: 每一季用該季之前的全部例行賽訓練

Usage:
  PYTHONPATH=. python scripts/backtest_playoff.py
  PYTHONPATH=. python scripts/backtest_playoff.py --seasons 2022-23,2023-24,2024-25
  PYTHONPATH=. python scripts/backtest_playoff.py --trials 50
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

def _find_data_file(filename: str) -> Path:
    """Find data file, checking main repo if worktree copy is empty."""
    candidates = [
        BASE_DIR / "Data" / filename,
        BASE_DIR.parent.parent.parent / "Data" / filename,  # worktree → main repo
    ]
    return next((p for p in candidates if p.exists() and p.stat().st_size > 1000), candidates[0])


DATASET_DB = _find_data_file("dataset.sqlite")
ODDS_DB = _find_data_file("OddsData.sqlite")

DATE_COLUMN = "Date"
TARGET_ML = "Home-Team-Win"
TARGET_ATS = "ATS_Cover"

# Columns to drop for ML model
DROP_COLUMNS_ML = [
    "index", "Score", "Home-Team-Win", "TEAM_NAME", "Date",
    "index.1", "TEAM_NAME.1", "Date.1", "OU-Cover", "OU",
]

# Columns to drop for ATS model (keeps Spread as feature)
DROP_COLUMNS_ATS = [
    "index", "Score", "Home-Team-Win", "TEAM_NAME", "Date",
    "index.1", "TEAM_NAME.1", "Date.1", "OU-Cover", "OU",
    "Win_Margin", "ATS_Cover",
    "Home", "Away", "Points", "ML_Home", "ML_Away",
]

ALL_SEASONS = [
    "2012-13", "2013-14", "2014-15", "2015-16", "2016-17",
    "2017-18", "2018-19", "2019-20", "2020-21", "2021-22",
    "2022-23", "2023-24", "2024-25", "2025-26",
]

# Playoff date ranges (from config.toml)
PLAYOFF_RANGES = {
    "2012-13": ("2013-04-20", "2013-06-20"),
    "2013-14": ("2014-04-19", "2014-06-15"),
    "2014-15": ("2015-04-18", "2015-06-16"),
    "2015-16": ("2016-04-16", "2016-06-19"),
    "2016-17": ("2017-04-15", "2017-06-12"),
    "2017-18": ("2018-04-14", "2018-06-08"),
    "2018-19": ("2019-04-13", "2019-06-13"),
    "2019-20": ("2020-08-15", "2020-10-11"),
    "2020-21": ("2021-05-18", "2021-07-20"),
    "2021-22": ("2022-04-11", "2022-06-16"),
    "2022-23": ("2023-04-11", "2023-06-12"),
    "2023-24": ("2024-04-16", "2024-06-17"),
    "2024-25": ("2025-04-15", "2025-06-22"),
    "2025-26": ("2026-04-14", "2026-06-22"),
}

# Regular season end is the day before playoffs start
REGULAR_SEASON_ENDS = {
    season: (pd.Timestamp(start) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    for season, (start, _) in PLAYOFF_RANGES.items()
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _table_exists(con, name):
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _freshest_odds_table(con, season_key):
    candidates = [
        f"odds_{season_key}_new", f"odds_{season_key}",
        f"{season_key}_new", season_key,
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


def load_dataset_with_odds() -> pd.DataFrame:
    """Load dataset + merge Spread/Win_Margin/ML from OddsData."""
    with sqlite3.connect(DATASET_DB) as con:
        df = pd.read_sql_query('SELECT * FROM "dataset_2012-26"', con)

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    # Merge odds
    frames = []
    with sqlite3.connect(ODDS_DB) as con:
        for season in ALL_SEASONS:
            tbl = _freshest_odds_table(con, season)
            if not tbl:
                continue
            try:
                o = pd.read_sql_query(
                    f'SELECT Date, Home, Away, Spread, Win_Margin, Points, ML_Home, ML_Away '
                    f'FROM "{tbl}"', con,
                )
                frames.append(o)
            except Exception:
                continue
    if frames:
        odds = pd.concat(frames, ignore_index=True)
        odds["Date"] = pd.to_datetime(odds["Date"], errors="coerce")
        odds = odds.drop_duplicates(subset=["Date", "Home", "Away"], keep="last")
        for col in ("Spread", "Win_Margin", "Points", "ML_Home", "ML_Away"):
            odds[col] = pd.to_numeric(odds[col], errors="coerce")
        df = df.merge(
            odds, how="left",
            left_on=["Date", "TEAM_NAME", "TEAM_NAME.1"],
            right_on=["Date", "Home", "Away"],
        )

    return df


def split_regular_playoff(df: pd.DataFrame, season: str):
    """Split into regular-season training and playoff evaluation sets."""
    po_start, po_end = PLAYOFF_RANGES[season]
    reg_end = REGULAR_SEASON_ENDS[season]

    # Training: all regular-season data strictly before this season's playoffs
    train_mask = (df["is_playoff"] == 0) & (df[DATE_COLUMN] < pd.Timestamp(po_start))
    # Evaluation: this season's playoff games
    eval_mask = (
        (df[DATE_COLUMN] >= pd.Timestamp(po_start))
        & (df[DATE_COLUMN] <= pd.Timestamp(po_end))
    )

    return df[train_mask].reset_index(drop=True), df[eval_mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# XGBoost training (reuse from backtest.py)
# ---------------------------------------------------------------------------

class _BoosterWrapper:
    def __init__(self, booster, num_class):
        self.booster = booster
        self.classes_ = np.arange(num_class)

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return self.booster.predict(xgb.DMatrix(X))


def train_xgb(X, y, num_classes, trials=20, seed=42):
    """Train XGBoost with random search + sigmoid calibration."""
    rng = np.random.default_rng(seed)
    n = len(X)
    calib_start = int(n * 0.9)
    X_train, y_train = X[:calib_start], y[:calib_start]
    X_calib, y_calib = X[calib_start:], y[calib_start:]

    counts = np.bincount(y_train, minlength=num_classes)
    total = len(y_train)
    class_w = {c: (total / (num_classes * cnt)) if cnt else 1.0 for c, cnt in enumerate(counts)}
    sw = np.array([class_w[v] for v in y_train])

    best = {"loss": float("inf"), "model": None}
    dval = xgb.DMatrix(X_calib, label=y_calib)
    from sklearn.metrics import log_loss as _log_loss

    for trial in range(trials):
        params = {
            "max_depth": int(rng.integers(2, 13)),
            "eta": float(10 ** rng.uniform(np.log10(0.01), np.log10(0.3))),
            "subsample": float(rng.uniform(0.6, 1.0)),
            "colsample_bytree": float(rng.uniform(0.5, 1.0)),
            "min_child_weight": int(rng.integers(1, 15)),
            "objective": "multi:softprob",
            "num_class": num_classes,
            "eval_metric": "mlogloss",
            "seed": seed + trial,
            "tree_method": "hist",
        }
        nbr = int(rng.integers(200, 1200))
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sw)
        booster = xgb.train(
            params, dtrain, num_boost_round=nbr,
            evals=[(dval, "val")], early_stopping_rounds=40, verbose_eval=False,
        )
        probs = booster.predict(dval)
        loss = _log_loss(y_calib, probs, labels=list(range(num_classes)))
        if loss < best["loss"]:
            best["loss"] = loss
            best["model"] = booster
        if (trial + 1) % 10 == 0:
            print(f"    trial {trial+1}/{trials}: best log loss {best['loss']:.4f}")

    booster = best["model"]
    calibrator = CalibratedClassifierCV(
        _BoosterWrapper(booster, num_classes), method="sigmoid", cv="prefit"
    )
    calibrator.fit(X_calib, y_calib)
    return booster, calibrator


def prepare_features(df, target, drop_cols):
    """Extract X, y from dataframe."""
    y = df[target].astype(int).to_numpy()
    X = df.drop(columns=drop_cols, errors="ignore")
    # Remove non-numeric columns from odds merge
    for col in ("Home", "Away", "ML_Home", "ML_Away", "Spread", "Win_Margin", "Points"):
        if col in X.columns:
            X = X.drop(columns=col)
    X = X.astype(float).to_numpy()
    return X, y


def predict_probs(booster, calibrator, X):
    if calibrator is not None:
        return np.asarray(calibrator.predict_proba(X))
    return np.asarray(booster.predict(xgb.DMatrix(X)))


# ---------------------------------------------------------------------------
# Per-team rolling features for playoff games
# ---------------------------------------------------------------------------

def add_team_form_from_odds(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Compute team form features from OddsData for playoff evaluation."""
    frames = []
    with sqlite3.connect(ODDS_DB) as con:
        for season in ALL_SEASONS:
            tbl = _freshest_odds_table(con, season)
            if not tbl:
                continue
            try:
                df = pd.read_sql_query(
                    f'SELECT Date, Home, Away, Spread, Win_Margin, Points FROM "{tbl}"', con,
                )
                frames.append(df)
            except Exception:
                continue
    if not frames:
        return eval_df

    games = pd.concat(frames, ignore_index=True)
    games = games.drop_duplicates(subset=["Date", "Home", "Away"], keep="last")
    games["Date"] = pd.to_datetime(games["Date"], errors="coerce")
    for col in ("Spread", "Win_Margin", "Points"):
        games[col] = pd.to_numeric(games[col], errors="coerce")
    games = games[games["Points"].fillna(0) > 0].reset_index(drop=True)
    games = games.sort_values("Date").reset_index(drop=True)

    # Build long-form team game log
    home = games[["Date", "Home", "Away", "Spread", "Win_Margin"]].copy()
    home.columns = ["Date", "Team", "Opp", "Spread", "Win_Margin"]
    home["Won"] = (home["Win_Margin"] > 0).astype(int)
    home["CoverMargin"] = home["Win_Margin"] - home["Spread"]
    home["Covered"] = (home["CoverMargin"] > 0).astype(int)

    away = games[["Date", "Home", "Away", "Spread", "Win_Margin"]].copy()
    away.columns = ["Date", "Opp", "Team", "Spread", "Win_Margin"]
    away["Won"] = (away["Win_Margin"] < 0).astype(int)
    away["CoverMargin"] = -(away["Win_Margin"] - away["Spread"])
    away["Covered"] = (away["CoverMargin"] > 0).astype(int)

    long = pd.concat([home, away], ignore_index=True).sort_values(["Team", "Date"]).reset_index(drop=True)
    grp = long.groupby("Team", sort=False)

    won_prev = grp["Won"].shift(1)
    cov_prev = grp["Covered"].shift(1)

    long["reg_w10"] = won_prev.groupby(long["Team"]).transform(
        lambda s: s.rolling(10, min_periods=5).mean()
    )
    long["reg_w20"] = won_prev.groupby(long["Team"]).transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    long["reg_ats10"] = cov_prev.groupby(long["Team"]).transform(
        lambda s: s.rolling(10, min_periods=5).mean()
    )

    # Streaks
    def _streak(series, value):
        out = []
        c = 0
        for v in series:
            if pd.isna(v):
                c = 0; out.append(0); continue
            c = c + 1 if v == value else 0
            out.append(c)
        return pd.Series(out, index=series.index)

    long["streak_w"] = grp["Won"].shift(1).groupby(long["Team"], sort=False).transform(
        lambda s: _streak(s, 1.0)
    )

    feature_cols = ["reg_w10", "reg_w20", "reg_ats10", "streak_w"]

    # Merge home team features
    home_f = long[["Team", "Date"] + feature_cols].rename(
        columns={c: f"H_{c}" for c in feature_cols} | {"Team": "TEAM_NAME"}
    )
    eval_df = eval_df.merge(home_f, on=["TEAM_NAME", "Date"], how="left")

    # Merge away team features
    away_f = long[["Team", "Date"] + feature_cols].rename(
        columns={c: f"A_{c}" for c in feature_cols} | {"Team": "TEAM_NAME.1"}
    )
    eval_df = eval_df.merge(away_f, on=["TEAM_NAME.1", "Date"], how="left")

    return eval_df


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

ATS_JUICE = 110  # -110 standard

def print_section(title):
    print(f"\n{'='*72}")
    print(f" {title}")
    print(f"{'='*72}")


def print_header():
    print(f"  {'Filter':<50s}  {'N':>4s}  {'W':>4s}  {'WR%':>6s}  {'ROI%':>7s}")
    print(f"  {'-'*50}  {'-'*4}  {'-'*4}  {'-'*6}  {'-'*7}")


def print_row(label, n, wins, highlight=0.53):
    if n < 10:
        return
    wr = wins / n
    roi = (wins * 100 / ATS_JUICE - (n - wins)) / n * 100
    marker = " ★" if wr >= highlight else ""
    print(f"  {label:<50s}  {n:>4d}  {wins:>4d}  {wr*100:>5.1f}%  {roi:>+6.1f}%{marker}")


def analyze_ml_on_playoffs(probs_ml, eval_df):
    """ML model accuracy on playoff games."""
    y = eval_df[TARGET_ML].astype(int).to_numpy()
    pred = np.argmax(probs_ml, axis=1)
    acc = (pred == y).mean()
    n = len(y)
    correct = int((pred == y).sum())
    print(f"  ML 準確率: {acc:.4f} ({correct}/{n})")
    return acc


def analyze_ats_on_playoffs(probs_ml, eval_df):
    """Use ML model probabilities to determine ATS picks and evaluate."""
    spreads = eval_df["Spread"].to_numpy()
    margins = eval_df["Win_Margin"].to_numpy()
    y_ml = eval_df[TARGET_ML].astype(int).to_numpy()

    # Determine ATS outcome
    home_cover_margin = margins - spreads
    home_covered = home_cover_margin > 0
    away_covered = home_cover_margin < 0
    push = home_cover_margin == 0

    # prob_home = probs_ml[:, 1] (home win probability)
    prob_home = probs_ml[:, 1]

    return prob_home, home_covered, away_covered, push, spreads, margins


def sweep_confidence(prob_home, home_covered, away_covered, push, spreads,
                     eval_df, label_prefix=""):
    """Sweep model confidence thresholds for ATS betting."""
    n_total = len(prob_home)
    valid = ~push & ~np.isnan(spreads)

    print_header()

    # Strategy 1: Bet home cover when model says home wins with high confidence
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        mask = (prob_home >= thresh) & valid
        n = mask.sum()
        wins = home_covered[mask].sum()
        print_row(f"{label_prefix}模型信心 ≥{thresh:.0%} → 押主場 cover", n, wins)

    # Strategy 2: Bet away cover when model says away wins with high confidence
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        mask = (prob_home <= (1 - thresh)) & valid
        n = mask.sum()
        wins = away_covered[mask].sum()
        print_row(f"{label_prefix}模型信心 ≥{thresh:.0%} → 押客場 cover", n, wins)

    # Strategy 3: Bet favorite cover / underdog cover
    home_fav = spreads > 0
    for thresh in [0.55, 0.60, 0.65, 0.70]:
        # High confidence on favorite
        fav_mask = ((home_fav & (prob_home >= thresh)) |
                    (~home_fav & (prob_home <= 1 - thresh))) & valid
        n = fav_mask.sum()
        fav_wins = (home_fav & home_covered & (prob_home >= thresh) & valid).sum() + \
                   (~home_fav & away_covered & (prob_home <= 1 - thresh) & valid).sum()
        print_row(f"{label_prefix}信心 ≥{thresh:.0%} → 押熱門 cover", n, int(fav_wins))

        # High confidence on underdog
        dog_mask = ((~home_fav & (prob_home >= thresh)) |
                    (home_fav & (prob_home <= 1 - thresh))) & valid
        n = dog_mask.sum()
        dog_wins = (~home_fav & home_covered & (prob_home >= thresh) & valid).sum() + \
                   (home_fav & away_covered & (prob_home <= 1 - thresh) & valid).sum()
        print_row(f"{label_prefix}信心 ≥{thresh:.0%} → 押冷門 cover", n, int(dog_wins))


def analyze_by_dimension(prob_home, home_covered, away_covered, push, spreads,
                         margins, eval_df):
    """Analyze playoff ATS by multiple dimensions."""
    valid = ~push & ~np.isnan(spreads)
    abs_spread = np.abs(spreads)

    # --- Series game number ---
    print_section("季後賽維度 1: 系列賽局數")
    print_header()
    series_gn = eval_df["series_game_num"].fillna(0).to_numpy()
    for gn in [1, 2, 3, 4, 5, 6, 7]:
        mask = (series_gn == gn) & valid
        n = mask.sum()
        # Home cover rate
        h_wins = home_covered[mask].sum()
        print_row(f"Game {gn} → 押主場 cover", n, h_wins)
        a_wins = away_covered[mask].sum()
        print_row(f"Game {gn} → 押客場 cover", n, a_wins)

    # Early vs Late games
    early = (series_gn <= 2) & valid
    mid = ((series_gn >= 3) & (series_gn <= 5)) & valid
    late = (series_gn >= 6) & valid
    for mask, label in [(early, "Game 1-2"), (mid, "Game 3-5"), (late, "Game 6-7")]:
        n = mask.sum()
        print_row(f"{label} → 押主場", n, home_covered[mask].sum())
        print_row(f"{label} → 押客場", n, away_covered[mask].sum())
        print_row(f"{label} → 押冷門", n,
                  int((((spreads > 0) & away_covered) | ((spreads < 0) & home_covered))[mask].sum()))

    # --- Elimination games ---
    print_section("季後賽維度 2: 淘汰局")
    print_header()
    is_elim = eval_df["is_elimination_game"].fillna(0).to_numpy() == 1
    not_elim = ~is_elim
    for mask, label in [(is_elim, "淘汰局"), (not_elim & valid, "非淘汰局")]:
        mask = mask & valid
        n = mask.sum()
        print_row(f"{label} → 押主場", n, home_covered[mask].sum())
        print_row(f"{label} → 押客場", n, away_covered[mask].sum())
        print_row(f"{label} → 押冷門", n,
                  int((((spreads > 0) & away_covered) | ((spreads < 0) & home_covered))[mask].sum()))
        print_row(f"{label} → 押熱門", n,
                  int((((spreads > 0) & home_covered) | ((spreads < 0) & away_covered))[mask].sum()))

    # --- Series lead ---
    print_section("季後賽維度 3: 系列賽領先/落後")
    print_header()
    series_lead = eval_df["series_lead_for_home"].fillna(0).to_numpy()
    for lead_range, label in [
        ((2, 3), "主場大幅領先 (2-3 場)"),
        ((1, 1), "主場小幅領先 (1 場)"),
        ((0, 0), "系列賽平手"),
        ((-1, -1), "主場小幅落後 (1 場)"),
        ((-3, -2), "主場大幅落後 (2-3 場)"),
    ]:
        lo, hi = lead_range
        mask = (series_lead >= lo) & (series_lead <= hi) & valid
        n = mask.sum()
        print_row(f"{label} → 押主場", n, home_covered[mask].sum())
        print_row(f"{label} → 押客場", n, away_covered[mask].sum())

    # --- Spread size ---
    print_section("季後賽維度 4: 讓分大小")
    print_header()
    for lo, hi in [(0, 3), (3, 5.5), (5.5, 8), (8, 15), (15, 30)]:
        mask = (abs_spread >= lo) & (abs_spread < hi) & valid
        n = mask.sum()
        print_row(f"讓分 {lo}-{hi} → 押熱門", n,
                  int((((spreads > 0) & home_covered) | ((spreads < 0) & away_covered))[mask].sum()))
        print_row(f"讓分 {lo}-{hi} → 押冷門", n,
                  int((((spreads > 0) & away_covered) | ((spreads < 0) & home_covered))[mask].sum()))

    # --- Regular season form entering playoffs ---
    print_section("季後賽維度 5: 進入季後賽前的例行賽戰績")
    print_header()
    h_form = eval_df.get("H_reg_w20", pd.Series(dtype=float)).fillna(0.5).to_numpy()
    a_form = eval_df.get("A_reg_w20", pd.Series(dtype=float)).fillna(0.5).to_numpy()

    for thresh in [0.65, 0.60, 0.55]:
        # Strong home team
        mask = (h_form >= thresh) & valid
        n = mask.sum()
        print_row(f"主場例行賽 L20 勝率 ≥{thresh:.0%} → 押主場", n, home_covered[mask].sum())
        print_row(f"主場例行賽 L20 勝率 ≥{thresh:.0%} → 押客場", n, away_covered[mask].sum())

    for thresh in [0.65, 0.60, 0.55]:
        # Strong away team
        mask = (a_form >= thresh) & valid
        n = mask.sum()
        print_row(f"客場例行賽 L20 勝率 ≥{thresh:.0%} → 押客場", n, away_covered[mask].sum())
        print_row(f"客場例行賽 L20 勝率 ≥{thresh:.0%} → 押主場", n, home_covered[mask].sum())

    # Form diff
    form_diff = h_form - a_form
    for lo, hi, label in [
        (0.15, 1.0, "主場明顯強於客場 (Δ≥15%)"),
        (-0.05, 0.05, "雙方戰力接近 (Δ<5%)"),
        (-1.0, -0.15, "客場明顯強於主場 (Δ≥15%)"),
    ]:
        mask = (form_diff >= lo) & (form_diff <= hi) & valid
        n = mask.sum()
        print_row(f"{label} → 押主場", n, home_covered[mask].sum())
        print_row(f"{label} → 押客場", n, away_covered[mask].sum())
        print_row(f"{label} → 押冷門", n,
                  int((((spreads > 0) & away_covered) | ((spreads < 0) & home_covered))[mask].sum()))

    # --- ATS form entering playoffs ---
    print_section("季後賽維度 6: 例行賽 ATS 表現")
    print_header()
    h_ats = eval_df.get("H_reg_ats10", pd.Series(dtype=float)).fillna(0.5).to_numpy()
    a_ats = eval_df.get("A_reg_ats10", pd.Series(dtype=float)).fillna(0.5).to_numpy()

    for thresh in [0.7, 0.6, 0.4, 0.3]:
        if thresh >= 0.5:
            mask = (h_ats >= thresh) & valid
            n = mask.sum()
            print_row(f"主場例行賽 ATS L10 ≥{thresh:.0%} → 押主場", n, home_covered[mask].sum())
            mask = (a_ats >= thresh) & valid
            n = mask.sum()
            print_row(f"客場例行賽 ATS L10 ≥{thresh:.0%} → 押客場", n, away_covered[mask].sum())
        else:
            mask = (h_ats <= thresh) & valid
            n = mask.sum()
            print_row(f"主場例行賽 ATS L10 ≤{thresh:.0%} → 押主場(反彈)", n, home_covered[mask].sum())
            print_row(f"主場例行賽 ATS L10 ≤{thresh:.0%} → 押客場", n, away_covered[mask].sum())
            mask = (a_ats <= thresh) & valid
            n = mask.sum()
            print_row(f"客場例行賽 ATS L10 ≤{thresh:.0%} → 押客場(反彈)", n, away_covered[mask].sum())


def analyze_combined_playoff(prob_home, home_covered, away_covered, push, spreads,
                             eval_df):
    """Multi-condition combined filters for playoff ATS."""
    print_section("季後賽維度 7: 組合篩選 (高勝率策略)")
    valid = ~push & ~np.isnan(spreads)
    abs_spread = np.abs(spreads)
    series_gn = eval_df["series_game_num"].fillna(0).to_numpy()
    series_lead = eval_df["series_lead_for_home"].fillna(0).to_numpy()
    is_elim = eval_df["is_elimination_game"].fillna(0).to_numpy() == 1
    h_form = eval_df.get("H_reg_w20", pd.Series(dtype=float)).fillna(0.5).to_numpy()
    a_form = eval_df.get("A_reg_w20", pd.Series(dtype=float)).fillna(0.5).to_numpy()
    home_fav = spreads > 0

    results = []

    def _add(label, mask, side="home"):
        mask = mask & valid
        n = int(mask.sum())
        if n < 10:
            return
        if side == "home":
            wins = int(home_covered[mask].sum())
        elif side == "away":
            wins = int(away_covered[mask].sum())
        elif side == "dog":
            wins = int((((spreads > 0) & away_covered) | ((spreads < 0) & home_covered))[mask].sum())
        elif side == "fav":
            wins = int((((spreads > 0) & home_covered) | ((spreads < 0) & away_covered))[mask].sum())
        else:
            return
        wr = wins / n if n else 0
        results.append((label, n, wins, wr))

    # Elimination + underdog
    _add("淘汰局 + 押冷門", is_elim, "dog")
    _add("淘汰局 + 押熱門", is_elim, "fav")

    # Trailing team in elimination (must-win)
    trailing_home_elim = is_elim & (series_lead < 0)
    trailing_away_elim = is_elim & (series_lead > 0)
    _add("主場淘汰局(落後方) → 押主場", trailing_home_elim, "home")
    _add("客場淘汰局(落後方) → 押客場", trailing_away_elim, "away")

    # Leading team
    _add("主場領先 2+ → 押主場 cover", (series_lead >= 2), "home")
    _add("主場領先 2+ → 押客場 cover", (series_lead >= 2), "away")
    _add("客場領先 2+ → 押客場 cover", (series_lead <= -2), "away")
    _add("客場領先 2+ → 押主場 cover", (series_lead <= -2), "home")

    # Game 1 (no series context yet)
    _add("Game 1 → 押主場", series_gn == 1, "home")
    _add("Game 1 → 押冷門", series_gn == 1, "dog")

    # Game 5 (pivot game)
    _add("Game 5 (tied 2-2) → 押主場", (series_gn == 5) & (series_lead == 0), "home")
    _add("Game 5 (tied 2-2) → 押客場", (series_gn == 5) & (series_lead == 0), "away")

    # Model confidence + series context
    for thresh in [0.60, 0.65, 0.70]:
        _add(f"模型 ≥{thresh:.0%} + 非淘汰局 → 押主場",
             (prob_home >= thresh) & ~is_elim, "home")
        _add(f"模型 ≥{thresh:.0%} + 淘汰局 → 押主場",
             (prob_home >= thresh) & is_elim, "home")
        _add(f"模型 ≥{thresh:.0%} + 小讓分(≤5) → 押主場",
             (prob_home >= thresh) & (abs_spread <= 5), "home")

    # Strong regular season team as underdog in playoffs
    _add("例行賽強隊受讓 (H_form≥60% + 客讓) → 押主場",
         (h_form >= 0.60) & ~home_fav, "home")
    _add("例行賽強隊受讓 (A_form≥60% + 主讓) → 押客場",
         (a_form >= 0.60) & home_fav, "away")

    # Spread size + series context
    for sp_hi in [5, 8]:
        _add(f"小讓分(≤{sp_hi}) + 淘汰局 → 押冷門",
             (abs_spread <= sp_hi) & is_elim, "dog")
        _add(f"小讓分(≤{sp_hi}) + Game 6-7 → 押主場",
             (abs_spread <= sp_hi) & (series_gn >= 6), "home")

    # Late series + trailing = desperation
    _add("Game 5+ + 落後方是主場 → 押主場",
         (series_gn >= 5) & (series_lead < 0), "home")
    _add("Game 5+ + 落後方是客場 → 押客場",
         (series_gn >= 5) & (series_lead > 0), "away")

    # Sort by win rate
    results.sort(key=lambda x: -x[3])

    print(f"\n  Top 組合篩選 (依勝率排序, ≥10 場):\n")
    print_header()
    for label, n, wins, wr in results:
        print_row(label, n, wins, highlight=0.53)


# ---------------------------------------------------------------------------
# Main: walk-forward per season
# ---------------------------------------------------------------------------

def run_season(df, season, trials, seed):
    """Train on regular season < playoff start, evaluate on this season's playoffs."""
    if season not in PLAYOFF_RANGES:
        print(f"  No playoff range for {season}, skip")
        return None

    train_df, eval_df = split_regular_playoff(df, season)
    # Need odds columns for ATS
    eval_df = eval_df.dropna(subset=["Spread", "Win_Margin"]).reset_index(drop=True)
    eval_df = eval_df[eval_df["Points"].fillna(0) > 0].reset_index(drop=True)

    # Compute ATS target
    eval_df["ATS_Cover"] = ((eval_df["Win_Margin"] - eval_df["Spread"]) > 0).astype(int)

    if len(eval_df) < 10:
        print(f"  {season}: only {len(eval_df)} playoff games with odds, skip")
        return None

    print(f"\n  {season}: {len(train_df)} train (reg season) → {len(eval_df)} playoff games")

    # Train ML model on regular season only
    train_df = train_df.dropna(subset=[TARGET_ML]).reset_index(drop=True)
    X_train, y_train = prepare_features(train_df, TARGET_ML, DROP_COLUMNS_ML)
    print(f"    Training ML model ({X_train.shape[1]} features)...")
    booster, calibrator = train_xgb(X_train, y_train, num_classes=2, trials=trials, seed=seed)

    # Predict on playoff games
    X_eval, y_eval = prepare_features(eval_df, TARGET_ML, DROP_COLUMNS_ML)

    # Align feature count (train may have fewer columns if eval has new ones)
    if X_eval.shape[1] > X_train.shape[1]:
        X_eval = X_eval[:, :X_train.shape[1]]
    elif X_eval.shape[1] < X_train.shape[1]:
        pad = np.zeros((X_eval.shape[0], X_train.shape[1] - X_eval.shape[1]))
        X_eval = np.hstack([X_eval, pad])

    probs_ml = predict_probs(booster, calibrator, X_eval)

    return {
        "season": season,
        "probs_ml": probs_ml,
        "eval_df": eval_df,
        "y_ml": y_eval,
    }


def main():
    parser = argparse.ArgumentParser(description="用例行賽訓練 → 預測季後賽 ATS 回測")
    parser.add_argument("--seasons", default="2012-13,2013-14,2014-15,2015-16,2016-17,2017-18,2018-19,2019-20,2020-21,2021-22,2022-23,2023-24,2024-25",
                        help="Comma-separated seasons to evaluate")
    parser.add_argument("--trials", type=int, default=20, help="XGBoost random search trials")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seasons = [s.strip() for s in args.seasons.split(",")]

    print("載入資料...")
    df = load_dataset_with_odds()
    print(f"  Dataset: {len(df)} rows")

    # Run walk-forward for each season
    all_results = []
    for season in seasons:
        result = run_season(df, season, args.trials, args.seed)
        if result is not None:
            all_results.append(result)

    if not all_results:
        print("No valid seasons to evaluate.")
        return

    # Combine all playoff evaluations
    combined_probs = np.concatenate([r["probs_ml"] for r in all_results])
    combined_eval = pd.concat([r["eval_df"] for r in all_results], ignore_index=True)
    combined_y = np.concatenate([r["y_ml"] for r in all_results])

    print_section(f"綜合季後賽回測結果 ({len(all_results)} 季, {len(combined_eval)} 場)")

    # ML accuracy per season
    print("\n  各賽季 ML 準確率:")
    print(f"  {'Season':<12s}  {'Games':>6s}  {'Acc':>6s}")
    for r in all_results:
        pred = np.argmax(r["probs_ml"], axis=1)
        y = r["y_ml"]
        acc = (pred == y).mean()
        print(f"  {r['season']:<12s}  {len(y):>6d}  {acc*100:>5.1f}%")

    total_acc = (np.argmax(combined_probs, axis=1) == combined_y).mean()
    print(f"  {'TOTAL':<12s}  {len(combined_y):>6d}  {total_acc*100:>5.1f}%")

    # Add team form features for dimensional analysis
    print("\n  計算球隊戰績特徵...")
    combined_eval = add_team_form_from_odds(combined_eval)

    # ATS analysis
    prob_home, home_covered, away_covered, push, spreads, margins = \
        analyze_ats_on_playoffs(combined_probs, combined_eval)

    # Overall ATS baseline
    valid = ~push & ~np.isnan(spreads)
    total_valid = valid.sum()
    h_wr = home_covered[valid].sum() / total_valid
    a_wr = away_covered[valid].sum() / total_valid
    print(f"\n  季後賽 ATS 基準線 ({total_valid} 場, 不含 push):")
    print(f"    主場 cover: {h_wr*100:.1f}%")
    print(f"    客場 cover: {a_wr*100:.1f}%")
    fav_wins = int((((spreads > 0) & home_covered) | ((spreads < 0) & away_covered))[valid].sum())
    dog_wins = int((((spreads > 0) & away_covered) | ((spreads < 0) & home_covered))[valid].sum())
    print(f"    熱門 cover: {fav_wins/total_valid*100:.1f}%")
    print(f"    冷門 cover: {dog_wins/total_valid*100:.1f}%")

    # Model confidence sweep
    print_section("模型信心度篩選")
    sweep_confidence(prob_home, home_covered, away_covered, push, spreads,
                     combined_eval, label_prefix="")

    # Dimensional analyses
    analyze_by_dimension(prob_home, home_covered, away_covered, push, spreads,
                         margins, combined_eval)

    # Combined filters
    analyze_combined_playoff(prob_home, home_covered, away_covered, push, spreads,
                             combined_eval)

    print(f"\n{'='*72}")
    print(f" 完成！★ = WR ≥ 53% (假設 -110 juice 的盈利門檻)")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
