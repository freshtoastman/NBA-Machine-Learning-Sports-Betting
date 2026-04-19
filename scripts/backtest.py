"""Backtest XGBoost ML/UO models against historical NBA games.

Modes:
  --mode test-split    Use existing trained models on the most recent 10%
                       chronological slice (matches the training script's
                       test split). Fast.
  --mode walk-forward  For each season specified, train fresh models on
                       all data BEFORE the season start, then evaluate on
                       the season. Slower but no in-season leakage.

Outputs accuracy, calibration, ROI (flat $1 + Kelly), and breakdowns by
month and team.

Usage:
  PYTHONPATH=. python scripts/backtest.py --mode test-split
  PYTHONPATH=. python scripts/backtest.py --mode walk-forward --seasons 2024-25,2025-26
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV

DATASET_DB = Path("Data") / "dataset.sqlite"
XGB_MODEL_DIR = Path("Models") / "XGBoost_Models"
DATE_COLUMN = "Date"
TARGET_ML = "Home-Team-Win"
TARGET_UO = "OU-Cover"
NUM_CLASSES_ML = 2
NUM_CLASSES_UO = 3

DROP_COLUMNS_ML = [
    "index", "Score", "Home-Team-Win", "TEAM_NAME", "Date",
    "index.1", "TEAM_NAME.1", "Date.1", "OU-Cover", "OU",
]
DROP_COLUMNS_UO = [
    "index", "Score", "Home-Team-Win", "TEAM_NAME", "Date",
    "index.1", "TEAM_NAME.1", "Date.1", "OU-Cover",
]


class _BoosterWrapper:
    _estimator_type = "classifier"

    def __init__(self, booster, num_class):
        self.booster = booster
        self.classes_ = np.arange(num_class)

    def fit(self, X, y):
        return self

    def predict(self, X):
        proba = self.booster.predict(xgb.DMatrix(X))
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        return self.booster.predict(xgb.DMatrix(X))


def train_xgb_quick(X, y, num_classes, trials=20, seed=42):
    """Light-weight XGBoost trainer for walk-forward use.

    Splits last 10% for calibration. Random search across `trials` parameter sets.
    """
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
        from sklearn.metrics import log_loss
        probs = booster.predict(dval)
        loss = log_loss(y_calib, probs, labels=list(range(num_classes)))
        if loss < best["loss"]:
            best["loss"] = loss
            best["model"] = booster
        print(f"  trial {trial+1}/{trials}: log loss {loss:.4f}")

    booster = best["model"]
    calibrator = CalibratedClassifierCV(_BoosterWrapper(booster, num_classes), method="sigmoid", cv="prefit")
    calibrator.fit(X_calib, y_calib)
    return booster, calibrator

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

ODDS_DB = Path("Data") / "OddsData.sqlite"


def load_dataset(table_name: str = "dataset_2012-26") -> pd.DataFrame:
    with sqlite3.connect(DATASET_DB) as con:
        df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', con)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    return df


def load_odds_for_seasons(seasons: list[str]) -> pd.DataFrame:
    """Load all OddsData season tables and concatenate."""
    frames = []
    with sqlite3.connect(ODDS_DB) as con:
        for season in seasons:
            try:
                df = pd.read_sql_query(f'SELECT * FROM "{season}"', con)
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                frames.append(df)
            except Exception as exc:  # noqa: BLE001
                print(f"warn: could not load odds for {season}: {exc}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def attach_odds(games: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Join money lines onto game rows on (Date, Home, Away).

    Dataset has columns 'TEAM_NAME' (home) and 'TEAM_NAME.1' (away).
    """
    if odds.empty:
        games["ML_Home"] = np.nan
        games["ML_Away"] = np.nan
        return games

    odds_subset = odds[["Date", "Home", "Away", "ML_Home", "ML_Away"]].copy()
    games = games.merge(
        odds_subset,
        how="left",
        left_on=[DATE_COLUMN, "TEAM_NAME", "TEAM_NAME.1"],
        right_on=["Date", "Home", "Away"],
        suffixes=("", "_odds"),
    )
    return games


def split_features_labels(df: pd.DataFrame, target: str, drop_cols: list[str]):
    y = df[target].astype(int).to_numpy()
    X = df.drop(columns=drop_cols, errors="ignore")
    # Drop any non-numeric columns we may have introduced via merge.
    for col in ("Home", "Away", "ML_Home", "ML_Away"):
        if col in X.columns:
            X = X.drop(columns=col)
    X = X.astype(float).to_numpy()
    return X, y


# ---------------------------------------------------------------------------
# Model loading / prediction
# ---------------------------------------------------------------------------

def latest_xgb_model(kind: str):
    """Pick newest+highest-accuracy XGBoost model + calibrator (kind in {ML, UO})."""
    candidates = list(XGB_MODEL_DIR.glob(f"*{kind}*.json"))
    if not candidates:
        raise FileNotFoundError(f"No XGBoost {kind} model in {XGB_MODEL_DIR}")
    import re
    import sys
    pat = re.compile(r"XGBoost_(\d+(?:\.\d+)?)%_")

    def score(p):
        m = pat.search(p.name)
        return (p.stat().st_mtime, float(m.group(1)) if m else 0.0)

    best = max(candidates, key=score)
    booster = xgb.Booster()
    booster.load_model(str(best))
    calib_path = best.with_name(f"{best.stem}_calibration.pkl")
    calibrator = None
    if calib_path.exists():
        # Calibrator was pickled with BoosterWrapper defined in the trainer's
        # __main__. Inject our local wrapper into __main__ so unpickling works.
        sys.modules["__main__"].BoosterWrapper = _BoosterWrapper
        try:
            calibrator = joblib.load(calib_path)
        except Exception as exc:
            print(f"warn: failed to load calibrator ({exc}); using raw probs")
            calibrator = None
    return booster, calibrator


def predict_probs(booster, calibrator, X):
    if calibrator is not None:
        return calibrator.predict_proba(X)
    return booster.predict(xgb.DMatrix(X))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def american_to_decimal(odds: float) -> float:
    if odds >= 100:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def kelly_fraction(prob: float, american_odds: float) -> float:
    """Kelly criterion fraction; clip to [0, 1]."""
    b = american_to_decimal(american_odds) - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - prob
    f = (b * prob - q) / b
    return max(0.0, min(1.0, f))


def expected_value(prob: float, american_odds: float) -> float:
    decimal = american_to_decimal(american_odds)
    return prob * (decimal - 1.0) - (1.0 - prob)


def calibration_table(probs_pos, y_true, n_bins=10):
    """Return list of (bin_low, bin_high, count, predicted_mean, actual_mean)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (probs_pos >= lo) & (probs_pos <= hi)
        else:
            mask = (probs_pos >= lo) & (probs_pos < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append((lo, hi, 0, None, None))
            continue
        pred = float(probs_pos[mask].mean())
        actual = float(y_true[mask].mean())
        rows.append((lo, hi, n, pred, actual))
    return rows


def simulate_bets(probs, y_true, ml_home, ml_away, bankroll0=1000.0):
    """Simulate flat-$1 and Kelly betting strategies on ML predictions.

    Bet on the side with higher model probability if EV > 0.
    Returns: dict with stats + bankroll curve.
    """
    flat_pl = 0.0
    kelly_bankroll = bankroll0
    flat_bankroll_curve = [bankroll0]
    kelly_bankroll_curve = [bankroll0]
    n_flat_bets = 0
    n_flat_wins = 0
    n_kelly_bets = 0
    n_kelly_wins = 0
    flat_stake_total = 0.0

    for i in range(len(probs)):
        ml_h = ml_home[i]
        ml_a = ml_away[i]
        if pd.isna(ml_h) or pd.isna(ml_a):
            flat_bankroll_curve.append(bankroll0 + flat_pl)
            kelly_bankroll_curve.append(kelly_bankroll)
            continue
        prob_home = float(probs[i, 1])
        prob_away = 1.0 - prob_home
        ev_home = expected_value(prob_home, ml_h)
        ev_away = expected_value(prob_away, ml_a)

        # Pick side with higher EV among the two; only bet if EV > 0.
        if ev_home <= 0 and ev_away <= 0:
            flat_bankroll_curve.append(bankroll0 + flat_pl)
            kelly_bankroll_curve.append(kelly_bankroll)
            continue
        if ev_home >= ev_away:
            side = "home"
            prob = prob_home
            odds = ml_h
        else:
            side = "away"
            prob = prob_away
            odds = ml_a

        won = (side == "home" and y_true[i] == 1) or (side == "away" and y_true[i] == 0)
        decimal = american_to_decimal(odds)

        # Flat $1 bet
        n_flat_bets += 1
        flat_stake_total += 1.0
        if won:
            n_flat_wins += 1
            flat_pl += decimal - 1.0
        else:
            flat_pl -= 1.0

        # Conservative fractional Kelly (0.10x). Full Kelly is wildly volatile;
        # 0.10 fractional smooths things out while still benefiting from edge.
        f = kelly_fraction(prob, odds) * 0.10
        stake = kelly_bankroll * f
        if stake > 0:
            n_kelly_bets += 1
            if won:
                n_kelly_wins += 1
                kelly_bankroll += stake * (decimal - 1.0)
            else:
                kelly_bankroll -= stake

        flat_bankroll_curve.append(bankroll0 + flat_pl)
        kelly_bankroll_curve.append(kelly_bankroll)

    def max_drawdown(curve):
        peak = curve[0]
        max_dd = 0.0
        for v in curve:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        return max_dd

    return {
        "flat": {
            "n_bets": n_flat_bets,
            "n_wins": n_flat_wins,
            "win_rate": (n_flat_wins / n_flat_bets) if n_flat_bets else 0.0,
            "pl": flat_pl,
            "roi": (flat_pl / flat_stake_total) if flat_stake_total else 0.0,
            "final_bankroll": bankroll0 + flat_pl,
            "max_drawdown": max_drawdown(flat_bankroll_curve),
            "curve": flat_bankroll_curve,
        },
        "kelly": {
            "n_bets": n_kelly_bets,
            "n_wins": n_kelly_wins,
            "win_rate": (n_kelly_wins / n_kelly_bets) if n_kelly_bets else 0.0,
            "final_bankroll": kelly_bankroll,
            "peak_bankroll": max(kelly_bankroll_curve),
            "roi": (kelly_bankroll - bankroll0) / bankroll0,
            "max_drawdown": max_drawdown(kelly_bankroll_curve),
            "curve": kelly_bankroll_curve,
        },
    }


def breakdown_by_key(probs, y_true, keys, label="key"):
    """Group by key and report accuracy + count."""
    bucket = defaultdict(lambda: {"n": 0, "correct": 0})
    pred = np.argmax(probs, axis=1)
    for i in range(len(y_true)):
        k = keys[i]
        bucket[k]["n"] += 1
        if pred[i] == y_true[i]:
            bucket[k]["correct"] += 1
    rows = []
    for k, v in sorted(bucket.items()):
        rows.append((k, v["n"], v["correct"], v["correct"] / v["n"] if v["n"] else 0.0))
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_header(title):
    print()
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)


def report_run(label, y_ml, probs_ml, y_uo, probs_uo, ml_home, ml_away, dates, home_teams, away_teams):
    print_header(f"BACKTEST: {label}")
    n = len(y_ml)
    print(f"Games evaluated: {n}")

    # ML accuracy
    ml_pred = np.argmax(probs_ml, axis=1)
    ml_acc = (ml_pred == y_ml).mean()
    print(f"\nML accuracy:   {ml_acc:.4f}  ({(ml_pred == y_ml).sum()}/{n})")

    # UO accuracy (if available)
    if probs_uo is not None and y_uo is not None:
        uo_pred = np.argmax(probs_uo, axis=1)
        uo_acc = (uo_pred == y_uo).mean()
        print(f"UO accuracy:   {uo_acc:.4f}  ({(uo_pred == y_uo).sum()}/{n})")

    # ML calibration
    print("\nML calibration (P(home_win) buckets):")
    print(f"  {'bin':>12}  {'count':>6}  {'pred':>8}  {'actual':>8}")
    for lo, hi, cnt, pred, actual in calibration_table(probs_ml[:, 1], y_ml):
        if cnt == 0:
            print(f"  {lo:.2f}-{hi:.2f}    {cnt:>6}      --        --")
        else:
            print(f"  {lo:.2f}-{hi:.2f}    {cnt:>6}    {pred:.4f}    {actual:.4f}")

    # ROI
    valid_odds = (~pd.isna(ml_home)).sum()
    print(f"\nGames with money line odds: {valid_odds}/{n}")
    if valid_odds > 0:
        sim = simulate_bets(probs_ml, y_ml, ml_home, ml_away)
        print("\nFlat $1 betting (only when EV > 0):")
        print(f"  bets:           {sim['flat']['n_bets']}")
        print(f"  win rate:       {sim['flat']['win_rate']:.4f}")
        print(f"  P/L:            ${sim['flat']['pl']:+.2f}")
        print(f"  ROI per bet:    {sim['flat']['roi']*100:+.2f}%")
        print(f"  max drawdown:   {sim['flat']['max_drawdown']*100:.1f}%")
        print("\nFractional Kelly betting (0.10x Kelly, $1000 bankroll):")
        print(f"  bets:           {sim['kelly']['n_bets']}")
        print(f"  win rate:       {sim['kelly']['win_rate']:.4f}")
        print(f"  peak bankroll:  ${sim['kelly']['peak_bankroll']:.2f}")
        print(f"  final bankroll: ${sim['kelly']['final_bankroll']:.2f}")
        print(f"  total return:   {sim['kelly']['roi']*100:+.2f}%")
        print(f"  max drawdown:   {sim['kelly']['max_drawdown']*100:.1f}%")
        if abs(sim['flat']['roi']) < 0.01:
            print("  note: flat ROI ≈ 0 → Kelly result is path-dependent noise, not skill.")

    # By month
    months = pd.to_datetime(pd.Series(dates).reset_index(drop=True)).dt.strftime("%Y-%m").to_numpy()
    print("\nBy month (ML accuracy):")
    print(f"  {'month':<10}  {'games':>6}  {'wins':>6}  {'acc':>8}")
    for k, total, wins, acc in breakdown_by_key(probs_ml, y_ml, months):
        print(f"  {k:<10}  {total:>6}  {wins:>6}  {acc:.4f}")

    # By home team
    print("\nBy home team (ML accuracy, top by games):")
    rows = breakdown_by_key(probs_ml, y_ml, home_teams)
    rows.sort(key=lambda r: -r[1])
    print(f"  {'team':<28}  {'games':>6}  {'wins':>6}  {'acc':>8}")
    for k, total, wins, acc in rows:
        print(f"  {k:<28}  {total:>6}  {wins:>6}  {acc:.4f}")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

SEASON_BOUNDS = {
    "2023-24": ("2023-10-01", "2024-09-30"),
    "2024-25": ("2024-10-01", "2025-09-30"),
    "2025-26": ("2025-10-01", "2026-09-30"),
}


def run_test_split(seasons):
    """Use existing trained models. Evaluate on the most recent 10% slice
    intersected with the requested seasons (so we report by-season too)."""
    df = load_dataset()
    odds = load_odds_for_seasons(seasons)
    df = attach_odds(df, odds)

    # Drop rows missing labels
    df = df.dropna(subset=[TARGET_ML, TARGET_UO]).reset_index(drop=True)

    # Test split = last 10% chronological (matches trainer)
    n = len(df)
    test_start = int(n * 0.9)
    test_df = df.iloc[test_start:].reset_index(drop=True)
    print(f"Total dataset rows: {n}; test split rows: {len(test_df)}")
    print(f"Test split date range: {test_df[DATE_COLUMN].min().date()} → {test_df[DATE_COLUMN].max().date()}")

    # Filter further by requested season
    season_mask = pd.Series(False, index=test_df.index)
    for s in seasons:
        if s not in SEASON_BOUNDS:
            print(f"warn: unknown season {s}")
            continue
        lo, hi = SEASON_BOUNDS[s]
        season_mask |= (test_df[DATE_COLUMN] >= lo) & (test_df[DATE_COLUMN] <= hi)
    eval_df = test_df[season_mask].reset_index(drop=True)
    print(f"Filtered to seasons {seasons}: {len(eval_df)} games")
    if eval_df.empty:
        print("No rows in the test split overlap requested seasons.")
        return

    booster_ml, calib_ml = latest_xgb_model("ML")
    booster_uo, calib_uo = latest_xgb_model("UO")

    X_ml, y_ml = split_features_labels(eval_df, TARGET_ML, DROP_COLUMNS_ML)
    X_uo, y_uo = split_features_labels(eval_df, TARGET_UO, DROP_COLUMNS_UO)

    probs_ml = predict_probs(booster_ml, calib_ml, X_ml)
    probs_uo = predict_probs(booster_uo, calib_uo, X_uo)

    report_run(
        f"test-split, seasons={','.join(seasons)}",
        y_ml=y_ml,
        probs_ml=np.asarray(probs_ml),
        y_uo=y_uo,
        probs_uo=np.asarray(probs_uo),
        ml_home=eval_df["ML_Home"].to_numpy(),
        ml_away=eval_df["ML_Away"].to_numpy(),
        dates=eval_df[DATE_COLUMN],
        home_teams=eval_df["TEAM_NAME"].to_numpy(),
        away_teams=eval_df["TEAM_NAME.1"].to_numpy(),
    )


def run_walk_forward(seasons, trials=20):
    df = load_dataset()
    odds = load_odds_for_seasons(seasons)
    df = attach_odds(df, odds)
    df = df.dropna(subset=[TARGET_ML, TARGET_UO]).reset_index(drop=True)

    for season in seasons:
        if season not in SEASON_BOUNDS:
            print(f"warn: skipping unknown season {season}")
            continue
        lo, hi = SEASON_BOUNDS[season]
        train_df = df[df[DATE_COLUMN] < lo].reset_index(drop=True)
        eval_df = df[(df[DATE_COLUMN] >= lo) & (df[DATE_COLUMN] <= hi)].reset_index(drop=True)
        print_header(f"WALK-FORWARD: training on {len(train_df)} rows < {lo}, evaluating on {len(eval_df)} rows in {season}")
        if train_df.empty or eval_df.empty:
            print("skipping (empty)")
            continue

        # Train ML
        X_train_ml, y_train_ml = split_features_labels(train_df, TARGET_ML, DROP_COLUMNS_ML)
        X_eval_ml, y_eval_ml = split_features_labels(eval_df, TARGET_ML, DROP_COLUMNS_ML)
        booster_ml, calib_ml = train_xgb_quick(X_train_ml, y_train_ml, num_classes=NUM_CLASSES_ML, trials=trials)

        # Train UO
        X_train_uo, y_train_uo = split_features_labels(train_df, TARGET_UO, DROP_COLUMNS_UO)
        X_eval_uo, y_eval_uo = split_features_labels(eval_df, TARGET_UO, DROP_COLUMNS_UO)
        booster_uo, calib_uo = train_xgb_quick(X_train_uo, y_train_uo, num_classes=NUM_CLASSES_UO, trials=trials)

        probs_ml = predict_probs(booster_ml, calib_ml, X_eval_ml)
        probs_uo = predict_probs(booster_uo, calib_uo, X_eval_uo)

        report_run(
            f"walk-forward, season={season}",
            y_ml=y_eval_ml,
            probs_ml=np.asarray(probs_ml),
            y_uo=y_eval_uo,
            probs_uo=np.asarray(probs_uo),
            ml_home=eval_df["ML_Home"].to_numpy(),
            ml_away=eval_df["ML_Away"].to_numpy(),
            dates=eval_df[DATE_COLUMN],
            home_teams=eval_df["TEAM_NAME"].to_numpy(),
            away_teams=eval_df["TEAM_NAME.1"].to_numpy(),
        )


def main():
    parser = argparse.ArgumentParser(description="Backtest XGBoost ML/UO models.")
    parser.add_argument("--mode", choices=["test-split", "walk-forward"], default="test-split")
    parser.add_argument("--seasons", default="2024-25,2025-26", help="Comma-separated season keys.")
    parser.add_argument("--trials", type=int, default=20, help="Walk-forward trials per model.")
    args = parser.parse_args()

    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    if args.mode == "test-split":
        run_test_split(seasons)
    else:
        run_walk_forward(seasons, trials=args.trials)


if __name__ == "__main__":
    main()
