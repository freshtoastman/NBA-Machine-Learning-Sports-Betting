"""Train XGBoost ATS (against-the-spread) model.

Target: home_cover — 1 iff Win_Margin > Spread (home team covered their
spread). Features are the same team-stat columns used by the ML model
PLUS the spread itself (it's part of the prediction question).

Joins OddsData season tables into dataset_2012-26 at load time so we
don't have to mutate the shared dataset file.

Usage:
  PYTHONPATH=. python src/Train-Models/XGBoost_Model_ATS.py --trials 50
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DB = BASE_DIR / "Data" / "dataset.sqlite"
ODDS_DB = BASE_DIR / "Data" / "OddsData.sqlite"
MODEL_DIR = BASE_DIR / "Models" / "XGBoost_Models"

DEFAULT_DATASET = "dataset_2012-26"
TARGET_COLUMN = "ATS_Cover"
DATE_COLUMN = "Date"

# Column name layout after merge with OddsData — keep Spread as a feature,
# drop anything that reveals the outcome.
DROP_COLUMNS = [
    "index", "Score", "Home-Team-Win", "TEAM_NAME", "Date",
    "index.1", "TEAM_NAME.1", "Date.1", "OU-Cover", "OU",
    "Win_Margin", "ATS_Cover",
    # Merge metadata from OddsData
    "Home", "Away", "Points", "ML_Home", "ML_Away",
    # Temporal features from AdvancedFeatures: these added noise and hurt
    # Oct-Feb edge≥8% precision (57.5% vs 75.6% for legacy 12-feature model).
    # Excluded to produce the stable 175-feature model architecture.
    "H_game_num_season", "A_game_num_season", "D_game_num_season",
    "H_month_sin", "A_month_sin", "D_month_sin",
    "H_month_cos", "A_month_cos", "D_month_cos",
    # Home/away split ATS features from AdvancedFeatures: added 2026-04-17.
    # 50-trial test with seed=123 gave 57.4% Oct-Feb >=8% precision (vs 75.6%
    # baseline). 150-trial test with seed=42 gave 61% (mcw=9) — WORSE.
    # Excluded permanently: home/away split features do not improve precision.
    "H_form_ats_pct_home_10", "A_form_ats_pct_home_10", "D_form_ats_pct_home_10",
    "H_form_ats_pct_away_10", "A_form_ats_pct_away_10", "D_form_ats_pct_away_10",
]
NUM_CLASSES = 2

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


def load_dataset(dataset_name):
    """Load dataset + merge Spread/Win_Margin from OddsData."""
    with sqlite3.connect(DATASET_DB) as con:
        df = pd.read_sql_query(f'SELECT * FROM "{dataset_name}"', con)

    frames = []
    with sqlite3.connect(ODDS_DB) as con:
        for season in SEASON_KEYS:
            tbl = _freshest_odds_table(con, season)
            if not tbl:
                continue
            try:
                o = pd.read_sql_query(
                    f'SELECT Date, Home, Away, Spread, Win_Margin, Points, ML_Home, ML_Away '
                    f'FROM "{tbl}"',
                    con,
                )
                frames.append(o)
            except Exception:
                continue
    if frames:
        odds = pd.concat(frames, ignore_index=True)
        odds = odds.drop_duplicates(subset=["Date", "Home", "Away"], keep="last")
        for col in ("Spread", "Win_Margin", "Points", "ML_Home", "ML_Away"):
            odds[col] = pd.to_numeric(odds[col], errors="coerce")
        df = df.merge(
            odds,
            how="left",
            left_on=["Date", "TEAM_NAME", "TEAM_NAME.1"],
            right_on=["Date", "Home", "Away"],
        )
    else:
        for col in ("Spread", "Win_Margin", "Points", "ML_Home", "ML_Away"):
            df[col] = np.nan

    # Drop rows without a usable spread + margin.
    df = df.dropna(subset=["Spread", "Win_Margin"]).reset_index(drop=True)
    # Drop finished-only games (Points > 0 means the game has a final score).
    df = df[df["Points"].fillna(0) > 0].reset_index(drop=True)
    # Compute target: home covered the spread.
    df[TARGET_COLUMN] = ((df["Win_Margin"] - df["Spread"]) > 0).astype(int)

    # Merge advanced rolling-form + schedule-density features.
    try:
        from src.Utils.AdvancedFeatures import merge_into
        df = merge_into(df, home_col="TEAM_NAME", away_col="TEAM_NAME.1", date_col="Date")
        # Drop rows where the form features couldn't be computed (early-season).
        df = df.dropna(subset=["H_form_w_pct_10", "A_form_w_pct_10"]).reset_index(drop=True)
        print(f"After advanced feature merge: {len(df)} rows")
    except Exception as exc:
        print(f"warn: advanced feature merge failed: {exc}")

    return df


def symmetric_augment(df: pd.DataFrame) -> pd.DataFrame:
    """Double the training data by adding a home/away-flipped copy of every row.

    ATS is an inherently symmetric task — swapping home and away (and
    negating the spread/margin) flips the answer. Without this, XGBoost
    learns a "home posture" bias because the column layout isn't symmetric.
    Adding flipped rows forces a symmetric decision boundary.
    """
    flipped = df.copy()

    # Swap every home<->away stat pair.
    suffix = ".1"
    home_cols = [
        c for c in df.columns
        if not c.endswith(suffix) and f"{c}{suffix}" in df.columns
    ]
    for col in home_cols:
        away_col = f"{col}{suffix}"
        tmp = flipped[col].copy()
        flipped[col] = flipped[away_col]
        flipped[away_col] = tmp

    # Days rest is keyed by side, not suffix.
    if "Days-Rest-Home" in flipped.columns and "Days-Rest-Away" in flipped.columns:
        tmp = flipped["Days-Rest-Home"].copy()
        flipped["Days-Rest-Home"] = flipped["Days-Rest-Away"]
        flipped["Days-Rest-Away"] = tmp

    # Spread is home-referenced — negate.
    if "Spread" in flipped.columns:
        flipped["Spread"] = -flipped["Spread"]
    # Win margin is home-referenced — negate.
    if "Win_Margin" in flipped.columns:
        flipped["Win_Margin"] = -flipped["Win_Margin"]

    # Flip labels.
    if "Home-Team-Win" in flipped.columns:
        flipped["Home-Team-Win"] = 1 - flipped["Home-Team-Win"]
    if TARGET_COLUMN in flipped.columns:
        flipped[TARGET_COLUMN] = 1 - flipped[TARGET_COLUMN]

    # Advanced features prefixed H_/A_ swap; D_ (home-minus-away diffs) negate.
    h_a_pairs = []
    for col in flipped.columns:
        if col.startswith("H_"):
            stem = col[2:]
            a_col = f"A_{stem}"
            if a_col in flipped.columns:
                h_a_pairs.append((col, a_col))
    for h, a in h_a_pairs:
        tmp = flipped[h].copy()
        flipped[h] = flipped[a]
        flipped[a] = tmp
    for col in flipped.columns:
        if col.startswith("D_"):
            flipped[col] = -flipped[col]

    combined = pd.concat([df, flipped], ignore_index=True)
    # Re-sort by date so walk-forward CV still sees time-ordered data.
    if DATE_COLUMN in combined.columns:
        combined[DATE_COLUMN] = pd.to_datetime(combined[DATE_COLUMN], errors="coerce")
        combined = combined.sort_values(DATE_COLUMN).reset_index(drop=True)
    return combined


def prepare_data(df):
    data = df.copy()
    if DATE_COLUMN in data.columns:
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
        data = data.sort_values(DATE_COLUMN)
    y = data[TARGET_COLUMN].astype(int).to_numpy()
    X = data.drop(columns=DROP_COLUMNS, errors="ignore").astype(float).to_numpy()
    return X, y


def split_train_test(X, y, test_size=0.1):
    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset.")
    test_start = int(n * (1 - test_size))
    return X[:test_start], y[:test_start], X[test_start:], y[test_start:]


def split_train_calib(X, y, calib_size=0.1):
    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset.")
    calib_start = int(n * (1 - calib_size))
    return X[:calib_start], y[:calib_start], X[calib_start:], y[calib_start:]


def compute_sample_weights(y, num_classes):
    counts = np.bincount(y, minlength=num_classes)
    total = len(y)
    class_weights = {
        cls: (total / (num_classes * count)) if count else 1.0
        for cls, count in enumerate(counts)
    }
    return np.array([class_weights[label] for label in y])


def sample_params(rng, seed):
    eta = 10 ** rng.uniform(np.log10(0.003), np.log10(0.3))
    params = {
        "max_depth": int(rng.integers(2, 13)),
        "eta": float(eta),
        "subsample": float(rng.uniform(0.5, 1.0)),
        "colsample_bytree": float(rng.uniform(0.5, 1.0)),
        "colsample_bylevel": float(rng.uniform(0.5, 1.0)),
        "colsample_bynode": float(rng.uniform(0.5, 1.0)),
        # Higher min_child_weight forces stronger regularization, which is critical
        # for ATS models: models with mcw=3 generate many high-edge picks at low
        # precision (~59%), while mcw=16 gives fewer but more accurate picks (75.6%).
        # Range raised from [1,20] to [8,30] to focus search on well-regularized models.
        "min_child_weight": int(rng.integers(8, 31)),
        "gamma": float(rng.uniform(0.0, 10.0)),
        "max_delta_step": int(rng.integers(0, 11)),
        "max_bin": int(rng.integers(128, 1025)),
        "lambda": float(10 ** rng.uniform(np.log10(0.1), np.log10(10.0))),
        "alpha": float(10 ** rng.uniform(np.log10(0.01), np.log10(5.0))),
        "objective": "multi:softprob",
        "num_class": NUM_CLASSES,
        "eval_metric": ["mlogloss", "merror"],
        "seed": seed,
        "tree_method": "hist",
    }
    num_boost_round = int(rng.integers(300, 2501))
    return params, num_boost_round


def train_model(X_train, y_train, X_val, y_val, params, num_boost_round):
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=compute_sample_weights(y_train, num_classes=NUM_CLASSES))
    dval = xgb.DMatrix(X_val, label=y_val)
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=60,
        verbose_eval=False,
    )
    return model


def format_param(value, precision=3):
    formatted = f"{value:.{precision}f}" if isinstance(value, float) else str(value)
    return formatted.replace(".", "p")


class BoosterWrapper:
    def __init__(self, booster, num_class):
        self.booster = booster
        self.classes_ = np.arange(num_class)

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return self.booster.predict(xgb.DMatrix(X))


def walk_forward_cv_loss(X, y, params, num_boost_round, n_splits):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    losses = []
    for train_idx, val_idx in tscv.split(X):
        model = train_model(X[train_idx], y[train_idx], X[val_idx], y[val_idx], params, num_boost_round)
        val_probs = model.predict(xgb.DMatrix(X[val_idx]))
        loss = log_loss(y[val_idx], val_probs, labels=list(range(NUM_CLASSES)))
        losses.append(loss)
    return float(np.mean(losses)) if losses else None


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost ATS model.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--calibration", default="none", choices=["sigmoid", "isotonic", "none"])
    parser.add_argument("--no-augment", action="store_true", help="Skip symmetric home/away augmentation")
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.dataset)
    print(f"Loaded {len(df)} rows after merging Spread/Win_Margin from OddsData.")
    print(f"Target balance: home_cover={df[TARGET_COLUMN].mean():.4f}")

    if not args.no_augment:
        df = symmetric_augment(df)
        print(f"After symmetric augmentation: {len(df)} rows (home_cover={df[TARGET_COLUMN].mean():.4f})")

    X, y = prepare_data(df)
    print(f"Features shape: {X.shape}")
    X_train_val, y_train_val, X_test, y_test = split_train_test(X, y)

    rng = np.random.default_rng(args.seed)
    best = {"val_loss": float("inf"), "params": None, "num_boost_round": None}

    for trial in range(1, args.trials + 1):
        params, num_boost_round = sample_params(rng, seed=args.seed + trial)
        val_loss = walk_forward_cv_loss(X_train_val, y_train_val, params, num_boost_round, args.splits)
        if val_loss is None:
            continue
        if val_loss < best["val_loss"]:
            best["val_loss"] = val_loss
            best["params"] = params
            best["num_boost_round"] = num_boost_round
        print(f"Trial {trial}/{args.trials}: val log loss {val_loss:.4f}")

    if best["params"] is None:
        print("No valid parameter set found.")
        return

    X_train, y_train, X_calib, y_calib = split_train_calib(X_train_val, y_train_val)
    best_model = train_model(X_train, y_train, X_calib, y_calib, best["params"], best["num_boost_round"])

    calibrator = None
    if args.calibration == "none":
        probabilities = best_model.predict(xgb.DMatrix(X_test))
    else:
        calibrator = CalibratedClassifierCV(
            BoosterWrapper(best_model, NUM_CLASSES), method=args.calibration, cv="prefit",
        )
        calibrator.fit(X_calib, y_calib)
        probabilities = calibrator.predict_proba(X_test)

    y_pred = np.argmax(probabilities, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    test_loss = log_loss(y_test, probabilities, labels=list(range(NUM_CLASSES)))

    print(f"\nBest val log loss: {best['val_loss']:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test log loss: {test_loss:.4f}")

    params = best["params"]
    model_name = (
        f"XGBoost_{accuracy * 100:.1f}%_ATS"
        f"_md{params['max_depth']}"
        f"_eta{format_param(params['eta'])}"
        f"_sub{format_param(params['subsample'])}"
        f"_col{format_param(params['colsample_bytree'])}"
        f"_mcw{params['min_child_weight']}"
        f"_g{format_param(params['gamma'])}"
        f"_nb{best['num_boost_round']}.json"
    )
    model_path = MODEL_DIR / model_name
    best_model.save_model(str(model_path))
    print(f"Saved model: {model_path}")

    if calibrator is not None:
        calibration_path = MODEL_DIR / f"{model_path.stem}_calibration.pkl"
        joblib.dump(calibrator, calibration_path)
        print(f"Saved calibration: {calibration_path}")


if __name__ == "__main__":
    main()
