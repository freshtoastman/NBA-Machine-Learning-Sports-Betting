"""Train CatBoost ATS model — evaluates same way as XGBoost_Model_ATS.py.

Loads same 175-feature dataset, does walk-forward CV, evaluates Oct-Feb 2025-26
at edge≥8% (production benchmark: 76.6%, 47 picks from XGBoost model).

Usage:
  PYTHONPATH=. python scripts/train_catboost_ats.py --trials 60 --seed 42
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
DATASET_DB = BASE_DIR / "Data" / "dataset.sqlite"
ODDS_DB = BASE_DIR / "Data" / "OddsData.sqlite"
MODEL_DIR = BASE_DIR / "Models" / "XGBoost_Models"

SEASON_KEYS = [
    "2007-08", "2008-09", "2009-10", "2010-11", "2011-12",
    "2012-13", "2013-14", "2014-15", "2015-16", "2016-17",
    "2017-18", "2018-19", "2019-20", "2020-21", "2021-22",
    "2022-23", "2023-24", "2024-25", "2025-26",
]

# Same drop columns as XGBoost trainer (175-feature model)
DROP_COLUMNS = [
    "index", "Score", "Home-Team-Win", "TEAM_NAME", "Date",
    "index.1", "TEAM_NAME.1", "Date.1", "OU-Cover", "OU",
    "Win_Margin", "ATS_Cover",
    "Home", "Away", "Points", "ML_Home", "ML_Away",
    "H_game_num_season", "A_game_num_season", "D_game_num_season",
    "H_month_sin", "A_month_sin", "D_month_sin",
    "H_month_cos", "A_month_cos", "D_month_cos",
    "H_form_ats_pct_home_10", "A_form_ats_pct_home_10", "D_form_ats_pct_home_10",
    "H_form_ats_pct_away_10", "A_form_ats_pct_away_10", "D_form_ats_pct_away_10",
    "H_form_pts_for_5", "A_form_pts_for_5", "D_form_pts_for_5",
    "H_form_pts_against_5", "A_form_pts_against_5", "D_form_pts_against_5",
    "H_form_pts_diff_10", "A_form_pts_diff_10", "D_form_pts_diff_10",
]


def _table_exists(con, name):
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _freshest_odds_table(con, season_key):
    candidates = [f"odds_{season_key}_new", f"odds_{season_key}", f"{season_key}_new", season_key]
    existing = [t for t in candidates if _table_exists(con, t)]
    if not existing:
        return None
    if len(existing) == 1:
        return existing[0]
    return max(existing, key=lambda t: con.execute(f'SELECT MAX(Date) FROM "{t}"').fetchone()[0] or "")


def load_dataset():
    with sqlite3.connect(DATASET_DB) as con:
        df = pd.read_sql_query('SELECT * FROM "dataset_2012-26"', con)

    frames = []
    with sqlite3.connect(ODDS_DB) as con:
        for season in SEASON_KEYS:
            tbl = _freshest_odds_table(con, season)
            if not tbl:
                continue
            try:
                o = pd.read_sql_query(
                    f'SELECT Date, Home, Away, Spread, Win_Margin, Points, ML_Home, ML_Away FROM "{tbl}"', con)
                frames.append(o)
            except Exception:
                continue

    odds = pd.concat(frames, ignore_index=True)
    odds = odds.drop_duplicates(subset=["Date", "Home", "Away"], keep="last")
    for col in ("Spread", "Win_Margin", "Points", "ML_Home", "ML_Away"):
        odds[col] = pd.to_numeric(odds[col], errors="coerce")
    df = df.merge(odds, how="left", left_on=["Date", "TEAM_NAME", "TEAM_NAME.1"],
                  right_on=["Date", "Home", "Away"])
    df = df.dropna(subset=["Spread", "Win_Margin"]).reset_index(drop=True)
    df = df[df["Points"].fillna(0) > 0].reset_index(drop=True)
    df["ATS_Cover"] = ((df["Win_Margin"] - df["Spread"]) > 0).astype(int)

    try:
        from src.Utils.AdvancedFeatures import merge_into
        df = merge_into(df, home_col="TEAM_NAME", away_col="TEAM_NAME.1", date_col="Date")
        df = df.dropna(subset=["H_form_w_pct_10", "A_form_w_pct_10"]).reset_index(drop=True)
        print(f"Advanced features merged: {len(df)} rows")
    except Exception as exc:
        print(f"warn: advanced features: {exc}")

    try:
        from src.Utils.PlayoffContext import is_playoff_date, build_playoff_features, get_series_state
        pf_rows = []
        for _, row in df.iterrows():
            date_str = str(row["Date"])[:10]
            if is_playoff_date(date_str):
                state = get_series_state(row["TEAM_NAME"], row["TEAM_NAME.1"], date_str)
                pf_rows.append(build_playoff_features(state))
            else:
                pf_rows.append(build_playoff_features(None))
        pf_df = pd.DataFrame(pf_rows)
        for col in pf_df.columns:
            df[col] = pf_df[col].values
        print(f"Playoff features merged: {int(df['is_playoff'].sum())} playoff games")
    except Exception as exc:
        print(f"warn: playoff features: {exc}")

    return df


def symmetric_augment(df: pd.DataFrame) -> pd.DataFrame:
    flipped = df.copy()
    suffix = ".1"
    home_cols = [c for c in df.columns if not c.endswith(suffix) and f"{c}{suffix}" in df.columns]
    for col in home_cols:
        away_col = f"{col}{suffix}"
        tmp = flipped[col].copy()
        flipped[col] = flipped[away_col]
        flipped[away_col] = tmp
    if "Days-Rest-Home" in flipped.columns and "Days-Rest-Away" in flipped.columns:
        tmp = flipped["Days-Rest-Home"].copy()
        flipped["Days-Rest-Home"] = flipped["Days-Rest-Away"]
        flipped["Days-Rest-Away"] = tmp
    if "Spread" in flipped.columns:
        flipped["Spread"] = -flipped["Spread"]
    if "Win_Margin" in flipped.columns:
        flipped["Win_Margin"] = -flipped["Win_Margin"]
    if "Home-Team-Win" in flipped.columns:
        flipped["Home-Team-Win"] = 1 - flipped["Home-Team-Win"]
    if "ATS_Cover" in flipped.columns:
        flipped["ATS_Cover"] = 1 - flipped["ATS_Cover"]
    h_a_pairs = []
    for col in flipped.columns:
        if col.startswith("H_"):
            a_col = f"A_{col[2:]}"
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
    combined["_date_sort"] = pd.to_datetime(combined["Date"], errors="coerce")
    combined = combined.sort_values("_date_sort").drop(columns=["_date_sort"]).reset_index(drop=True)
    return combined


def prepare_data(df):
    data = df.copy()
    data["_date_sort"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.sort_values("_date_sort").drop(columns=["_date_sort"])
    y = data["ATS_Cover"].astype(int).to_numpy()
    X = data.drop(columns=DROP_COLUMNS, errors="ignore").astype(float).to_numpy()
    return X, y


def evaluate_ats_precision(model, df_raw, label=""):
    """Evaluate Oct-Feb 2025-26 ATS precision at edge >= 8%."""
    try:
        df = df_raw.copy()
        df["_date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["_month"] = df["_date"].dt.month
        # Oct-Feb 2025-26: Date >= 2025-10-01 and month in [10,11,12,1,2]
        mask_oct_feb = (
            (df["_date"] >= "2025-10-01") &
            (df["_month"].isin([10, 11, 12, 1, 2]))
        )
        df_eval = df[mask_oct_feb].copy()
        if len(df_eval) == 0:
            return 0, 0, 0, 0
        X_eval = df_eval.drop(columns=DROP_COLUMNS + ["_date", "_month"], errors="ignore").astype(float).to_numpy()
        y_eval = df_eval["ATS_Cover"].astype(int).to_numpy()
        spreads = df_eval["Spread"].to_numpy()

        probs = model.predict_proba(X_eval)
        prob_home = probs[:, 1]
        prob_away = 1 - prob_home

        edge_threshold = 0.08
        value_mask = (prob_home >= 0.5 + edge_threshold) | (prob_away >= 0.5 + edge_threshold)
        n_picks = value_mask.sum()
        if n_picks == 0:
            return 0, 0, 0, 0

        pred_cover = (prob_home >= 0.5 + edge_threshold).astype(int)
        # For away bets: pred_cover=0 is correct when y=0
        correct = np.where(
            prob_home >= 0.5 + edge_threshold,
            (y_eval == 1).astype(int),
            (y_eval == 0).astype(int)
        )
        correct_picks = (correct * value_mask).sum()
        precision = correct_picks / n_picks if n_picks > 0 else 0

        # Same for 2024-25
        mask_2425 = (
            (df["_date"] >= "2024-10-01") & (df["_date"] < "2025-04-01") &
            (df["_month"].isin([10, 11, 12, 1, 2]))
        )
        df_2425 = df[mask_2425].copy()
        if len(df_2425) == 0:
            return precision, n_picks, 0, 0
        X_2425 = df_2425.drop(columns=DROP_COLUMNS + ["_date", "_month"], errors="ignore").astype(float).to_numpy()
        y_2425 = df_2425["ATS_Cover"].astype(int).to_numpy()
        p2 = model.predict_proba(X_2425)
        ph2 = p2[:, 1]
        pa2 = 1 - ph2
        vm2 = (ph2 >= 0.5 + edge_threshold) | (pa2 >= 0.5 + edge_threshold)
        n2 = vm2.sum()
        if n2 == 0:
            return precision, n_picks, 0, 0
        c2 = np.where(ph2[vm2] >= 0.5 + edge_threshold, (y_2425[vm2] == 1).astype(int), (y_2425[vm2] == 0).astype(int))
        prec2 = c2.sum() / n2

        return precision, n_picks, prec2, n2
    except Exception as exc:
        print(f"  eval error: {exc}")
        return 0, 0, 0, 0


def sample_params(rng, seed):
    return {
        "depth": int(rng.integers(3, 9)),
        "learning_rate": float(10 ** rng.uniform(np.log10(0.005), np.log10(0.15))),
        "l2_leaf_reg": float(10 ** rng.uniform(np.log10(0.5), np.log10(30.0))),
        "min_data_in_leaf": int(rng.integers(15, 100)),
        "iterations": int(rng.integers(400, 2001)),
        "border_count": int(rng.integers(64, 512)),
        "random_seed": seed,
        "verbose": False,
        "allow_writing_files": False,
        "task_type": "CPU",
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
    }


def walk_forward_cv_loss(X, y, params, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    losses = []
    for train_idx, val_idx in tscv.split(X):
        m = CatBoostClassifier(**params)
        m.fit(X[train_idx], y[train_idx],
              eval_set=Pool(X[val_idx], y[val_idx]),
              early_stopping_rounds=50,
              verbose=False)
        probs = m.predict_proba(X[val_idx])
        loss = log_loss(y[val_idx], probs)
        losses.append(loss)
    return float(np.mean(losses)) if losses else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--splits", type=int, default=5)
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    df = load_dataset()
    print(f"Loaded {len(df)} rows, ATS balance: {df['ATS_Cover'].mean():.4f}")

    df_aug = symmetric_augment(df)
    print(f"After augmentation: {len(df_aug)} rows")

    X, y = prepare_data(df_aug)
    print(f"Feature shape: {X.shape}")

    n = len(X)
    test_start = int(n * 0.9)
    X_tv, y_tv = X[:test_start], y[:test_start]
    X_test, y_test = X[test_start:], y[test_start:]

    rng = np.random.default_rng(args.seed)
    best = {"val_loss": float("inf"), "params": None}

    for trial in range(1, args.trials + 1):
        params = sample_params(rng, seed=args.seed + trial)
        val_loss = walk_forward_cv_loss(X_tv, y_tv, params, args.splits)
        if val_loss is None:
            continue
        if val_loss < best["val_loss"]:
            best["val_loss"] = val_loss
            best["params"] = params.copy()
        print(f"Trial {trial}/{args.trials}: val_loss={val_loss:.4f}  "
              f"depth={params['depth']} lr={params['learning_rate']:.4f} "
              f"min_leaf={params['min_data_in_leaf']} l2={params['l2_leaf_reg']:.2f} "
              f"iters={params['iterations']}")

    if best["params"] is None:
        print("No valid parameters found.")
        return

    print(f"\nBest val_loss: {best['val_loss']:.4f}")
    print(f"Best params: {best['params']}")

    # Train final model on train_val
    final_model = CatBoostClassifier(**best["params"])
    calib_start = int(len(X_tv) * 0.9)
    X_tr, y_tr = X_tv[:calib_start], y_tv[:calib_start]
    X_cal, y_cal = X_tv[calib_start:], y_tv[calib_start:]
    final_model.fit(X_tr, y_tr, eval_set=Pool(X_cal, y_cal),
                    early_stopping_rounds=50, verbose=False)

    test_probs = final_model.predict_proba(X_test)
    test_acc = accuracy_score(y_test, np.argmax(test_probs, axis=1))
    test_loss = log_loss(y_test, test_probs)
    print(f"Test accuracy: {test_acc:.4f}, Test log_loss: {test_loss:.4f}")

    # OOS ATS precision evaluation
    print("\nEvaluating Oct-Feb OOS ATS precision...")
    prec_2526, n_2526, prec_2425, n_2425 = evaluate_ats_precision(final_model, df)
    print(f"2025-26 Oct-Feb edge≥8%: {prec_2526:.1%} ({n_2526} picks)")
    print(f"2024-25 Oct-Feb edge≥8%: {prec_2425:.1%} ({n_2425} picks)")
    if n_2526 > 0 and n_2425 > 0:
        combined = (prec_2526 * n_2526 + prec_2425 * n_2425) / (n_2526 + n_2425)
        print(f"Combined: {combined:.1%} ({n_2526+n_2425} picks)")

    # Save if good enough (>= production 76.6%)
    if prec_2526 >= 0.766 and n_2526 >= 20:
        model_name = (
            f"CatBoost_{test_acc*100:.1f}%_ATS"
            f"_d{best['params']['depth']}"
            f"_lr{best['params']['learning_rate']:.4f}"
            f"_mdl{best['params']['min_data_in_leaf']}"
            f"_l2{best['params']['l2_leaf_reg']:.2f}"
            f"_i{best['params']['iterations']}.cbm"
        )
        model_path = MODEL_DIR / model_name
        final_model.save_model(str(model_path))
        print(f"\nSAVED NEW MODEL: {model_path}")
        print(f"Beats production XGBoost! ({prec_2526:.1%} vs 76.6%)")
    else:
        print(f"\nModel REJECTED: {prec_2526:.1%} < 76.6% production benchmark")

    return final_model, prec_2526, n_2526, prec_2425, n_2425


if __name__ == "__main__":
    main()
