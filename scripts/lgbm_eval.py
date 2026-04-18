"""
LightGBM ATS evaluation - same 175 features as XGBoost production model.
Evaluates Oct-Feb 2025-26 OOS precision at edge>=8%.
"""
from __future__ import annotations
import sys, sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DB = BASE_DIR / "Data" / "dataset.sqlite"
ODDS_DB = BASE_DIR / "Data" / "OddsData.sqlite"

SEASON_KEYS = [
    "2007-08","2008-09","2009-10","2010-11","2011-12","2012-13","2013-14",
    "2014-15","2015-16","2016-17","2017-18","2018-19","2019-20","2020-21",
    "2021-22","2022-23","2023-24","2024-25","2025-26",
]

DROP_COLUMNS = [
    "index","Score","Home-Team-Win","TEAM_NAME","Date",
    "index.1","TEAM_NAME.1","Date.1","OU-Cover","OU",
    "Win_Margin","ATS_Cover",
    "Home","Away","Points","ML_Home","ML_Away",
    "H_game_num_season","A_game_num_season","D_game_num_season",
    "H_month_sin","A_month_sin","D_month_sin",
    "H_month_cos","A_month_cos","D_month_cos",
    "H_form_ats_pct_home_10","A_form_ats_pct_home_10","D_form_ats_pct_home_10",
    "H_form_ats_pct_away_10","A_form_ats_pct_away_10","D_form_ats_pct_away_10",
    "H_form_pts_for_5","A_form_pts_for_5","D_form_pts_for_5",
    "H_form_pts_against_5","A_form_pts_against_5","D_form_pts_against_5",
    "H_form_pts_diff_10","A_form_pts_diff_10","D_form_pts_diff_10",
]


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
    return max(
        existing,
        key=lambda t: con.execute(f'SELECT MAX(Date) FROM "{t}"').fetchone()[0] or "",
    )


def load_data():
    print("Loading dataset...")
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
                    f'SELECT Date, Home, Away, Spread, Win_Margin, Points, ML_Home, ML_Away FROM "{tbl}"',
                    con,
                )
                frames.append(o)
            except Exception:
                pass

    odds = pd.concat(frames, ignore_index=True)
    odds = odds.drop_duplicates(subset=["Date", "Home", "Away"], keep="last")
    for col in ("Spread", "Win_Margin", "Points", "ML_Home", "ML_Away"):
        odds[col] = pd.to_numeric(odds[col], errors="coerce")
    df = df.merge(
        odds, how="left",
        left_on=["Date", "TEAM_NAME", "TEAM_NAME.1"],
        right_on=["Date", "Home", "Away"],
    )
    df = df.dropna(subset=["Spread", "Win_Margin"]).reset_index(drop=True)
    df = df[df["Points"].fillna(0) > 0].reset_index(drop=True)
    df["ATS_Cover"] = ((df["Win_Margin"] - df["Spread"]) > 0).astype(int)

    from src.Utils.AdvancedFeatures import merge_into
    df = merge_into(df, home_col="TEAM_NAME", away_col="TEAM_NAME.1", date_col="Date")
    df = df.dropna(subset=["H_form_w_pct_10", "A_form_w_pct_10"]).reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    return df


def augment(df_in):
    flipped = df_in.copy()
    suffix = ".1"
    home_cols = [c for c in df_in.columns if not c.endswith(suffix) and f"{c}{suffix}" in df_in.columns]
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
    flipped["ATS_Cover"] = 1 - flipped["ATS_Cover"]
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
    combined = pd.concat([df_in, flipped], ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
    return combined.sort_values("Date").reset_index(drop=True)


def eval_oos(prob_home, y_oos, label="Model"):
    print(f"\n--- {label} OOS Eval (2025-26 Oct-Feb) ---")
    for thresh in [0.08, 0.09, 0.10]:
        mask = np.abs(prob_home - 0.5) >= thresh
        n = mask.sum()
        if n == 0:
            print(f"  edge≥{thresh*100:.0f}%: 0 picks")
            continue
        hits = (prob_home[mask] > 0.5) == y_oos[mask]
        pct = hits.sum() / n
        print(f"  edge≥{thresh*100:.0f}%: {n} picks, {hits.sum()}/{n} = {pct:.1%}")


def main():
    df = load_data()
    df_orig = df.copy()

    # Augment for training
    df_aug = augment(df)
    print(f"Augmented: {len(df_aug)} rows")

    # Prepare feature matrices
    y_aug = df_aug["ATS_Cover"].astype(int).values
    X_aug = df_aug.drop(columns=DROP_COLUMNS, errors="ignore").astype(float).values
    print(f"Feature shape: {X_aug.shape}")

    # Verify feature count matches production (175)
    prod = xgb.Booster()
    prod.load_model(
        str(BASE_DIR / "Models" / "XGBoost_Models" /
            "XGBoost_51.8%_ATS_md4_eta0p033_sub0p657_col0p919_mcw26_g0p721_nb1544.json")
    )
    print(f"XGBoost production features: {prod.num_features()} (target=175)")
    if X_aug.shape[1] != prod.num_features():
        print(f"MISMATCH: got {X_aug.shape[1]} features, expected {prod.num_features()}")
        return

    # 90/10 split
    n = len(X_aug)
    test_start = int(n * 0.9)
    X_train, y_train = X_aug[:test_start], y_aug[:test_start]
    X_test, y_test = X_aug[test_start:], y_aug[test_start:]

    # 2025-26 Oct-Feb OOS data (from original, unaugmented)
    mask_oos = (df_orig["Date"] >= "2025-10-01") & (df_orig["Date"].dt.month.isin([10, 11, 12, 1, 2]))
    df_oos = df_orig[mask_oos].copy()
    y_oos = df_oos["ATS_Cover"].astype(int).values
    X_oos = df_oos.drop(columns=DROP_COLUMNS, errors="ignore").astype(float).values
    print(f"\n2025-26 Oct-Feb: {len(df_oos)} games")

    # XGBoost production baseline
    prob_xgb = prod.predict(xgb.DMatrix(X_oos))[:, 1]
    eval_oos(prob_xgb, y_oos, "XGBoost Production (51.8%, mcw=26)")

    # LightGBM random search
    print("\n=== LightGBM Random Search (40 trials) ===")
    rng = np.random.default_rng(42)
    n_trials = 40
    best = {"val_loss": float("inf"), "params": None, "n_est": None}

    for trial in range(1, n_trials + 1):
        n_est = int(rng.integers(300, 2000))
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": float(10 ** rng.uniform(-2.5, -0.5)),
            "num_leaves": int(rng.integers(8, 128)),
            "min_child_samples": int(rng.integers(20, 300)),
            "subsample": float(rng.uniform(0.5, 1.0)),
            "colsample_bytree": float(rng.uniform(0.5, 1.0)),
            "reg_alpha": float(10 ** rng.uniform(-2, 1)),
            "reg_lambda": float(10 ** rng.uniform(-1, 2)),
            "max_depth": int(rng.integers(3, 10)),
            "verbose": -1,
            "n_jobs": -1,
            "random_state": int(rng.integers(0, 100000)),
        }

        tscv = TimeSeriesSplit(n_splits=5)
        losses = []
        for tr_idx, val_idx in tscv.split(X_train):
            m = lgb.LGBMClassifier(**params, n_estimators=n_est)
            m.fit(
                X_train[tr_idx], y_train[tr_idx],
                eval_set=[(X_train[val_idx], y_train[val_idx])],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
            )
            p = m.predict_proba(X_train[val_idx])
            losses.append(log_loss(y_train[val_idx], p))
        val_loss = float(np.mean(losses))

        if val_loss < best["val_loss"]:
            best["val_loss"] = val_loss
            best["params"] = dict(params)
            best["n_est"] = n_est
            print(
                f"  Trial {trial}: val_loss={val_loss:.4f} NEW BEST "
                f"leaves={params['num_leaves']} min_child={params['min_child_samples']} "
                f"lr={params['learning_rate']:.4f} n_est={n_est}"
            )
        elif trial % 10 == 0:
            print(f"  Trial {trial}: val_loss={val_loss:.4f} (best={best['val_loss']:.4f})")

    print(f"\nBest LightGBM val_loss: {best['val_loss']:.4f}")
    print(f"Best params: {best['params']}")

    # Train final LightGBM on full training set
    print("\nTraining final LightGBM model...")
    final = lgb.LGBMClassifier(**best["params"], n_estimators=best["n_est"])
    final.fit(X_train, y_train, callbacks=[lgb.log_evaluation(-1)])

    # Test set accuracy
    p_test = final.predict_proba(X_test)[:, 1]
    test_acc = ((p_test > 0.5) == y_test).mean()
    print(f"LightGBM test set accuracy: {test_acc:.1%}")

    # OOS evaluation
    p_oos = final.predict_proba(X_oos)[:, 1]
    eval_oos(p_oos, y_oos, f"LightGBM (test_acc={test_acc:.1%})")

    # Also evaluate on 2024-25 Oct-Feb for cross-season check
    mask_2425 = (df_orig["Date"] >= "2024-10-01") & (df_orig["Date"] < "2025-05-01") & df_orig["Date"].dt.month.isin([10,11,12,1,2])
    df_2425 = df_orig[mask_2425].copy()
    if len(df_2425) > 0:
        y_2425 = df_2425["ATS_Cover"].astype(int).values
        X_2425 = df_2425.drop(columns=DROP_COLUMNS, errors="ignore").astype(float).values
        p_2425_lgb = final.predict_proba(X_2425)[:, 1]
        p_2425_xgb = prod.predict(xgb.DMatrix(X_2425))[:, 1]
        print(f"\n--- 2024-25 Oct-Feb ({len(df_2425)} games) ---")
        print("XGBoost production:")
        for thresh in [0.08, 0.09]:
            mask = np.abs(p_2425_xgb - 0.5) >= thresh
            n = mask.sum()
            if n > 0:
                hits = (p_2425_xgb[mask] > 0.5) == y_2425[mask]
                print(f"  edge≥{thresh*100:.0f}%: {n} picks, {hits.sum()}/{n} = {hits.mean():.1%}")
        print("LightGBM:")
        for thresh in [0.08, 0.09]:
            mask = np.abs(p_2425_lgb - 0.5) >= thresh
            n = mask.sum()
            if n > 0:
                hits = (p_2425_lgb[mask] > 0.5) == y_2425[mask]
                print(f"  edge≥{thresh*100:.0f}%: {n} picks, {hits.sum()}/{n} = {hits.mean():.1%}")


if __name__ == "__main__":
    main()
