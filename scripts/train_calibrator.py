"""Train an isotonic calibrator for the current ML booster.

Uses the 2025-26 season as out-of-sample calibration data (the active model
was trained on data through the prior season, so 2025-26 is genuinely held
out).

Saves the resulting `IsotonicCalibrator` to the same directory as the booster
with filename `{booster_stem}_calibration.pkl`, which `_load_calibrator` in
XGBoost_Runner picks up automatically.

Usage:
    PYTHONPATH=. python scripts/train_calibrator.py [season_key ...]
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.Predict.XGBoost_Runner import _select_model_path
from src.Utils.Calibration import IsotonicCalibrator
from src.Utils.SeasonStats import (
    DATASET_DB, ODDS_DB, DATASET_TABLE, SEASON_BOUNDS,
    DROP_FEATURE_COLS, _freshest_odds_table,
)

DEFAULT_SEASONS = ["2025-26"]


def load_raw_probs(season_key: str, booster: xgb.Booster):
    lo, hi = SEASON_BOUNDS[season_key]
    with sqlite3.connect(DATASET_DB) as con:
        df = pd.read_sql_query(
            f'SELECT * FROM "{DATASET_TABLE}" WHERE Date >= ? AND Date <= ?',
            con, params=(lo, hi),
        )
    df = df[df["Score"].fillna(0) > 0].reset_index(drop=True)
    if df.empty:
        return None, None

    with sqlite3.connect(ODDS_DB) as con:
        ot = _freshest_odds_table(con, season_key)
        if ot:
            odds = pd.read_sql_query(
                f'SELECT Date, Home, Away, Spread FROM "{ot}"', con,
            )
            odds["Date"] = odds["Date"].astype(str)
            df = df.merge(odds, how="left",
                          left_on=["Date", "TEAM_NAME", "TEAM_NAME.1"],
                          right_on=["Date", "Home", "Away"])

    frame = df.drop(columns=DROP_FEATURE_COLS, errors="ignore").copy()
    frame = frame.drop(
        columns=["Home", "Away", "ML_Home", "ML_Away", "Spread", "Win_Margin"],
        errors="ignore",
    )
    if "OU" in frame.columns:
        frame = frame.drop(columns=["OU"])
    X = frame.astype(float).to_numpy()
    try:
        n = int(booster.num_features())
        if X.shape[1] > n:
            X = X[:, :n]
    except Exception:
        pass

    raw = np.asarray(booster.predict(xgb.DMatrix(X)))
    y = df["Home-Team-Win"].astype(int).to_numpy()
    return raw, y


def main():
    seasons = sys.argv[1:] or DEFAULT_SEASONS
    ml_path = _select_model_path("ML")
    print(f"Model: {ml_path.name}")

    booster = xgb.Booster()
    booster.load_model(str(ml_path))

    all_p = []
    all_y = []
    for s in seasons:
        raw, y = load_raw_probs(s, booster)
        if raw is None:
            print(f"  {s}: no data, skipped")
            continue
        print(f"  {s}: {len(y)} games")
        all_p.append(raw[:, 1])
        all_y.append(y)

    if not all_p:
        raise SystemExit("No calibration data loaded.")

    p_home = np.concatenate(all_p)
    y_home = np.concatenate(all_y)
    print(f"\nTotal calibration games: {len(p_home)}")
    print(f"Raw Brier: {np.mean((p_home - y_home) ** 2):.4f}")

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_home, y_home)
    p_cal = iso.transform(p_home)
    print(f"Calibrated Brier: {np.mean((p_cal - y_home) ** 2):.4f}")

    # Sanity check: gap by decile
    print(f"\n{'bucket':<10}{'n':>6}{'raw':>9}{'cal':>9}{'actual':>9}")
    for lo_ in np.arange(0.0, 1.0, 0.1):
        hi_ = lo_ + 0.1
        mask = (p_home >= lo_) & (p_home < hi_ + (0.0001 if abs(lo_ - 0.9) < 1e-9 else 0))
        n = int(mask.sum())
        if n == 0:
            continue
        print(f"{lo_:.1f}-{hi_:.1f}   {n:>6}{p_home[mask].mean():>9.3f}"
              f"{p_cal[mask].mean():>9.3f}{y_home[mask].mean():>9.3f}")

    cal = IsotonicCalibrator(iso)
    out_path = ml_path.with_name(f"{ml_path.stem}_calibration.pkl")
    joblib.dump(cal, out_path)
    print(f"\nSaved → {out_path.name}")


if __name__ == "__main__":
    main()
