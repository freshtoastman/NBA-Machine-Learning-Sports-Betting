"""Alternative validation strategies backtest.

Runs three diagnostics that don't require changing the odds pipeline:
  1. Brier score + calibration by probability bucket
  2. EV-tiered ROI (per-side expected value)
  3. Spread-bucketed value-pick ROI

Usage:
    PYTHONPATH=. python scripts/backtest_strategies.py [season_key]
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.Predict.XGBoost_Runner import _select_model_path, _load_calibrator
from src.Utils.SeasonStats import (
    DATASET_DB, ODDS_DB, DATASET_TABLE, SEASON_BOUNDS,
    DROP_FEATURE_COLS, _freshest_odds_table,
)
from src.Utils.ValueFinder import evaluate_value, american_to_decimal


def load_predictions(season_key: str):
    lo, hi = SEASON_BOUNDS[season_key]
    with sqlite3.connect(DATASET_DB) as con:
        df = pd.read_sql_query(
            f'SELECT * FROM "{DATASET_TABLE}" WHERE Date >= ? AND Date <= ?',
            con, params=(lo, hi),
        )
    df = df[df["Score"].fillna(0) > 0].reset_index(drop=True)
    if df.empty:
        raise SystemExit(f"No graded games for season {season_key}")

    with sqlite3.connect(ODDS_DB) as con:
        ot = _freshest_odds_table(con, season_key)
        if ot:
            odds = pd.read_sql_query(
                f'SELECT Date, Home, Away, ML_Home, ML_Away, Spread, Win_Margin FROM "{ot}"',
                con,
            )
            odds["Date"] = odds["Date"].astype(str)
            for c in ("ML_Home", "ML_Away", "Spread", "Win_Margin"):
                odds[c] = pd.to_numeric(odds[c], errors="coerce")
            df = df.merge(
                odds, how="left",
                left_on=["Date", "TEAM_NAME", "TEAM_NAME.1"],
                right_on=["Date", "Home", "Away"],
            )

    frame = df.drop(columns=DROP_FEATURE_COLS, errors="ignore").copy()
    frame = frame.drop(
        columns=["Home", "Away", "ML_Home", "ML_Away", "Spread", "Win_Margin"],
        errors="ignore",
    )
    if "OU" in frame.columns:
        frame = frame.drop(columns=["OU"])
    X = frame.astype(float).to_numpy()

    ml_path = _select_model_path("ML")
    booster = xgb.Booster()
    booster.load_model(str(ml_path))
    try:
        n = int(booster.num_features())
        if X.shape[1] > n:
            X = X[:, :n]
    except Exception:
        pass
    cal = _load_calibrator(ml_path)
    probs = np.asarray(booster.predict(xgb.DMatrix(X)))
    if cal is not None:
        probs = np.asarray(cal.predict_proba(probs))

    y = df["Home-Team-Win"].astype(int).to_numpy()
    return df, probs, y


# ---------------------------------------------------------------------------
# Strategy 1: Brier score + reliability diagram
# ---------------------------------------------------------------------------

def brier_and_calibration(probs, y):
    p_home = probs[:, 1]
    brier = float(np.mean((p_home - y) ** 2))
    buckets = []
    for lo in np.arange(0.0, 1.0, 0.1):
        hi = lo + 0.1
        if abs(lo - 0.9) < 1e-9:
            mask = (p_home >= lo) & (p_home <= hi)
        else:
            mask = (p_home >= lo) & (p_home < hi)
        n = int(mask.sum())
        if n == 0:
            buckets.append((lo, hi, 0, None, None))
        else:
            buckets.append((lo, hi, n, float(p_home[mask].mean()), float(y[mask].mean())))
    return brier, buckets


# ---------------------------------------------------------------------------
# Strategy 2: EV-tiered ROI
# ---------------------------------------------------------------------------

def ev_tiers(probs, y, ml_h, ml_a):
    tiers = [(-100, 0), (0, 2), (2, 5), (5, 10), (10, 500)]
    bets = {t: {"n": 0, "w": 0, "pl": 0.0} for t in tiers}
    p_home = probs[:, 1]
    for i in range(len(p_home)):
        for side, p, odds in (
            ("home", p_home[i], ml_h[i]),
            ("away", 1 - p_home[i], ml_a[i]),
        ):
            dec = american_to_decimal(odds)
            if dec is None:
                continue
            ev_pct = (p * (dec - 1) - (1 - p)) * 100
            won = (side == "home" and y[i] == 1) or (side == "away" and y[i] == 0)
            for t in tiers:
                if t[0] <= ev_pct < t[1]:
                    bets[t]["n"] += 1
                    if won:
                        bets[t]["w"] += 1
                        bets[t]["pl"] += (dec - 1)
                    else:
                        bets[t]["pl"] -= 1
                    break
    return tiers, bets


# ---------------------------------------------------------------------------
# Strategy 3: Spread-bucketed value-pick ROI
# ---------------------------------------------------------------------------

def spread_buckets(probs, y, spreads, ml_h, ml_a):
    buckets = [(0, 3), (3, 6), (6, 10), (10, 100)]
    out = {b: {"n": 0, "w": 0, "pl": 0.0} for b in buckets}
    for i in range(len(probs)):
        sp = spreads[i]
        if pd.isna(sp):
            continue
        v = evaluate_value(float(probs[i, 1]), ml_h[i], ml_a[i])
        if not v["is_value"]:
            continue
        actual_home = bool(y[i] == 1)
        won = (v["value_side"] == "home" and actual_home) or (
            v["value_side"] == "away" and not actual_home
        )
        odds = ml_h[i] if v["value_side"] == "home" else ml_a[i]
        dec = american_to_decimal(odds)
        if dec is None:
            continue
        abs_sp = abs(float(sp))
        for b in buckets:
            if b[0] <= abs_sp < b[1]:
                out[b]["n"] += 1
                if won:
                    out[b]["w"] += 1
                    out[b]["pl"] += (dec - 1)
                else:
                    out[b]["pl"] -= 1
                break
    return buckets, out


def golden_vs_silver(probs, y, spreads, ml_h, ml_a):
    """Compare picks flagged golden (|spread| ≤ 6) vs silver (rest)."""
    out = {
        "golden": {"n": 0, "w": 0, "pl": 0.0},
        "silver": {"n": 0, "w": 0, "pl": 0.0},
    }
    for i in range(len(probs)):
        sp = spreads[i]
        sp_arg = None if pd.isna(sp) else float(sp)
        v = evaluate_value(float(probs[i, 1]), ml_h[i], ml_a[i], spread=sp_arg)
        if not v["is_value"]:
            continue
        actual_home = bool(y[i] == 1)
        won = (v["value_side"] == "home" and actual_home) or (
            v["value_side"] == "away" and not actual_home
        )
        odds = ml_h[i] if v["value_side"] == "home" else ml_a[i]
        dec = american_to_decimal(odds)
        if dec is None:
            continue
        tier = "golden" if v.get("is_golden") else "silver"
        out[tier]["n"] += 1
        if won:
            out[tier]["w"] += 1
            out[tier]["pl"] += (dec - 1)
        else:
            out[tier]["pl"] -= 1
    return out


# ---------------------------------------------------------------------------

def main():
    season = sys.argv[1] if len(sys.argv) > 1 else "2025-26"
    print(f"\n=== Backtest Strategies — season {season} ===\n")
    df, probs, y = load_predictions(season)
    print(f"Games loaded: {len(y)}")

    print("\n--- #1 Brier score & calibration ---")
    brier, buckets = brier_and_calibration(probs, y)
    base = 0.25  # flat 50% predictor
    print(f"Brier: {brier:.4f}  (base=0.25, lower is better)")
    print(f"{'bucket':<10}{'n':>6}{'pred':>9}{'actual':>9}{'gap':>9}")
    for lo, hi, n, pred, actual in buckets:
        if n == 0:
            print(f"{lo:.1f}-{hi:.1f}   {n:>6}")
        else:
            gap = actual - pred
            print(f"{lo:.1f}-{hi:.1f}   {n:>6}{pred:>9.3f}{actual:>9.3f}{gap:>+9.3f}")

    print("\n--- #2 EV-tiered ROI (both sides of every game) ---")
    ml_h = df["ML_Home"].to_numpy() if "ML_Home" in df.columns else np.full(len(y), np.nan)
    ml_a = df["ML_Away"].to_numpy() if "ML_Away" in df.columns else np.full(len(y), np.nan)
    tiers, bets = ev_tiers(probs, y, ml_h, ml_a)
    print(f"{'EV tier':<14}{'n':>6}{'wins':>6}{'hit%':>8}{'roi%':>10}")
    for t in tiers:
        b = bets[t]
        hit = 100 * b["w"] / b["n"] if b["n"] else 0
        roi = 100 * b["pl"] / b["n"] if b["n"] else 0
        label = f"<0%" if t[0] < 0 else f"{t[0]}-{t[1]}%" if t[1] < 100 else f"{t[0]}+%"
        print(f"{label:<14}{b['n']:>6}{b['w']:>6}{hit:>8.1f}{roi:>+10.2f}")

    spreads = df["Spread"].to_numpy() if "Spread" in df.columns else np.full(len(y), np.nan)

    print("\n--- #3a Golden vs Silver value picks ---")
    gs = golden_vs_silver(probs, y, spreads, ml_h, ml_a)
    for tier in ("golden", "silver"):
        r = gs[tier]
        hit = 100 * r["w"] / r["n"] if r["n"] else 0
        roi = 100 * r["pl"] / r["n"] if r["n"] else 0
        label = "🥇 GOLDEN (|sp|≤6)" if tier == "golden" else "🥈 silver (|sp|>6)"
        print(f"{label:<22}{r['n']:>6}{r['w']:>6}{hit:>8.1f}{roi:>+10.2f}")

    print("\n--- #3 Spread-bucketed value-pick ROI ---")
    sbuckets, sout = spread_buckets(probs, y, spreads, ml_h, ml_a)
    print(f"{'|spread|':<10}{'n':>6}{'wins':>6}{'hit%':>8}{'roi%':>10}")
    for b in sbuckets:
        r = sout[b]
        hit = 100 * r["w"] / r["n"] if r["n"] else 0
        roi = 100 * r["pl"] / r["n"] if r["n"] else 0
        label = f"{b[0]}-{b[1]}" if b[1] < 100 else f"{b[0]}+"
        print(f"{label:<10}{r['n']:>6}{r['w']:>6}{hit:>8.1f}{roi:>+10.2f}")
    print()


if __name__ == "__main__":
    main()
