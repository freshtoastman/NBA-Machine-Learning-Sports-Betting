"""Strict out-of-sample backtest + strategy sweep.

Trains XGBoost ML on data strictly BEFORE 2024-10-01, then evaluates on
2024-25 + 2025-26. Sweeps value-finder thresholds and reports the best
ROI / win-rate / EV combinations for both moneyline and ATS strategies.

Usage:
  PYTHONPATH=. python scripts/strict_backtest.py
  PYTHONPATH=. python scripts/strict_backtest.py --trials 30 --cutoff 2024-10-01
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

# Reuse helpers from the existing backtest script.
sys.path.insert(0, str(Path(__file__).parent))
from backtest import (
    DATASET_DB,
    DROP_COLUMNS_ML,
    NUM_CLASSES_ML,
    TARGET_ML,
    DATE_COLUMN,
    load_dataset,
    load_odds_for_seasons,
    attach_odds,
    split_features_labels,
    train_xgb_quick,
    _BoosterWrapper,
)
from src.Utils.ValueFinder import (
    evaluate_value,
    american_to_decimal,
    american_to_implied,
    remove_vig,
    expected_value,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_eval_data(cutoff_iso: str, eval_seasons):
    """Return (train_X, train_y, eval_df) where eval_df has features + odds + labels."""
    df = load_dataset()
    df = df.dropna(subset=[TARGET_ML]).reset_index(drop=True)

    # Attach money lines + spread + win margin from OddsData.
    odds = load_odds_for_seasons(eval_seasons + ["2007-08","2008-09","2009-10","2010-11","2011-12","2012-13","2013-14","2014-15","2015-16","2016-17","2017-18","2018-19","2019-20","2020-21","2021-22","2022-23","2023-24"])
    df = attach_odds(df, odds)

    # Splits.
    cutoff = pd.Timestamp(cutoff_iso)
    train_df = df[df[DATE_COLUMN] < cutoff].reset_index(drop=True)
    eval_df = df[df[DATE_COLUMN] >= cutoff].reset_index(drop=True)

    print(f"Train rows: {len(train_df)} (< {cutoff_iso})")
    print(f"Eval rows:  {len(eval_df)} (>= {cutoff_iso})")
    return train_df, eval_df


# ---------------------------------------------------------------------------
# Strategy backtests
# ---------------------------------------------------------------------------

STAKE = 1000.0
ATS_PROFIT = 900.0  # -110 juice


def _features_only(eval_df):
    """Drop merged-in odds/team-name columns before feeding to the booster."""
    drop = ["Home", "Away", "ML_Home", "ML_Away", "Spread", "Points", "Win_Margin",
            "Date_o", "Home_o", "Away_o"]
    return eval_df.drop(columns=drop, errors="ignore")


def backtest_moneyline(probs, eval_df, min_prob, edge_pp, max_edge_pp=None):
    """Bet $1 flat on every value pick at the moneyline."""
    bets = wins = 0
    pl = 0.0
    ml_h = eval_df["ML_Home"].to_numpy()
    ml_a = eval_df["ML_Away"].to_numpy()
    y = eval_df[TARGET_ML].astype(int).to_numpy()

    for i in range(len(probs)):
        v = evaluate_value(
            float(probs[i, 1]), ml_h[i], ml_a[i],
            min_model_prob=min_prob, edge_threshold_pp=edge_pp,
        )
        if not v["is_value"]:
            continue
        if max_edge_pp is not None and v["value_edge"] > max_edge_pp:
            continue
        bets += 1
        actual_home_win = bool(y[i] == 1)
        won = (v["value_side"] == "home" and actual_home_win) or (
            v["value_side"] == "away" and not actual_home_win
        )
        if won:
            wins += 1
            odds_used = ml_h[i] if v["value_side"] == "home" else ml_a[i]
            dec = american_to_decimal(odds_used)
            if dec is not None:
                pl += dec - 1.0
        else:
            pl -= 1.0
    return bets, wins, pl


def backtest_ats(probs, eval_df, min_prob, edge_pp, max_edge_pp=None):
    """Bet $1000 ATS at -110 on every value pick. Uses spread from OddsData."""
    bets = wins = pushes = perfect = 0
    pl = 0.0
    ml_h = eval_df["ML_Home"].to_numpy()
    ml_a = eval_df["ML_Away"].to_numpy()
    spreads = eval_df["Spread"].to_numpy()
    margins = eval_df["Win_Margin"].to_numpy()
    y = eval_df[TARGET_ML].astype(int).to_numpy()

    for i in range(len(probs)):
        v = evaluate_value(
            float(probs[i, 1]), ml_h[i], ml_a[i],
            min_model_prob=min_prob, edge_threshold_pp=edge_pp,
        )
        if not v["is_value"]:
            continue
        if max_edge_pp is not None and v["value_edge"] > max_edge_pp:
            continue
        sp = spreads[i]
        mg = margins[i]
        if pd.isna(sp) or pd.isna(mg):
            continue
        home_cover = float(mg) - float(sp)
        if home_cover == 0:
            pushes += 1
            continue
        ats_won = (
            (v["value_side"] == "home" and home_cover > 0)
            or (v["value_side"] == "away" and home_cover < 0)
        )
        bets += 1
        if ats_won:
            wins += 1
            pl += ATS_PROFIT
            actual_home_win = bool(y[i] == 1)
            ml_won = (v["value_side"] == "home" and actual_home_win) or (
                v["value_side"] == "away" and not actual_home_win
            )
            if ml_won:
                perfect += 1
        else:
            pl -= STAKE
    return bets, wins, pushes, perfect, pl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def attach_extra_odds(df, eval_seasons):
    """Re-attach odds keeping Spread + Win_Margin columns (attach_odds drops them)."""
    frames = []
    with sqlite3.connect("Data/OddsData.sqlite") as con:
        for season in eval_seasons:
            try:
                f = pd.read_sql_query(f'SELECT * FROM "{season}"', con)
                f["Date"] = pd.to_datetime(f["Date"], errors="coerce")
                frames.append(f)
            except Exception:
                continue
    if not frames:
        return df
    odds = pd.concat(frames, ignore_index=True)
    for col in ("ML_Home", "ML_Away", "Spread", "Points", "Win_Margin"):
        if col in odds.columns:
            odds[col] = pd.to_numeric(odds[col], errors="coerce")
    odds = odds[["Date", "Home", "Away", "ML_Home", "ML_Away", "Spread", "Points", "Win_Margin"]]
    df = df.drop(columns=["ML_Home", "ML_Away", "Spread", "Win_Margin"], errors="ignore")
    df = df.merge(
        odds,
        how="left",
        left_on=[DATE_COLUMN, "TEAM_NAME", "TEAM_NAME.1"],
        right_on=["Date", "Home", "Away"],
        suffixes=("", "_o"),
    )
    return df


def main():
    parser = argparse.ArgumentParser(description="Strict OOS backtest + strategy sweep.")
    parser.add_argument("--cutoff", default="2024-10-01")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    eval_seasons = ["2024-25", "2025-26"]
    train_df, eval_df = load_eval_data(args.cutoff, eval_seasons)

    # We need spread/margin in eval_df. attach_odds in load_eval_data may have
    # dropped them via the helper's column projection — re-attach explicitly.
    eval_df = attach_extra_odds(eval_df, eval_seasons)
    # Drop rows missing money lines (can't compute value).
    eval_df = eval_df.dropna(subset=["ML_Home", "ML_Away"]).reset_index(drop=True)
    print(f"Eval rows with odds: {len(eval_df)}")

    print(f"\n=== Training XGBoost on {len(train_df)} rows (trials={args.trials}) ===")
    X_train, y_train = split_features_labels(train_df, TARGET_ML, DROP_COLUMNS_ML)
    booster, calibrator = train_xgb_quick(
        X_train, y_train, num_classes=NUM_CLASSES_ML, trials=args.trials, seed=args.seed
    )

    print("\n=== Predicting on eval set ===")
    X_eval, y_eval = split_features_labels(_features_only(eval_df), TARGET_ML, DROP_COLUMNS_ML)
    if calibrator is not None:
        probs = np.asarray(calibrator.predict_proba(X_eval))
    else:
        probs = np.asarray(booster.predict(xgb.DMatrix(X_eval)))

    raw_acc = (np.argmax(probs, axis=1) == y_eval).mean()
    print(f"Raw ML accuracy on OOS: {raw_acc:.4f}")

    print("\n=== Strategy sweep — Moneyline (flat $1) ===")
    print(f"  {'min_p':>6}  {'edge':>5}  {'bets':>5}  {'wins':>5}  {'win%':>6}  {'P/L':>9}  {'ROI':>7}")
    best_ml = None
    for min_prob in [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65]:
        for edge in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0]:
            bets, wins, pl = backtest_moneyline(probs, eval_df, min_prob, edge)
            if bets < 30:
                continue
            roi = pl / bets * 100
            wr = wins / bets * 100
            row = (min_prob, edge, bets, wins, wr, pl, roi)
            print(f"  {min_prob:>6.2f}  {edge:>5.1f}  {bets:>5d}  {wins:>5d}  {wr:>5.1f}%  ${pl:>+8.2f}  {roi:>+6.2f}%")
            if best_ml is None or roi > best_ml[6]:
                best_ml = row

    print("\n=== Strategy sweep — ATS ($1000 @ -110) ===")
    print(f"  {'min_p':>6}  {'edge':>5}  {'bets':>5}  {'wins':>5}  {'win%':>6}  {'P/L':>11}  {'ROI':>7}")
    best_ats = None
    for min_prob in [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65]:
        for edge in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0]:
            bets, wins, pushes, perfect, pl = backtest_ats(probs, eval_df, min_prob, edge)
            if bets < 30:
                continue
            roi = pl / (bets * STAKE) * 100
            wr = wins / bets * 100
            row = (min_prob, edge, bets, wins, wr, pl, roi, pushes, perfect)
            print(f"  {min_prob:>6.2f}  {edge:>5.1f}  {bets:>5d}  {wins:>5d}  {wr:>5.1f}%  ${pl:>+10.0f}  {roi:>+6.2f}%")
            if best_ats is None or roi > best_ats[6]:
                best_ats = row

    print("\n=== Best Moneyline strategy ===")
    if best_ml:
        mp, e, b, w, wr, pl, roi = best_ml
        print(f"  min_prob >= {mp}, edge >= {e}pp")
        print(f"  bets={b}, wins={w} ({wr:.1f}%), P/L=${pl:+.2f}, ROI={roi:+.2f}%")

    print("\n=== Best ATS strategy ===")
    if best_ats:
        mp, e, b, w, wr, pl, roi, pu, pf = best_ats
        print(f"  min_prob >= {mp}, edge >= {e}pp")
        print(f"  bets={b}, wins={w} ({wr:.1f}%), pushes={pu}, perfect={pf}")
        print(f"  Total stake: ${b*1000:,}, P/L: ${pl:+,.0f}, ROI: {roi:+.2f}%")

    # Test edge cap variants on the best strategies
    if best_ats:
        mp, e, *_ = best_ats
        print(f"\n=== ATS edge caps (min_p={mp}, edge>={e}) ===")
        print(f"  {'cap':>5}  {'bets':>5}  {'wins':>5}  {'win%':>6}  {'P/L':>11}  {'ROI':>7}")
        for cap in [None, 25.0, 20.0, 15.0, 12.0, 10.0]:
            bets, wins, pushes, perfect, pl = backtest_ats(probs, eval_df, mp, e, max_edge_pp=cap)
            if bets < 20:
                continue
            roi = pl / (bets * STAKE) * 100
            wr = wins / bets * 100
            cap_str = f"{cap:>4.1f}" if cap is not None else " inf"
            print(f"  {cap_str}  {bets:>5d}  {wins:>5d}  {wr:>5.1f}%  ${pl:>+10.0f}  {roi:>+6.2f}%")


if __name__ == "__main__":
    main()
