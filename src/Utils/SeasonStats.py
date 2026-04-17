"""Season-wide model performance: ML / UO / value-pick rollups.

Loads every graded game in a season from dataset.sqlite + OddsData.sqlite,
runs the current XGBoost models on them in one batch, and returns
aggregate hit rates and value-pick ROI.

Cached by (season_key, dataset.sqlite mtime) so it auto-invalidates after
the daily ETL refresh.
"""

from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.Predict.XGBoost_Runner import (
    _select_model_path,
    _load_calibrator,
    _build_frame_uo,
    _build_frame_ats,
)
from src.Utils.ValueFinder import evaluate_value, american_to_decimal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DB = PROJECT_ROOT / "Data" / "dataset.sqlite"
ODDS_DB = PROJECT_ROOT / "Data" / "OddsData.sqlite"
DATASET_TABLE = "dataset_2012-26"

DROP_FEATURE_COLS = [
    "index", "Score", "Home-Team-Win", "TEAM_NAME", "Date",
    "index.1", "TEAM_NAME.1", "Date.1", "OU-Cover",
]

SEASON_BOUNDS = {
    "2007-08": ("2007-10-01", "2008-09-30"),
    "2008-09": ("2008-10-01", "2009-09-30"),
    "2009-10": ("2009-10-01", "2010-09-30"),
    "2010-11": ("2010-10-01", "2011-09-30"),
    "2011-12": ("2011-10-01", "2012-09-30"),
    "2012-13": ("2012-10-01", "2013-09-30"),
    "2013-14": ("2013-10-01", "2014-09-30"),
    "2014-15": ("2014-10-01", "2015-09-30"),
    "2015-16": ("2015-10-01", "2016-09-30"),
    "2016-17": ("2016-10-01", "2017-09-30"),
    "2017-18": ("2017-10-01", "2018-09-30"),
    "2018-19": ("2018-10-01", "2019-09-30"),
    "2019-20": ("2019-10-01", "2020-09-30"),
    "2020-21": ("2020-10-01", "2021-09-30"),
    "2021-22": ("2021-10-01", "2022-09-30"),
    "2022-23": ("2022-10-01", "2023-09-30"),
    "2023-24": ("2023-10-01", "2024-09-30"),
    "2024-25": ("2024-10-01", "2025-09-30"),
    "2025-26": ("2025-10-01", "2026-09-30"),
}


def _freshest_odds_table(con, season_key):
    candidates = [
        f"odds_{season_key}_new",
        f"odds_{season_key}",
        f"{season_key}_new",
        season_key,
    ]
    existing = []
    for t in candidates:
        if con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (t,)
        ).fetchone():
            existing.append(t)
    if not existing:
        return None
    if len(existing) == 1:
        return existing[0]
    return max(
        existing,
        key=lambda t: con.execute(f'SELECT MAX(Date) FROM "{t}"').fetchone()[0] or "",
    )


def _dataset_fingerprint():
    try:
        return DATASET_DB.stat().st_mtime
    except FileNotFoundError:
        return 0.0


@lru_cache(maxsize=16)
def _compute(season_key: str, _fingerprint: float):
    bounds = SEASON_BOUNDS.get(season_key)
    if not bounds:
        return None
    lo, hi = bounds

    with sqlite3.connect(DATASET_DB) as con:
        df = pd.read_sql_query(
            f'SELECT * FROM "{DATASET_TABLE}" WHERE Date >= ? AND Date <= ?',
            con,
            params=(lo, hi),
        )
    if df.empty:
        return None
    # Only graded games (Score > 0 i.e. game finished and scored).
    df = df[df["Score"].fillna(0) > 0].reset_index(drop=True)
    if df.empty:
        return None

    # Pull money lines + spread + win margin from OddsData.
    with sqlite3.connect(ODDS_DB) as con:
        odds_table = _freshest_odds_table(con, season_key)
        if odds_table:
            odds = pd.read_sql_query(
                f'SELECT Date, Home, Away, ML_Home, ML_Away, Spread, Win_Margin '
                f'FROM "{odds_table}"',
                con,
            )
            odds["Date"] = odds["Date"].astype(str)
            for col in ("ML_Home", "ML_Away", "Spread", "Win_Margin"):
                odds[col] = pd.to_numeric(odds[col], errors="coerce")
            df = df.merge(
                odds,
                how="left",
                left_on=["Date", "TEAM_NAME", "TEAM_NAME.1"],
                right_on=["Date", "Home", "Away"],
            )
        else:
            for col in ("ML_Home", "ML_Away", "Spread", "Win_Margin"):
                df[col] = np.nan

    # Build the same feature frames the runners would build.
    frame_keep_ou = df.drop(columns=DROP_FEATURE_COLS, errors="ignore").copy()
    frame_keep_ou = frame_keep_ou.drop(
        columns=["Home", "Away", "ML_Home", "ML_Away", "Spread", "Win_Margin"],
        errors="ignore",
    )
    if "OU" in frame_keep_ou.columns:
        frame_for_ml = frame_keep_ou.drop(columns=["OU"])
    else:
        frame_for_ml = frame_keep_ou
    X_ml = frame_for_ml.astype(float).to_numpy()

    def _trim(model, X):
        """Backward-compat: trim trailing feature cols if dataset has more
        cols than the loaded model (e.g. is_playoff was added after training)."""
        try:
            n = int(model.num_features())
        except Exception:
            n = X.shape[1]
        return X[:, :n] if X.shape[1] > n else X

    # Load + run ML model.
    ml_path = _select_model_path("ML")
    booster_ml = xgb.Booster()
    booster_ml.load_model(str(ml_path))
    calib_ml = _load_calibrator(ml_path)
    X_ml_aligned = _trim(booster_ml, X_ml)
    probs_ml = np.asarray(booster_ml.predict(xgb.DMatrix(X_ml_aligned)))
    if calib_ml is not None:
        probs_ml = np.asarray(calib_ml.predict_proba(probs_ml))

    # UO model with proper column position via _build_frame_uo.
    uo_path = _select_model_path("UO")
    booster_uo = xgb.Booster()
    booster_uo.load_model(str(uo_path))
    ou_array = frame_keep_ou["OU"].astype(float).to_numpy() if "OU" in frame_keep_ou.columns else None
    frame_uo = _build_frame_uo(frame_for_ml, ou_array if ou_array is not None else [])
    X_uo = frame_uo.astype(float).to_numpy()
    probs_uo = np.asarray(booster_uo.predict(xgb.DMatrix(_trim(booster_uo, X_uo))))

    # ATS model — optional, may not exist yet.
    probs_ats = None
    spread_array = df["Spread"].to_numpy() if "Spread" in df.columns else None
    if spread_array is not None and not np.all(pd.isna(spread_array)):
        try:
            ats_path = _select_model_path("ATS")
            booster_ats = xgb.Booster()
            booster_ats.load_model(str(ats_path))
            safe_spreads = np.where(pd.isna(spread_array), 0.0, spread_array.astype(float))

            # Try to attach advanced rolling features (post-Apr 2026 ATS model
            # was trained with these). Falls back to legacy if shape mismatch.
            advanced_df = None
            try:
                from src.Utils.AdvancedFeatures import merge_into as _merge_adv
                helper = df[["TEAM_NAME", "TEAM_NAME.1", "Date"]].copy()
                helper = _merge_adv(helper, "TEAM_NAME", "TEAM_NAME.1", "Date")
                cols = [c for c in helper.columns if c.startswith(("H_", "A_", "D_"))]
                if cols:
                    advanced_df = helper[cols].fillna(0.0).reset_index(drop=True)
            except Exception:
                advanced_df = None

            frame_ats = _build_frame_ats(frame_for_ml, safe_spreads, advanced=advanced_df)
            X_ats = frame_ats.astype(float).to_numpy()
            try:
                probs_ats = np.asarray(booster_ats.predict(xgb.DMatrix(X_ats)))
            except Exception:
                # Shape mismatch (model expects different feature count) — try
                # without advanced features as a fallback.
                frame_ats = _build_frame_ats(frame_for_ml, safe_spreads)
                X_ats = frame_ats.astype(float).to_numpy()
                probs_ats = np.asarray(booster_ats.predict(xgb.DMatrix(X_ats)))
        except FileNotFoundError:
            probs_ats = None

    y_ml = df["Home-Team-Win"].astype(int).to_numpy()
    y_uo = df["OU-Cover"].astype(int).to_numpy()
    pred_ml = np.argmax(probs_ml, axis=1)
    pred_uo = np.argmax(probs_uo, axis=1)

    ml_hits = int((pred_ml == y_ml).sum())
    ou_hits = int((pred_uo == y_uo).sum())
    games = len(y_ml)

    # ATS model stats — only on rows with usable spread + win_margin.
    ats_stats = None
    if probs_ats is not None and "Win_Margin" in df.columns:
        wm = df["Win_Margin"].to_numpy()
        sp = spread_array
        valid = ~pd.isna(wm) & ~pd.isna(sp)
        if valid.sum() > 0:
            home_cover_margin = wm[valid] - sp[valid]
            actual_home_cover = (home_cover_margin > 0).astype(int)
            pred_ats = np.argmax(probs_ats[valid], axis=1)
            ats_total = int(valid.sum())
            ats_hits = int((pred_ats == actual_home_cover).sum())
            home_picks = int((pred_ats == 1).sum())

            # ATS value-tier strategy ($1k @ -110, edge ≥threshold).
            # Use the same date-adaptive thresholds as main.py:
            #   Oct-Feb: ≥8%  (reliable window)
            #   March: suppressed (0 picks)
            #   Apr 1-13: suppressed (end-of-season garbage time)
            #   Apr 14+: ≥10% (playoffs/play-in, thin sample)
            import datetime as _dt
            def _ats_threshold(date_str):
                try:
                    d = _dt.date.fromisoformat(date_str)
                    if d.month == 3:
                        return 100.0
                    if d.month == 4 and d.day < 14:
                        return 100.0
                    if d.month == 4:
                        return 10.0
                    return 8.0
                except Exception:
                    return 8.0

            date_col = df["Date"].to_numpy()
            ats_v_bets = ats_v_wins = 0
            ats_v_pl = 0.0
            valid_idx = np.where(valid)[0]
            for j, i in enumerate(valid_idx):
                p_home = float(probs_ats[i, 1])
                edge = abs(p_home - 0.5) * 100
                threshold = _ats_threshold(date_col[i])
                if edge < threshold:
                    continue
                pred_side = 1 if p_home >= 0.5 else 0
                if home_cover_margin[j] == 0:
                    continue  # push
                ats_v_bets += 1
                if pred_side == actual_home_cover[j]:
                    ats_v_wins += 1
                    ats_v_pl += 900.0
                else:
                    ats_v_pl -= 1000.0

            ats_stats = {
                "games": ats_total,
                "hits": ats_hits,
                "hit_rate": round(ats_hits / ats_total * 100, 1),
                "home_pick_pct": round(home_picks / ats_total * 100, 1),
                "value_bets": ats_v_bets,
                "value_wins": ats_v_wins,
                "value_losses": ats_v_bets - ats_v_wins,
                "value_hit_rate": round(ats_v_wins / ats_v_bets * 100, 1) if ats_v_bets else None,
                "value_pl": round(ats_v_pl, 0),
                "value_roi": round(ats_v_pl / (ats_v_bets * 1000) * 100, 1) if ats_v_bets else None,
            }

    # Value picks scan.
    ml_h = df["ML_Home"].to_numpy() if "ML_Home" in df.columns else np.full(games, np.nan)
    ml_a = df["ML_Away"].to_numpy() if "ML_Away" in df.columns else np.full(games, np.nan)
    spreads = df["Spread"].to_numpy() if "Spread" in df.columns else np.full(games, np.nan)
    margins = df["Win_Margin"].to_numpy() if "Win_Margin" in df.columns else np.full(games, np.nan)

    # Standard ATS payout: $1000 stake, win returns $1900 (-110ish juice).
    STAKE = 1000.0
    PAYOUT_WIN = 900.0  # profit per winning $1000 ATS bet

    # Brier + calibration gap on the ML home probability.
    p_home = probs_ml[:, 1]
    brier = float(np.mean((p_home - y_ml) ** 2))
    mid_mask = (p_home >= 0.5) & (p_home < 0.8)
    if mid_mask.sum() > 0:
        calib_gap_mid = float(np.mean(y_ml[mid_mask] - p_home[mid_mask]))
    else:
        calib_gap_mid = None

    n_value = 0
    value_wins = 0
    value_pl = 0.0
    # Golden tier: value pick AND |spread| ≤ 6.
    golden_n = golden_w = 0
    golden_pl = 0.0
    # Combined: diamond pick that ALSO covers the spread (the value side)
    elite_bets = 0       # value picks with valid spread/margin (excludes pushes)
    elite_wins = 0       # value side covered the spread
    elite_pushes = 0
    elite_pl = 0.0
    # Combined: diamond pick AND covers spread AND wins moneyline
    perfect_hits = 0     # ML correct AND ATS covered

    for i in range(games):
        sp_i = spreads[i] if i < len(spreads) else None
        sp_arg = None if pd.isna(sp_i) else float(sp_i)
        v = evaluate_value(float(probs_ml[i, 1]), ml_h[i], ml_a[i], spread=sp_arg)
        if not v["is_value"]:
            continue
        n_value += 1
        actual_home_win = bool(y_ml[i] == 1)
        won_ml = (v["value_side"] == "home" and actual_home_win) or (
            v["value_side"] == "away" and not actual_home_win
        )
        if won_ml:
            value_wins += 1
        odds_used = ml_h[i] if v["value_side"] == "home" else ml_a[i]
        dec = american_to_decimal(odds_used)
        if dec is not None:
            pl_unit = (dec - 1.0) if won_ml else -1.0
            value_pl += pl_unit
            if v.get("is_golden"):
                golden_n += 1
                if won_ml:
                    golden_w += 1
                golden_pl += pl_unit

        # ATS evaluation on the value side.
        sp = spreads[i]
        mg = margins[i]
        if pd.isna(sp) or pd.isna(mg):
            continue
        home_cover_margin = float(mg) - float(sp)
        if home_cover_margin == 0:
            elite_pushes += 1
            continue  # push: no money exchanged
        ats_won = (
            (v["value_side"] == "home" and home_cover_margin > 0)
            or (v["value_side"] == "away" and home_cover_margin < 0)
        )
        elite_bets += 1
        if ats_won:
            elite_wins += 1
            elite_pl += PAYOUT_WIN
            if won_ml:
                perfect_hits += 1
        else:
            elite_pl -= STAKE

    return {
        "season": season_key,
        "games": games,
        "ml_hits": ml_hits,
        "ml_hit_rate": round(ml_hits / games * 100, 1),
        "ou_hits": ou_hits,
        "ou_hit_rate": round(ou_hits / games * 100, 1),
        "value_picks": n_value,
        "value_wins": value_wins,
        "value_losses": n_value - value_wins,
        "value_hit_rate": round(value_wins / n_value * 100, 1) if n_value else None,
        "value_pl": round(value_pl, 2),
        "value_roi": round(value_pl / n_value * 100, 1) if n_value else None,
        # Golden tier = value pick AND |spread| ≤ 6 (backtest sweet spot).
        "golden_picks": golden_n,
        "golden_wins": golden_w,
        "golden_losses": golden_n - golden_w,
        "golden_hit_rate": round(golden_w / golden_n * 100, 1) if golden_n else None,
        "golden_pl": round(golden_pl, 2),
        "golden_roi": round(golden_pl / golden_n * 100, 1) if golden_n else None,
        # Model-quality diagnostics.
        "brier": round(brier, 4),
        "calib_gap_mid": round(calib_gap_mid, 4) if calib_gap_mid is not None else None,
        # Diamond ATS strategy: bet $1000 on value side ATS at -110.
        "elite_bets": elite_bets,
        "elite_wins": elite_wins,
        "elite_losses": elite_bets - elite_wins,
        "elite_pushes": elite_pushes,
        "elite_hit_rate": round(elite_wins / elite_bets * 100, 1) if elite_bets else None,
        "elite_pl": round(elite_pl, 2),
        "elite_roi": round(elite_pl / (elite_bets * STAKE) * 100, 1) if elite_bets else None,
        "elite_total_stake": int(elite_bets * STAKE),
        "perfect_hits": perfect_hits,
        "ats_model": ats_stats,
    }


def compute_season_stats(season_key: str):
    """Public entry — handles fingerprint invalidation."""
    return _compute(season_key, _dataset_fingerprint())


def reset_cache():
    _compute.cache_clear()
