"""Retrain the production ML / ATS boosters on the corrected (pre-game) dataset.

Same recipe that scripts/walk_forward_clean.py validated out-of-sample — the
production hyper-parameters, early stopping on the chronological last 10% of
rows, every tree kept — applied with *all* seasons as the training set. Nothing
is tuned here: the walk-forward run is the only place hyper-parameters or
thresholds may be chosen, so the file name carries the walk-forward
out-of-sample accuracy from web/data/walk_forward.json, not an in-sample score.

Outputs (never auto-pinned; edit config.toml [production-models] to switch):
    Models/XGBoost_Models/XGBoost_<oos>%_ML_clean_<date>_... .json
    Models/XGBoost_Models/XGBoost_<oos>%_ML_clean_<date>_..._calibration.pkl
        (IsotonicCalibrator fitted on the early-stopping tail — mildly optimistic,
        same construction as the `iso` variant in walk_forward_ml_value.py)
    Models/XGBoost_Models/XGBoost_<oos>%_ATS_clean_<date>_... .json
    web/data/production_models.json   provenance + in-sample vs out-of-sample numbers

Usage:
    PYTHONPATH=. python3 scripts/retrain_production.py [--seed 42] [--tag 2026-10-07]
        [--walk-forward web/data/walk_forward.json] [--out web/data/production_models.json]
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import walk_forward_clean as wf  # noqa: E402
from src.Utils.Calibration import IsotonicCalibrator  # noqa: E402

MODEL_DIR = REPO / "Models" / "XGBoost_Models"


def _rec(hits, n):
    return wf.rec(int(hits), int(n))


def refit_all_rows(X, y, mcw, seed, rounds):
    """Stage 2: the tree count chosen by early stopping on the tail, refitted on
    every row (the tail is the newest season, which matters most for the opener)."""
    params = dict(wf.BASE_PARAMS, min_child_weight=mcw, seed=seed, objective="multi:softprob", num_class=2,
                  eval_metric=["mlogloss", "merror"])
    counts = np.bincount(y, minlength=2)
    w = np.array([len(y) / (2 * counts[c]) for c in y])
    booster = xgb.train(params, xgb.DMatrix(X, label=y, weight=w), num_boost_round=rounds, verbose_eval=False)
    return booster, lambda Z: booster.predict(xgb.DMatrix(Z))[:, 1]


def train_ml(ml, ml_df, seed):
    cols = [c for c in ml_df.columns if c not in ml.DROP_COLUMNS + ["season"]]
    X = ml_df[cols].astype(float).to_numpy()
    y = ml_df["Home-Team-Win"].astype(int).to_numpy()
    stage1, rounds, predict1 = wf.fit_classifier(X, y, wf.ML_MCW, seed, "merror")
    cut = int(len(X) * 0.9)
    p_tail = predict1(X[cut:])                                   # held out from stage 1
    tail_acc = _rec(((p_tail >= 0.5).astype(int) == y[cut:]).sum(), len(y) - cut)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_tail, y[cut:])
    booster, predict = refit_all_rows(X, y, wf.ML_MCW, seed, rounds)
    return booster, rounds, predict, cols, cut, IsotonicCalibrator.from_isotonic(iso), iso, tail_acc, p_tail


def train_ats(ats, df, seed):
    aug = ats.symmetric_augment(df).sort_values("Date", kind="stable")
    drop = list(ats.DROP_COLUMNS) + ["season"]
    cols = [c for c in aug.columns if c not in drop]
    X = aug[cols].astype(float).to_numpy()
    y = aug["ATS_Cover"].astype(int).to_numpy()
    stage1, rounds, predict1 = wf.fit_classifier(X, y, wf.ATS_MCW, seed, "merror")
    cut = int(len(X) * 0.9)
    p_tail = predict1(X[cut:])
    tail_acc = _rec(((p_tail >= 0.5).astype(int) == y[cut:]).sum(), len(y) - cut)
    booster, predict = refit_all_rows(X, y, wf.ATS_MCW, seed, rounds)
    return booster, rounds, predict, cols, cut, tail_acc


def ml_name(oos_pct, tag, seed, rounds):
    p = wf.BASE_PARAMS
    return (f"XGBoost_{oos_pct:.1f}%_ML_clean_{tag}"
            f"_md{p['max_depth']}_eta{wf_fmt(p['eta'])}_sub{wf_fmt(p['subsample'])}_col{wf_fmt(p['colsample_bytree'])}"
            f"_cbl{wf_fmt(p['colsample_bylevel'])}_cbn{wf_fmt(p['colsample_bynode'])}_mcw{wf.ML_MCW}"
            f"_g{wf_fmt(p['gamma'])}_mds{p['max_delta_step']}_mb{p['max_bin']}_l{wf_fmt(p['lambda'])}"
            f"_a{wf_fmt(p['alpha'])}_s{seed}_t{rounds}.json")


def ats_name(oos_pct, tag, seed, rounds):
    p = wf.BASE_PARAMS
    return (f"XGBoost_{oos_pct:.1f}%_ATS_clean_{tag}"
            f"_md{p['max_depth']}_eta{wf_fmt(p['eta'])}_sub{wf_fmt(p['subsample'])}_col{wf_fmt(p['colsample_bytree'])}"
            f"_mcw{wf.ATS_MCW}_g{wf_fmt(p['gamma'])}_s{seed}_t{rounds}.json")


def wf_fmt(v):
    return (f"{v:.3f}" if isinstance(v, float) else str(v)).replace(".", "p")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="dataset_2012-26")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=dt.date.today().isoformat())
    ap.add_argument("--walk-forward", default="web/data/walk_forward.json")
    ap.add_argument("--out", default="web/data/production_models.json")
    args = ap.parse_args()

    import os
    os.chdir(REPO)
    walk = json.loads(Path(args.walk_forward).read_text())
    oos_ml = walk["ml"]["model"]
    oos_ats = walk["ats"][walk["ats_recipe"]]["raw"]
    last_season = walk["test_seasons"][-1]

    ats = wf._load("ats_trainer", "src/Train-Models/XGBoost_Model_ATS.py")
    ml = wf._load("ml_trainer", "src/Train-Models/XGBoost_Model_ML.py")

    df = ats.load_dataset(args.dataset)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date", kind="stable").reset_index(drop=True)
    df["season"] = df["Date"].apply(wf.season_of)

    ml_df = ml.load_dataset(args.dataset)
    ml_df["Date"] = pd.to_datetime(ml_df["Date"].astype(str).str[:10])
    ml_df = ml_df[ml_df["Score"].fillna(0) > 0].sort_values("Date", kind="stable").reset_index(drop=True)
    ml_df["season"] = ml_df["Date"].apply(wf.season_of)
    trained_through = max(df["season"].max(), ml_df["season"].max())
    print(f"ML rows {len(ml_df)} | ATS rows {len(df)} | seasons through {trained_through}")

    # ---- moneyline -------------------------------------------------------
    booster, rounds, predict, cols, cut, calibrator, iso, tail_acc, tail_p = train_ml(ml, ml_df, args.seed)
    name = ml_name(oos_ml["pct"], args.tag, args.seed, rounds)
    booster.save_model(str(MODEL_DIR / name))
    joblib.dump(calibrator, MODEL_DIR / f"{Path(name).stem}_calibration.pkl")
    p_all = predict(ml_df[cols].astype(float).to_numpy())
    y_all = ml_df["Home-Team-Win"].astype(int).to_numpy()
    last = (ml_df["season"] == last_season).to_numpy()
    tail_y = y_all[cut:]
    ml_out = {
        "file": name, "kind": "ML", "features": len(cols), "trees": rounds, "seed": args.seed,
        "train_rows": int(len(y_all)), "early_stop_rows": int(len(y_all) - cut),
        "early_stop_from": ml_df["Date"].iloc[cut].strftime("%Y-%m-%d"), "trained_through": trained_through,
        "recipe": "stage 1: early stopping on the chronological last 10% (held out); stage 2: same tree count refitted on all rows",
        "held_out_tail": tail_acc,
        "calibration": "isotonic fitted on the stage-1 model's held-out tail (stored as step points, sklearn-version independent), applied to the refitted booster",
        "tail_brier_raw": round(float(np.mean((tail_p - tail_y) ** 2)), 4),
        "tail_brier_iso": round(float(np.mean((np.clip(iso.transform(tail_p), 1e-6, 1 - 1e-6) - tail_y) ** 2)), 4),
        "in_sample": {last_season: _rec(((p_all[last] >= 0.5).astype(int) == y_all[last]).sum(), last.sum()),
                      "all": _rec(((p_all >= 0.5).astype(int) == y_all).sum(), len(y_all))},
        "out_of_sample": {"pooled": oos_ml, last_season: walk["ml_by_season"][last_season]["model"],
                          "market_favorite_pooled": walk["ml"]["market_favorite"]},
    }
    print(f"ML  {name}\n    trees {rounds} | in-sample {last_season} {wf.fmt(ml_out['in_sample'][last_season])} "
          f"| walk-forward {last_season} {wf.fmt(ml_out['out_of_sample'][last_season])} | pooled OOS {wf.fmt(oos_ml)}")

    # ---- against the spread ------------------------------------------------
    booster, rounds, predict, cols, cut, tail_acc = train_ats(ats, df, args.seed)
    name = ats_name(oos_ats["pct"], args.tag, args.seed, rounds)
    booster.save_model(str(MODEL_DIR / name))
    Xs = df[cols].astype(float).to_numpy()                       # un-augmented, home-referenced rows
    score = predict(Xs)
    cover = (df["Win_Margin"] - df["Spread"]).to_numpy()
    keep = cover != 0
    hit = ((score >= 0.5).astype(int) == (cover > 0).astype(int)) & keep
    last = (df["season"] == last_season).to_numpy()
    ats_out = {
        "file": name, "kind": "ATS", "features": len(cols), "trees": rounds, "seed": args.seed,
        "train_rows": int(2 * len(df)), "early_stop_rows": int(2 * len(df) - cut), "trained_through": trained_through,
        "recipe": "stage 1: early stopping on the chronological last 10% (held out); stage 2: same tree count refitted on all rows",
        "held_out_tail": tail_acc,
        "augmentation": "symmetric home/away copy of every row",
        "in_sample": {last_season: _rec(hit[last].sum(), keep[last].sum()), "all": _rec(hit.sum(), keep.sum())},
        "out_of_sample": {"pooled": oos_ats, last_season: walk["ats_by_season"][last_season]["raw"],
                          "top5_pooled": walk["ats"][walk["ats_recipe"]]["top5"]},
    }
    print(f"ATS {name}\n    trees {rounds} | in-sample {last_season} {wf.fmt(ats_out['in_sample'][last_season])} "
          f"| walk-forward {last_season} {wf.fmt(ats_out['out_of_sample'][last_season])} | pooled OOS {wf.fmt(oos_ats)}")

    out = {
        "generated": args.tag,
        "script": "scripts/retrain_production.py",
        "dataset": args.dataset,
        "walk_forward": {"file": args.walk_forward, "generated": walk.get("generated"),
                         "test_seasons": walk["test_seasons"], "break_even_pct": walk.get("break_even_pct")},
        "note_zh": ("正式模型用修正後（賽前快照、讓分已補正負號）的全部賽季重訓；超參數與 walk-forward 相同，沒有重新挑選。"
                    "檔名上的百分比是 10 季樣本外的命中率，不是訓練季的重算。"),
        "models": {"ml": ml_out, "ats": ats_out},
    }
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
