"""Walk-forward hyper-parameter search for the ATS classifier.

Every configuration is scored season by season with a model trained only on
earlier seasons (same protocol as scripts/walk_forward_clean.py, `noadv` columns,
early stopping on log loss). To keep the search honest the configurations are
ranked on the SELECTION seasons only; the winner is then read off the HOLDOUT
seasons it never influenced. A config that only looks good on the seasons used
to choose it is noise, not an improvement.

Usage: PYTHONPATH=. python scripts/tune_ats_walk_forward.py [--json web/data/ats_tuning.json]
"""
import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("wf", REPO / "scripts" / "walk_forward_clean.py")
wf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wf)

PROD = dict(wf.BASE_PARAMS, min_child_weight=wf.ATS_MCW)
CONFIGS = {
    "prod": PROD,
    "shallow_d2": dict(PROD, max_depth=2),
    "shallow_d3": dict(PROD, max_depth=3),
    "deep_d6": dict(PROD, max_depth=6),
    "slow_eta": dict(PROD, eta=0.01),
    "fast_eta": dict(PROD, eta=0.1),
    "mcw_5": dict(PROD, min_child_weight=5),
    "mcw_100": dict(PROD, min_child_weight=100),
    "heavy_reg": dict(PROD, max_depth=3, min_child_weight=100, gamma=5.0, **{"lambda": 10.0, "alpha": 2.0}),
    "stumps_reg": dict(PROD, max_depth=1, eta=0.05, min_child_weight=200, **{"lambda": 20.0}),
    "subsample_low": dict(PROD, subsample=0.4, colsample_bytree=0.4),
    "d3_mcw60_slow": dict(PROD, max_depth=3, min_child_weight=60, eta=0.02),
}


def fit_predict(params, X, y, Xtest, seed):
    p = dict(params, seed=seed, objective="binary:logistic", eval_metric="logloss")
    p.pop("max_delta_step", None)
    Xt, yt, Xv, yv = wf.split_tail(X, y)
    b = xgb.train(p, xgb.DMatrix(Xt, label=yt), num_boost_round=wf.NUM_ROUNDS,
                  evals=[(xgb.DMatrix(Xv, label=yv), "val")], early_stopping_rounds=wf.PATIENCE, verbose_eval=False)
    r = b.best_iteration + 1
    return b.predict(xgb.DMatrix(Xtest), iteration_range=(0, r)), r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset_2012-26")
    ap.add_argument("--first-test", default="2016-17")
    ap.add_argument("--holdout-from", default="2023-24")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", default="web/data/ats_tuning.json")
    args = ap.parse_args()
    os.chdir(REPO)
    ats = wf._load("ats_trainer", "src/Train-Models/XGBoost_Model_ATS.py")
    df = ats.load_dataset(args.dataset)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date", kind="stable").reset_index(drop=True)
    df["season"] = df["Date"].apply(wf.season_of)
    tests = [s for s in sorted(df["season"].unique()) if args.first_test <= s <= "2025-26"]
    rows = []
    for s in tests:
        train, test = df[df.season < s], df[df.season == s].reset_index(drop=True)
        X, y, Xtest, _ = wf.ats_matrices(ats, train, test, "noadv")
        cover = (test["Win_Margin"] - test["Spread"]).to_numpy()
        keep = cover != 0
        ytest = (cover > 0).astype(int)
        for name, params in CONFIGS.items():
            p, rounds = fit_predict(params, X, y, Xtest, args.seed)
            pc = np.clip(p, 1e-6, 1 - 1e-6)
            ll = -(ytest * np.log(pc) + (1 - ytest) * np.log(1 - pc))[keep]
            hit = ((p >= 0.5).astype(int) == ytest)[keep]
            conf = (np.abs(p - 0.5) >= 0.05) & keep
            rows.append({"season": s, "config": name, "n": int(keep.sum()), "hits": int(hit.sum()),
                         "logloss_sum": float(ll.sum()), "rounds": rounds,
                         "conf_n": int(conf.sum()), "conf_hits": int(((p >= 0.5).astype(int) == ytest)[conf].sum())})
        print(s, "done", flush=True)
    r = pd.DataFrame(rows)
    r["part"] = np.where(r["season"] >= args.holdout_from, "holdout", "selection")

    def agg(g):
        n, h = g["n"].sum(), g["hits"].sum()
        lo, hi = wf.wilson(int(h), int(n)) if n else (None, None)
        cn, ch = g["conf_n"].sum(), g["conf_hits"].sum()
        return {"n": int(n), "hit_rate": round(h / n * 100, 2), "ci95": [round(lo * 100, 1) if lo <= 1 else lo, round(hi * 100, 1) if hi <= 1 else hi],
                "logloss": round(g["logloss_sum"].sum() / n, 4),
                "edge5_n": int(cn), "edge5_hit_rate": round(ch / cn * 100, 1) if cn else None}

    out = {"protocol": f"walk-forward, tests {tests[0]}..{tests[-1]}, selection < {args.holdout_from} <= holdout, seed {args.seed}, coin-flip logloss {math.log(2):.4f}",
           "configs": {}}
    for name in CONFIGS:
        g = r[r.config == name]
        out["configs"][name] = {"params": {k: v for k, v in CONFIGS[name].items() if PROD.get(k) != v} or "production",
                                "selection": agg(g[g.part == "selection"]), "holdout": agg(g[g.part == "holdout"]),
                                "mean_rounds": round(float(g["rounds"].mean()), 1)}
    ranked = sorted(out["configs"], key=lambda k: out["configs"][k]["selection"]["logloss"])
    out["best_on_selection"] = ranked[0]
    print(f"{'config':<16}{'sel hit':>8}{'sel ll':>9}{'hold hit':>9}{'hold ll':>9}{'hold edge≥5':>16}{'rounds':>8}")
    for k in ranked:
        c = out["configs"][k]
        print(f"{k:<16}{c['selection']['hit_rate']:>8}{c['selection']['logloss']:>9}{c['holdout']['hit_rate']:>9}{c['holdout']['logloss']:>9}"
              f"{str(c['holdout']['edge5_hit_rate']) + '/' + str(c['holdout']['edge5_n']):>16}{c['mean_rounds']:>8}")
    print("best on selection:", ranked[0], "→ holdout", out["configs"][ranked[0]]["holdout"])
    Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
