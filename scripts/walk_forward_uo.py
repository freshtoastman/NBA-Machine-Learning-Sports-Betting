"""Walk-forward evaluation of the totals (over/under) model on the clean dataset.

Same method as scripts/walk_forward_clean.py: for every test season N a model is trained
only on seasons before N (production hyper-parameters of the pinned UO model, 3-class
UNDER / OVER / PUSH, class-balanced weights, early stopping on the tail 10% of the
training rows), then season N is scored once. Pushes are no bet.

Recipes:
    prod     trainer as-is: eval metrics [mlogloss, merror] → merror decides the stop,
             every tree is used at predict time (how the pinned model was built)
    logloss  stop on log loss, best iteration used at predict time

Usage:
    PYTHONPATH=. python3 scripts/walk_forward_uo.py [--first-test 2016-17]
        [--recipes prod logloss] [--seeds 42] [--json OUT] [--picks-csv OUT]
        [--summary web/data/walk_forward.json]   # merges a "uo" block into the file
"""
import argparse
import datetime as dt
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

# Hyper-parameters of the pinned production totals model
# (XGBoost_52.2%_UO_md10_eta0p081_sub0p636_col0p548_cbl0p951_cbn0p728_mcw9_g2p024_mds7_mb402_l1p440_a0p030_nb1319).
UO_PARAMS = {
    "max_depth": 10, "eta": 0.081, "subsample": 0.636, "colsample_bytree": 0.548,
    "colsample_bylevel": 0.951, "colsample_bynode": 0.728, "min_child_weight": 9,
    "gamma": 2.024, "max_delta_step": 7, "max_bin": 402, "lambda": 1.440, "alpha": 0.030,
    "objective": "multi:softprob", "num_class": 3, "tree_method": "hist",
}
NUM_ROUNDS, PATIENCE = 1319, 60
RECIPES = ("prod", "logloss")
CONF_TIERS = (55, 60, 65, 70)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def season_of(date):
    y = date.year if date.month >= 10 else date.year - 1
    return f"{y}-{(y + 1) % 100:02d}"


def wilson(k, n, z=1.96):
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * (c - h), 1), round(100 * (c + h), 1)


def rec(hits, n):
    lo, hi = wilson(int(hits), int(n))
    return {"hits": int(hits), "n": int(n), "pct": round(100 * hits / n, 1) if n else None,
            "ci_lo": lo, "ci_hi": hi}


def fmt(r):
    if not r["n"]:
        return "      —"
    return f"{r['hits']:4d}-{r['n'] - r['hits']:<4d} {r['pct']:5.1f}% [{r['ci_lo']:.0f}-{r['ci_hi']:.0f}]"


def split_tail(X, y, frac=0.1):
    cut = int(len(X) * (1 - frac))
    return X[:cut], y[:cut], X[cut:], y[cut:]


def fit(X, y, seed, recipe):
    params = dict(UO_PARAMS, seed=seed,
                  eval_metric=["mlogloss", "merror"] if recipe == "prod" else ["merror", "mlogloss"])
    Xt, yt, Xv, yv = split_tail(X, y)
    counts = np.bincount(yt, minlength=3)
    w = np.array([len(yt) / (3 * counts[c]) if counts[c] else 1.0 for c in yt])
    booster = xgb.train(params, xgb.DMatrix(Xt, label=yt, weight=w), num_boost_round=NUM_ROUNDS,
                        evals=[(xgb.DMatrix(Xv, label=yv), "val")], early_stopping_rounds=PATIENCE,
                        verbose_eval=False)
    if recipe == "prod":
        return booster.num_boosted_rounds(), lambda Z: booster.predict(xgb.DMatrix(Z))
    rounds = booster.best_iteration + 1
    return rounds, lambda Z: booster.predict(xgb.DMatrix(Z), iteration_range=(0, rounds))


def run(args):
    os.chdir(REPO)
    uo = _load("uo_trainer", "src/Train-Models/XGBoost_Model_UO.py")
    df = uo.load_dataset(args.dataset)
    df["Date"] = pd.to_datetime(df["Date"].astype(str).str[:10])
    df = df[(df["Score"].fillna(0) > 0) & df["OU"].fillna(0).gt(0)]
    df = df.sort_values("Date", kind="stable").reset_index(drop=True)
    df["season"] = df["Date"].apply(season_of)
    cols = [c for c in df.columns if c not in uo.DROP_COLUMNS + ["season"]]
    # Actual total = home score + away score is not stored; the label already encodes it.
    seasons = sorted(df["season"].unique())
    tests = [s for s in seasons if s >= args.first_test]
    print(f"rows with total line+result: {len(df)} | features {len(cols)} | seasons {seasons[0]} … {seasons[-1]} "
          f"| test seasons: {tests}")

    picks, fits = [], []
    for s in tests:
        train, test = df[df.season < s], df[df.season == s].reset_index(drop=True)
        X = train[cols].astype(float).to_numpy()
        y = train[uo.TARGET_COLUMN].astype(int).to_numpy()
        Xtest = test[cols].astype(float).to_numpy()
        for seed in args.seeds:
            for recipe in args.recipes:
                rounds, predict = fit(X, y, seed, recipe)
                p = predict(Xtest)
                cls = p.argmax(axis=1)                       # production: argmax over all 3 classes
                out = pd.DataFrame({
                    "season": s, "seed": seed, "recipe": recipe,
                    "Date": test["Date"].dt.strftime("%Y-%m-%d"), "Home": test["TEAM_NAME"],
                    "Away": test["TEAM_NAME.1"], "OU": test["OU"], "label": test[uo.TARGET_COLUMN].astype(int),
                    "is_playoff": test["is_playoff"].fillna(0).astype(int) if "is_playoff" in test else 0,
                    "p_under": p[:, 0], "p_over": p[:, 1], "p_push": p[:, 2],
                    "pick_over": (cls == 1).astype(int), "conf": 100 * p[np.arange(len(p)), cls],
                })
                picks.append(out)
                fits.append({"season": s, "seed": seed, "recipe": recipe, "rounds": int(rounds), "train_rows": len(X)})
                print(f"  {s} seed {seed} {recipe:<8} trees {rounds:4d} | train rows {len(X)}", flush=True)

    picks = pd.concat(picks, ignore_index=True)
    picks = picks[picks.label != 2].copy()                   # pushes are no bet
    picks["hit"] = (picks.pick_over == picks.label).astype(int)
    picks["over_hit"] = (picks.label == 1).astype(int)       # "always OVER" baseline

    result = {"dataset": args.dataset, "first_test": args.first_test, "seeds": args.seeds,
              "params": UO_PARAMS, "fits": fits, "uo": {}}
    seed0 = args.seeds[0]
    for recipe in args.recipes:
        p0 = picks[(picks.recipe == recipe) & (picks.seed == seed0)]
        print(f"\n== Totals / {recipe} (seed {seed0}) — trained on earlier seasons only ==")
        out = {"by_season": {}, "conf": {}, "seeds": {}}
        for s in tests + ["ALL"]:
            x = p0 if s == "ALL" else p0[p0.season == s]
            r = {"raw": rec(x.hit.sum(), len(x)),
                 "over_pick_share": round(100 * x.pick_over.mean(), 1) if len(x) else None,
                 "always_over": rec(x.over_hit.sum(), len(x)),
                 "always_under": rec(len(x) - x.over_hit.sum(), len(x))}
            for t in CONF_TIERS:
                z = x[x.conf >= t]
                r[f"conf{t}"] = rec(z.hit.sum(), len(z))
            out["by_season"][s] = r
            print(f"  {s:<8} model {fmt(r['raw'])} | always UNDER {fmt(r['always_under'])} "
                  f"| conf≥55 {fmt(r['conf55'])} | conf≥60 {fmt(r['conf60'])} | conf≥65 {fmt(r['conf65'])}")
        for seg, mask in (("regular", p0.is_playoff == 0), ("playoffs", p0.is_playoff == 1)):
            x = p0[mask]
            out["conf"][seg] = {str(t): rec(x[x.conf >= t].hit.sum(), (x.conf >= t).sum()) for t in CONF_TIERS}
            out["conf"][seg]["all"] = rec(x.hit.sum(), len(x))
            print(f"  {seg:<9}" + " | ".join(f"≥{t}: {fmt(out['conf'][seg][str(t)])}" for t in CONF_TIERS))
        if len(args.seeds) > 1:
            for seed in args.seeds:
                x = picks[(picks.recipe == recipe) & (picks.seed == seed)]
                out["seeds"][str(seed)] = {"raw": rec(x.hit.sum(), len(x)),
                                           "conf60": rec(x[x.conf >= 60].hit.sum(), (x.conf >= 60).sum())}
                print(f"    seed {seed:<6} raw {fmt(out['seeds'][str(seed)]['raw'])} "
                      f"| conf≥60 {fmt(out['seeds'][str(seed)]['conf60'])}")
        result["uo"][recipe] = out

    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1))
        print(f"\nwrote {args.json}")
    if args.summary:
        merge_summary(Path(args.summary), result, tests, args)
        print(f"wrote {args.summary}")
    if args.picks_csv:
        picks.to_csv(args.picks_csv, index=False)
        print(f"wrote {args.picks_csv}")
    return result


def merge_summary(path, result, tests, args):
    """Add/replace the "uo" block of web/data/walk_forward.json; other keys untouched."""
    lead = args.recipes[0]
    by = result["uo"][lead]["by_season"]
    summary = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    summary["uo"] = {
        "generated": dt.date.today().isoformat(),
        "script": "scripts/walk_forward_uo.py",
        "recipe": lead, "seeds": args.seeds, "test_seasons": tests,
        "raw": by["ALL"]["raw"], "always_under": by["ALL"]["always_under"],
        "conf60": by["ALL"]["conf60"], "conf65": by["ALL"]["conf65"],
        "raw_pct_by_seed": [round(100 * r["raw"]["hits"] / r["raw"]["n"], 1)
                            for r in result["uo"][lead]["seeds"].values()] or [by["ALL"]["raw"]["pct"]],
        "by_season": {s: {"raw": by[s]["raw"], "conf60": by[s]["conf60"]} for s in tests},
    }
    path.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="dataset_2012-26")
    ap.add_argument("--first-test", default="2016-17")
    ap.add_argument("--recipes", nargs="+", default=list(RECIPES), choices=RECIPES)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--json")
    ap.add_argument("--picks-csv")
    ap.add_argument("--summary")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
