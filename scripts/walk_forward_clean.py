"""Walk-forward evaluation of the ML / ATS recipes on the clean (pre-game) dataset.

For every test season N the models are trained only on seasons before N, then score
season N once. Nothing about season N (rows, thresholds, early stopping) reaches the
model that predicts it, so each season's numbers are out-of-sample.

ATS recipes (all use the production hyper-parameters, 175-column layout unless noted):
    prod     production trainer as-is: early stopping on validation error rate, all trees
             used at predict time (what Models/XGBoost_Models/*ATS*mcw26* was built with)
    logloss  early stopping on validation log loss, best iteration used at predict time
    noadv    `logloss` without the 28 ADV_ columns (they exist for 2025-26 only, so no
             earlier season can validate them)
    margin   regression on the cover margin (Win_Margin - Spread), same columns as `noadv`;
             "edge" is the predicted cover margin in points

Spreads come from src.Utils.OddsTables, which signs the legacy (through 2021-22) tables;
with the unsigned values those seasons score 58-60% "ATS" because the label degenerates
into "did the home team win by more than the line".

Usage:
    PYTHONPATH=. python3 scripts/walk_forward_clean.py [--first-test 2016-17]
        [--recipes prod logloss noadv margin] [--seeds 42 7 2026]
        [--json OUT] [--picks-csv OUT] [--summary web/data/walk_forward.json]
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

# Hyper-parameters of the production models (file names under Models/XGBoost_Models;
# the ATS file name omits the columns it shares with the ML model of the same seed).
BASE_PARAMS = {
    "max_depth": 4, "eta": 0.033, "subsample": 0.657, "colsample_bytree": 0.919,
    "colsample_bylevel": 0.747, "colsample_bynode": 0.558, "gamma": 0.721,
    "max_delta_step": 9, "max_bin": 883, "lambda": 0.129, "alpha": 0.057,
    "tree_method": "hist",
}
ML_MCW, ATS_MCW, NUM_ROUNDS, PATIENCE = 16, 26, 1544, 60
RECIPES = ("prod", "logloss", "noadv", "margin")


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


def fit_classifier(X, y, mcw, seed, stop_on):
    """stop_on='merror' reproduces the production trainers (last eval metric decides,
    every tree is used afterwards); 'mlogloss' stops on log loss and keeps the best."""
    params = dict(BASE_PARAMS, min_child_weight=mcw, seed=seed, objective="multi:softprob", num_class=2,
                  eval_metric=["mlogloss", "merror"] if stop_on == "merror" else ["merror", "mlogloss"])
    Xt, yt, Xv, yv = split_tail(X, y)
    counts = np.bincount(yt, minlength=2)
    w = np.array([len(yt) / (2 * counts[c]) for c in yt])
    booster = xgb.train(params, xgb.DMatrix(Xt, label=yt, weight=w), num_boost_round=NUM_ROUNDS,
                        evals=[(xgb.DMatrix(Xv, label=yv), "val")], early_stopping_rounds=PATIENCE,
                        verbose_eval=False)
    if stop_on == "merror":
        return booster, booster.num_boosted_rounds(), lambda Z: booster.predict(xgb.DMatrix(Z))[:, 1]
    rounds = booster.best_iteration + 1
    return booster, rounds, lambda Z: booster.predict(xgb.DMatrix(Z), iteration_range=(0, rounds))[:, 1]


def fit_margin(X, y, seed):
    params = dict(BASE_PARAMS, min_child_weight=ATS_MCW, seed=seed, objective="reg:pseudohubererror",
                  huber_slope=12.0, eval_metric="mae", base_score=0.0)
    params.pop("max_delta_step")
    Xt, yt, Xv, yv = split_tail(X, y)
    booster = xgb.train(params, xgb.DMatrix(Xt, label=yt), num_boost_round=NUM_ROUNDS,
                        evals=[(xgb.DMatrix(Xv, label=yv), "val")], early_stopping_rounds=PATIENCE,
                        verbose_eval=False)
    rounds = booster.best_iteration + 1
    return booster, rounds, lambda Z: booster.predict(xgb.DMatrix(Z), iteration_range=(0, rounds))


def ats_matrices(ats, train_df, test_df, recipe):
    """Augment the training rows exactly like the production trainer and return
    (X_train, y_train, X_test, column names) sorted by date."""
    aug = ats.symmetric_augment(train_df)
    aug = aug.sort_values("Date", kind="stable")
    drop = list(ats.DROP_COLUMNS) + ["season"]
    if recipe in ("noadv", "margin"):
        drop += [c for c in aug.columns if c.startswith("ADV_")]
    cols = [c for c in aug.columns if c not in drop]
    X = aug[cols].astype(float).to_numpy()
    if recipe == "margin":
        y = (aug["Win_Margin"] - aug["Spread"]).astype(float).to_numpy()
    else:
        y = aug["ATS_Cover"].astype(int).to_numpy()
    return X, y, test_df[cols].astype(float).to_numpy(), cols


def segment(date, is_playoff):
    if is_playoff:
        return "playoffs"
    return "oct_feb" if date.month in (10, 11, 12, 1, 2) else "mar_apr"


def run(args):
    os.chdir(REPO)
    ats = _load("ats_trainer", "src/Train-Models/XGBoost_Model_ATS.py")
    ml = _load("ml_trainer", "src/Train-Models/XGBoost_Model_ML.py")

    df = ats.load_dataset(args.dataset)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date", kind="stable").reset_index(drop=True)
    df["season"] = df["Date"].apply(season_of)

    ml_df = ml.load_dataset(args.dataset)
    ml_df["Date"] = pd.to_datetime(ml_df["Date"].astype(str).str[:10])
    ml_df = ml_df[ml_df["Score"].fillna(0) > 0].sort_values("Date", kind="stable").reset_index(drop=True)
    ml_df["season"] = ml_df["Date"].apply(season_of)
    ml_cols = [c for c in ml_df.columns if c not in ml.DROP_COLUMNS + ["season"]]

    seasons = sorted(df["season"].unique())
    tests = [s for s in seasons if s >= args.first_test]
    print(f"rows with spread+result: {len(df)} | seasons {seasons[0]} … {seasons[-1]} | test seasons: {tests}")

    picks, ml_rows, fits = [], [], []
    for s in tests:
        train, test = df[df.season < s], df[df.season == s].reset_index(drop=True)
        mtrain, mtest = ml_df[ml_df.season < s], ml_df[ml_df.season == s].reset_index(drop=True)
        base = pd.DataFrame({
            "season": s, "Date": test["Date"].dt.strftime("%Y-%m-%d"), "Home": test["TEAM_NAME"],
            "Away": test["TEAM_NAME.1"], "Spread": test["Spread"], "Win_Margin": test["Win_Margin"],
            "segment": [segment(d, p) for d, p in zip(test["Date"], test["is_playoff"].fillna(0) > 0)],
            "pm_home": test["PLUS_MINUS"], "pm_away": test["PLUS_MINUS.1"],
        })
        for seed in args.seeds:
            # Moneyline: production recipe, no augmentation, raw booster probabilities.
            _, rounds, predict = fit_classifier(mtrain[ml_cols].astype(float).to_numpy(),
                                                mtrain["Home-Team-Win"].astype(int).to_numpy(),
                                                ML_MCW, seed, "merror")
            p = predict(mtest[ml_cols].astype(float).to_numpy())
            ml_rows.append(pd.DataFrame({
                "season": s, "seed": seed, "Date": mtest["Date"].dt.strftime("%Y-%m-%d"),
                "Home": mtest["TEAM_NAME"], "Away": mtest["TEAM_NAME.1"],
                "p_home": p, "home_win": mtest["Home-Team-Win"].astype(int)}))
            fits.append({"season": s, "seed": seed, "model": "ML", "rounds": rounds, "train_rows": len(mtrain)})
            for recipe in args.recipes:
                X, y, Xtest, cols = ats_matrices(ats, train, test, recipe)
                if recipe == "margin":
                    _, rounds, predict = fit_margin(X, y, seed)
                    score = predict(Xtest)                      # predicted cover margin, points
                    pick_home, edge = score > 0, np.abs(score)
                else:
                    _, rounds, predict = fit_classifier(X, y, ATS_MCW, seed,
                                                        "merror" if recipe == "prod" else "mlogloss")
                    score = predict(Xtest)
                    pick_home, edge = score >= 0.5, np.abs(score - 0.5) * 100
                out = base.copy()
                out["recipe"], out["seed"], out["score"] = recipe, seed, score
                out["pick_home"], out["edge"] = pick_home.astype(int), edge
                picks.append(out)
                fits.append({"season": s, "seed": seed, "model": f"ATS/{recipe}", "rounds": rounds,
                             "train_rows": len(X), "features": len(cols)})
                print(f"  {s} seed {seed} {recipe:<8} trees {rounds:4d} | train rows {len(X)}", flush=True)

    picks = pd.concat(picks, ignore_index=True)
    cover = picks.Win_Margin - picks.Spread
    picks = picks[cover != 0].copy()                            # pushes are no bet
    picks["hit"] = ((picks.Win_Margin - picks.Spread > 0).astype(int) == picks.pick_home).astype(int)
    # Away-pick quality filter from main.py, with per-game PLUS_MINUS standing in for
    # ADV_NET_RATING (the two agree within ~1 point; ADV_ exists for 2025-26 only).
    picks["qfilt"] = ((picks.pick_home == 0) & (picks.pm_home < -3) & (picks.Spread.abs() <= 8)
                      & ((picks.pm_home - picks.pm_away) > -7))
    picks["prod_value"] = (picks.segment == "oct_feb") & (picks.edge >= np.where(picks.pick_home == 1, 8.0, 9.0))
    # Edge rank inside (recipe, seed, season): comparable across recipes whose raw
    # edges live on different scales.
    picks["edge_rank"] = picks.groupby(["recipe", "seed", "season"])["edge"].rank(pct=True, ascending=False)
    ml_all = pd.concat(ml_rows, ignore_index=True)
    ml_all["hit"] = ((ml_all.p_home >= 0.5).astype(int) == ml_all.home_win).astype(int)

    result = {"dataset": args.dataset, "first_test": args.first_test, "seeds": args.seeds,
              "params": dict(BASE_PARAMS, ml_min_child_weight=ML_MCW, ats_min_child_weight=ATS_MCW),
              "fits": fits, "ml": {}, "ats": {}}

    seed0 = args.seeds[0]
    print(f"\n== Moneyline (seed {seed0}) — trained on earlier seasons only ==")
    m0 = ml_all[ml_all.seed == seed0]
    odds = df[["Date", "TEAM_NAME", "TEAM_NAME.1", "Spread"]].copy()
    odds["Date"] = odds["Date"].dt.strftime("%Y-%m-%d")
    m0 = m0.merge(odds, how="left", left_on=["Date", "Home", "Away"], right_on=["Date", "TEAM_NAME", "TEAM_NAME.1"])
    m0["fav_hit"] = ((m0.Spread > 0).astype(int) == m0.home_win).astype(int)
    for s in tests + ["ALL"]:
        x = m0 if s == "ALL" else m0[m0.season == s]
        f = x[x.Spread.notna() & (x.Spread != 0)]
        hi = x[(x.p_home - 0.5).abs() >= 0.20]
        r = {"model": rec(x.hit.sum(), len(x)), "market_favorite": rec(f.fav_hit.sum(), len(f)),
             "model_same_games": rec(f.hit.sum(), len(f)), "conf70": rec(hi.hit.sum(), len(hi))}
        result["ml"][s] = r
        print(f"  {s:<8} model {fmt(r['model'])} | closing-line favourite {fmt(r['market_favorite'])} "
              f"| model ≥70% conf {fmt(r['conf70'])}")

    for recipe in args.recipes:
        p0 = picks[(picks.recipe == recipe) & (picks.seed == seed0)]
        unit = "pts" if recipe == "margin" else "%"
        print(f"\n== ATS / {recipe} (seed {seed0}) ==")
        out = {"by_season": {}, "pooled": {}, "seeds": {}}
        for s in tests + ["ALL"]:
            x = p0 if s == "ALL" else p0[p0.season == s]
            reg = x[x.segment != "playoffs"]
            v = x[x.prod_value]
            r = {"raw": rec(x.hit.sum(), len(x)),
                 "home_pick_share": round(100 * x.pick_home.mean(), 1) if len(x) else None,
                 "top10": rec(reg[reg.edge_rank <= 0.10].hit.sum(), (reg.edge_rank <= 0.10).sum()),
                 "top5": rec(reg[reg.edge_rank <= 0.05].hit.sum(), (reg.edge_rank <= 0.05).sum()),
                 "top2": rec(reg[reg.edge_rank <= 0.02].hit.sum(), (reg.edge_rank <= 0.02).sum()),
                 "prod_value": rec(v.hit.sum(), len(v)),
                 "prod_value_filtered": rec(v[~v.qfilt].hit.sum(), (~v.qfilt).sum())}
            out["by_season"][s] = r
            line = (f"  {s:<8} raw {fmt(r['raw'])} | top10% {fmt(r['top10'])} | top5% {fmt(r['top5'])} "
                    f"| top2% {fmt(r['top2'])}")
            if recipe != "margin":
                line += f" | H8/A9 Oct-Feb {fmt(r['prod_value'])} | +filter {fmt(r['prod_value_filtered'])}"
            print(line)
        print(f"  pooled by edge ({unit}) and segment:")
        grid = (1.5, 2, 3, 4, 5, 6) if recipe == "margin" else (2, 4, 6, 8, 10, 12)
        for seg in ("oct_feb", "mar_apr", "playoffs"):
            x = p0[p0.segment == seg]
            cells = {}
            for t in grid:
                z = x[x.edge >= t]
                cells[str(t)] = {"all": rec(z.hit.sum(), len(z)),
                                 "home": rec(z[z.pick_home == 1].hit.sum(), (z.pick_home == 1).sum()),
                                 "away": rec(z[z.pick_home == 0].hit.sum(), (z.pick_home == 0).sum())}
            out["pooled"][seg] = cells
            print(f"    {seg:<9}" + " | ".join(f"≥{t}: {fmt(cells[str(t)]['all'])}" for t in grid))
        if len(args.seeds) > 1:
            print("  seed spread (pooled over test seasons):")
            for seed in args.seeds:
                x = picks[(picks.recipe == recipe) & (picks.seed == seed)]
                reg = x[x.segment != "playoffs"]
                v = x[x.prod_value]
                r = {"raw": rec(x.hit.sum(), len(x)),
                     "top5": rec(reg[reg.edge_rank <= 0.05].hit.sum(), (reg.edge_rank <= 0.05).sum()),
                     "prod_value": rec(v.hit.sum(), len(v))}
                out["seeds"][str(seed)] = r
                print(f"    seed {seed:<6} raw {fmt(r['raw'])} | top5% {fmt(r['top5'])} | H8/A9 {fmt(r['prod_value'])}")
        result["ats"][recipe] = out

    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1))
        print(f"\nwrote {args.json}")
    if args.summary:
        Path(args.summary).write_text(json.dumps(summarize(result, picks, tests, args), indent=1, ensure_ascii=False) + "\n")
        print(f"wrote {args.summary}")
    if args.picks_csv:
        picks.to_csv(args.picks_csv, index=False)
        print(f"wrote {args.picks_csv}")
    return result


def summarize(result, picks, tests, args):
    """Compact out-of-sample record for the frontend (web/data/walk_forward.json)."""
    seed0 = args.seeds[0]
    recipes = {}
    for recipe in args.recipes:
        raw, top5 = [], []
        for seed in args.seeds:
            x = picks[(picks.recipe == recipe) & (picks.seed == seed)]
            reg = x[x.segment != "playoffs"]
            raw.append(round(100 * x.hit.mean(), 1))
            top5.append(round(100 * reg[reg.edge_rank <= 0.05].hit.mean(), 1))
        pooled = result["ats"][recipe]["by_season"]["ALL"]
        recipes[recipe] = {"raw": pooled["raw"], "top5": pooled["top5"], "top2": pooled["top2"],
                           "raw_pct_by_seed": raw, "top5_pct_by_seed": top5}
    lead = args.recipes[0]
    return {
        "generated": dt.date.today().isoformat(),
        "script": "scripts/walk_forward_clean.py",
        "method": "each season scored by models trained only on earlier seasons",
        "test_seasons": tests, "seeds": args.seeds, "first_seed": seed0,
        "break_even_pct": 52.4,
        "ml": result["ml"]["ALL"],
        "ml_by_season": {s: result["ml"][s] for s in tests},
        "ats_recipe": lead,
        "ats": recipes,
        "ats_by_season": {s: {k: result["ats"][lead]["by_season"][s][k] for k in ("raw", "top5")} for s in tests},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="dataset_2012-26")
    ap.add_argument("--first-test", default="2016-17")
    ap.add_argument("--recipes", nargs="+", default=list(RECIPES), choices=RECIPES)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 2026])
    ap.add_argument("--json")
    ap.add_argument("--picks-csv")
    ap.add_argument("--summary", help="compact JSON for the frontend, e.g. web/data/walk_forward.json")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
