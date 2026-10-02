"""Out-of-sample ROI of the moneyline value ("diamond" / "golden") rule.

For every test season the ML model is trained on earlier seasons only (same recipe
and hyper-parameters as scripts/walk_forward_clean.py), then the production rule
`ValueFinder.evaluate_value` (model prob >= 70%, edge >= 5pp over the vig-free
moneyline, EV > 0; golden = value and |spread| <= 6) is applied to the test season
and settled at the actual closing moneyline.

Two probability variants:
  raw  booster output
  iso  isotonic calibration fitted on the last 10% of the training rows (the
       production runner applies an isotonic calibrator; this is its walk-forward
       counterpart — the tail is also the early-stopping set, so it is mildly optimistic)

Usage:
  PYTHONPATH=. python3 scripts/walk_forward_ml_value.py [--summary web/data/walk_forward_ml_value.json]
"""
import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import walk_forward_clean as wf  # noqa: E402
from src.Utils.OddsTables import load_seasons_odds  # noqa: E402
from src.Utils.ValueFinder import american_to_decimal, evaluate_value  # noqa: E402


def roi(returns, rng):
    """Mean return per unit staked with a bootstrap 95% CI (percent)."""
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return {"n": 0, "roi": None, "ci_lo": None, "ci_hi": None, "units": 0.0}
    boot = rng.choice(r, size=(4000, len(r)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n": int(len(r)), "roi": round(100 * r.mean(), 1), "ci_lo": round(100 * lo, 1),
            "ci_hi": round(100 * hi, 1), "units": round(float(r.sum()), 1)}


def line(label, x, rng):
    h = wf.rec(x.won.sum(), len(x))
    r = roi(x.ret, rng)
    out = {"record": h, "roi": r,
           "underdog_share": round(100 * (x.odds > 0).mean(), 1) if len(x) else None,
           "home_share": round(100 * (x.side == "home").mean(), 1) if len(x) else None}
    if len(x):
        print(f"  {label:<10} {wf.fmt(h)} | ROI {r['roi']:+6.1f}% [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}] "
              f"| {r['units']:+7.1f}u | underdogs {out['underdog_share']:.0f}%")
    else:
        print(f"  {label:<10}       —")
    return out


def run(args):
    os.chdir(REPO)
    ml = wf._load("ml_trainer", "src/Train-Models/XGBoost_Model_ML.py")
    ats = wf._load("ats_trainer", "src/Train-Models/XGBoost_Model_ATS.py")

    df = ml.load_dataset(args.dataset)
    df["Date"] = pd.to_datetime(df["Date"].astype(str).str[:10])
    df = df[df["Score"].fillna(0) > 0].sort_values("Date", kind="stable").reset_index(drop=True)
    df["season"] = df["Date"].apply(wf.season_of)
    cols = [c for c in df.columns if c not in ml.DROP_COLUMNS + ["season"]]

    with sqlite3.connect(ats.ODDS_DB) as con:
        odds = load_seasons_odds(con, ats.SEASON_KEYS, ["Date", "Home", "Away", "Spread", "ML_Home", "ML_Away"])
    odds["Date"] = odds["Date"].astype(str).str[:10]
    odds = odds.drop_duplicates(["Date", "Home", "Away"])

    tests = [s for s in sorted(df["season"].unique()) if s >= args.first_test]
    rows = []
    for s in tests:
        train, test = df[df.season < s], df[df.season == s]
        X, y = train[cols].astype(float).to_numpy(), train["Home-Team-Win"].astype(int).to_numpy()
        _, _, Xv, yv = wf.split_tail(X, y)
        for seed in args.seeds:
            _, _, predict = wf.fit_classifier(X, y, wf.ML_MCW, seed, "merror")
            p_raw = predict(test[cols].astype(float).to_numpy())
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(predict(Xv), yv)
            p_iso = np.clip(iso.transform(p_raw), 1e-6, 1 - 1e-6)
            for variant, p in (("raw", p_raw), ("iso", p_iso)):
                rows.append(pd.DataFrame({
                    "season": s, "seed": seed, "variant": variant,
                    "Date": test["Date"].dt.strftime("%Y-%m-%d").to_numpy(),
                    "Home": test["TEAM_NAME"].to_numpy(), "Away": test["TEAM_NAME.1"].to_numpy(),
                    "p_home": p, "home_win": test["Home-Team-Win"].astype(int).to_numpy()}))
            print(f"  {s} seed {seed} | train rows {len(X)}", flush=True)

    g = pd.concat(rows, ignore_index=True).merge(odds, how="left", on=["Date", "Home", "Away"])
    priced = g[g.ML_Home.notna() & g.ML_Away.notna()].copy()
    ev = [evaluate_value(p, h, a, spread=None if pd.isna(sp) else sp)
          for p, h, a, sp in zip(priced.p_home, priced.ML_Home, priced.ML_Away, priced.Spread)]
    priced["is_value"] = [e["is_value"] for e in ev]
    priced["is_golden"] = [e["is_golden"] for e in ev]
    priced["side"] = [e["value_side"] for e in ev]
    bets = priced[priced.is_value].copy()
    bets["odds"] = np.where(bets.side == "home", bets.ML_Home, bets.ML_Away)
    bets["won"] = ((bets.side == "home") == (bets.home_win == 1)).astype(int)
    bets["ret"] = np.where(bets.won == 1, bets.odds.map(american_to_decimal) - 1.0, -1.0)

    # Reference: stake one unit on the closing-line favourite in every priced game.
    ref = priced[(priced.seed == args.seeds[0]) & (priced.variant == "raw") & (priced.ML_Home != priced.ML_Away)].copy()
    ref["side"] = np.where(ref.ML_Home < ref.ML_Away, "home", "away")
    ref["odds"] = np.where(ref.side == "home", ref.ML_Home, ref.ML_Away)
    ref["won"] = ((ref.side == "home") == (ref.home_win == 1)).astype(int)
    ref["ret"] = np.where(ref.won == 1, ref.odds.map(american_to_decimal) - 1.0, -1.0)

    rng = np.random.default_rng(20261002)
    seed0 = args.seeds[0]
    n_priced = int(((priced.seed == seed0) & (priced.variant == "raw")).sum())
    n_all = int(((g.seed == seed0) & (g.variant == "raw")).sum())
    print(f"\ntest games {n_all} | with both moneylines {n_priced}")
    result = {"dataset": args.dataset, "tests": tests, "seeds": args.seeds, "games_priced": n_priced,
              "rule": "model prob >= 70%, edge >= 5pp vs vig-free moneyline, EV > 0; golden: |spread| <= 6",
              "variants": {}}
    print("\n== reference: closing-line favourite, every priced game ==")
    result["favourite_every_game"] = line("ALL", ref, rng)

    for variant in ("raw", "iso"):
        out = {}
        for tier, mask in (("value", bets.is_value), ("golden", bets.is_golden)):
            b = bets[(bets.variant == variant) & mask]
            b0 = b[b.seed == seed0]
            print(f"\n== {variant} / {tier} (seed {seed0}) ==")
            by_season = {s: line(s, b0[b0.season == s], rng) for s in tests}
            pooled = line("ALL", b0, rng)
            seeds = {}
            if len(args.seeds) > 1:
                print("  seed spread (pooled):")
                for seed in args.seeds:
                    seeds[str(seed)] = line(f"seed {seed}", b[b.seed == seed], rng)
            out[tier] = {"by_season": by_season, "pooled": pooled, "seeds": seeds,
                         "seasons_positive": sum(1 for v in by_season.values()
                                                 if v["roi"]["roi"] is not None and v["roi"]["roi"] > 0)}
        result["variants"][variant] = out

    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1) + "\n")
        print(f"\nwrote {args.json}")
    if args.summary:
        compact = {
            "generated": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "script": "scripts/walk_forward_ml_value.py",
            "method": "each season scored by an ML model trained only on earlier seasons; bets settled at the closing moneyline",
            "rule": result["rule"], "test_seasons": tests, "seeds": args.seeds, "first_seed": seed0,
            "games_priced": n_priced,
            "favourite_every_game": {k: result["favourite_every_game"][k] for k in ("record", "roi")},
        }
        for variant, v in result["variants"].items():
            compact[variant] = {tier: {"record": t["pooled"]["record"], "roi": t["pooled"]["roi"],
                                       "seasons_positive": t["seasons_positive"],
                                       "seed_roi": {k: x["roi"]["roi"] for k, x in t["seeds"].items()}}
                                for tier, t in v.items()}
        Path(args.summary).write_text(json.dumps(compact, indent=1, ensure_ascii=False) + "\n")
        print(f"wrote {args.summary}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="dataset_2012-26")
    ap.add_argument("--first-test", default="2016-17")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 2026])
    ap.add_argument("--json")
    ap.add_argument("--summary", help="compact JSON for the frontend, e.g. web/data/walk_forward_ml_value.json")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
