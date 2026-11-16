"""Validate the regular-season backtest strategies on every past season.

For each finished regular-season game in OddsData.sqlite the pre-game team
profiles (strictly before the game date) decide which rules fire; the pick is
graded against the stored spread. Output: per-rule record by season, pooled
record, ROI at the 0.87-0.90 payout, and a walk-forward verdict test where a
rule's vote weight on date D only uses results before D.

Usage: PYTHONPATH=. python scripts/validate_strategies.py [--json web/data/strategy_validation.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import date as _date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.Utils import StrategyMatches as SM
from src.Utils.PlayoffContext import is_playoff_date
from src.Utils.SeasonStats import SEASON_BOUNDS
from src.Utils.TeamProfile import _load_all_games, team_profile_for_date, grade_spread


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(100 * (c - m) / d, 1), round(100 * (c + m) / d, 1)]


def rec(w, l):
    n = w + l
    return {"wins": w, "losses": l, "n": n, "hit_rate": round(w / n * 100, 1) if n else None,
            "ci95": wilson(w, n), **SM.payout_profile(w, l)}


def season_of(d: str) -> str | None:
    for key, (lo, hi) in SEASON_BOUNDS.items():
        if lo <= d <= hi:
            return key
    return None


def _profile(team, d):
    try:
        return team_profile_for_date(team, d)
    except KeyError:   # team with no earlier game at all (first row of the data)
        return None


def collect(current_season: str | None = None) -> pd.DataFrame:
    """One row per (game, fired rule): date, season, key, side, result."""
    games = _load_all_games().sort_values("Date")
    rows = []
    for g in games.itertuples(index=False):
        d = g.Date.strftime("%Y-%m-%d")
        if g.Spread is None or pd.isna(g.Spread) or pd.isna(g.Win_Margin):
            continue
        try:
            if is_playoff_date(_date.fromisoformat(d)):
                continue
        except Exception:
            pass
        season = season_of(d)
        if season is None:
            continue
        hp, ap = _profile(g.Home, d), _profile(g.Away, d)
        keys = SM.matched_keys(g.Spread, hp, ap)
        if not keys:
            continue
        winner, _ = grade_spread(g.Spread, g.Win_Margin)
        for k in keys:
            side = SM.pick_side(k, g.Spread)
            res = "push" if winner == "push" else ("win" if winner == side else "loss")
            rows.append({"date": d, "season": season, "home": g.Home, "away": g.Away,
                         "spread": float(g.Spread), "key": k, "side": side, "result": res})
    return pd.DataFrame(rows)


def verdict_walk_forward(df: pd.DataFrame, prior_rates: dict[str, float], prior_n: int) -> pd.DataFrame:
    """Replay verdict_for chronologically; weights use only earlier dates."""
    running = defaultdict(lambda: [0, 0])
    out = []

    def weight(k):
        w, l = running[k]
        pooled = (w + prior_n * prior_rates[k]) / (w + l + prior_n)
        return max(0.0, pooled * 100 - 50)

    for d, day in df.groupby("date", sort=True):
        for (home, away), sig in day.groupby(["home", "away"]):
            votes = [{"key": r.key, "side": r.side, "w": weight(r.key)} for r in sig.itertuples()]
            dog = [v for v in votes if v["key"] in SM.DOG_FAMILY]
            if len(dog) > 1:
                keep = max(dog, key=lambda v: v["w"])
                for v in dog:
                    if v is not keep:
                        v["w"] = 0.0
            sc = {"home": 0.0, "away": 0.0}
            for v in votes:
                sc[v["side"]] += v["w"]
            if sc["home"] == sc["away"]:
                continue
            side = "home" if sc["home"] > sc["away"] else "away"
            ref = sig[sig["side"] == side].iloc[0]
            out.append({"date": d, "season": ref.season, "margin": abs(sc["home"] - sc["away"]),
                        "conflict": sig["side"].nunique() > 1, "result": ref.result})
        for r in day.itertuples():
            if r.result == "win":
                running[r.key][0] += 1
            elif r.result == "loss":
                running[r.key][1] += 1
    return pd.DataFrame(out)


def _wl(frame):
    return int((frame["result"] == "win").sum()), int((frame["result"] == "loss").sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="web/data/strategy_validation.json")
    ap.add_argument("--cache", help="pickle of the collected rows (skips the 6-minute scan)")
    ap.add_argument("--current", default="2026-27", help="season excluded from the 'past seasons' pool")
    args = ap.parse_args()

    cache = Path(args.cache) if args.cache else None
    if cache and cache.exists():
        df = pd.read_pickle(cache)
    else:
        df = collect()
        if cache:
            df.to_pickle(cache)
    past = df[df["season"] != args.current]
    seasons = sorted(past["season"].unique())
    result = {"generated_for": "scripts/validate_strategies.py", "past_seasons": seasons,
              "current_season": args.current, "strategies": {}}
    print(f"{len(seasons)} past seasons {seasons[0]} → {seasons[-1]}; rows {len(past)}")
    for s in SM.STRATEGIES:
        k = s["key"]
        sub = past[past["key"] == k]
        pooled = rec(*_wl(sub))
        by_season = {sn: rec(*_wl(sub[sub["season"] == sn])) for sn in seasons}
        above = sum(1 for r in by_season.values() if r["hit_rate"] is not None and r["hit_rate"] >= SM.BREAKEVEN_HIGH)
        cur = rec(*_wl(df[(df["season"] == args.current) & (df["key"] == k)]))
        result["strategies"][k] = {"name_zh": s["name_zh"], "claimed_hist_wr": s["hist_wr"], "past": pooled,
                                   "seasons_above_breakeven": above, "seasons_total": len(seasons),
                                   "by_season": by_season, "current": cur}
        print(f"{s['name_zh']:<16} claimed {s['hist_wr']:>5}% | past {pooled['wins']}-{pooled['losses']} "
              f"{pooled['hit_rate']}% CI {pooled['ci95']} ROI {pooled['roi_low']}~{pooled['roi_high']} "
              f"| seasons ≥{SM.BREAKEVEN_HIGH}%: {above}/{len(seasons)} | {args.current} {cur['wins']}-{cur['losses']} {cur['hit_rate']}%")

    # Verdict, walk-forward over all seasons, prior = coin flip (no borrowed backtest number).
    vd = verdict_walk_forward(df, {s["key"]: 0.5 for s in SM.STRATEGIES}, SM.PRIOR_N)
    vpast = vd[vd["season"] != args.current]
    result["verdict_walk_forward"] = {"prior": f"50% x {SM.PRIOR_N} games", "all": rec(*_wl(vpast)),
                                      "conflict_only": rec(*_wl(vpast[vpast["conflict"]])), "by_margin": []}
    print("verdict (walk-forward, 50% prior) past:", result["verdict_walk_forward"]["all"])
    for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 5), (5, 99)]:
        r = rec(*_wl(vpast[(vpast["margin"] >= lo) & (vpast["margin"] < hi)]))
        result["verdict_walk_forward"]["by_margin"].append({"lo": lo, "hi": hi, **r})
        print(f"  margin {lo}-{hi}: {r['wins']}-{r['losses']} {r['hit_rate']}% ROI {r['roi_low']}~{r['roi_high']}")
    Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", args.json)


if __name__ == "__main__":
    main()
