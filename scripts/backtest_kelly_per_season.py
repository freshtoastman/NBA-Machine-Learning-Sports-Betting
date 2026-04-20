"""Per-season Quarter-Kelly backtest for ATS value picks (regular season only).

Runs the production ATS strategy (asymmetric threshold, Oct-Feb only: home>=8%,
away>=9%; Mar/early-Apr suppressed) on every season with available data and
simulates Quarter-Kelly sizing.

Usage: PYTHONPATH=. python scripts/backtest_kelly_per_season.py
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from main import _DATASET_DB, _DATASET_TABLE, _ODDS_DB, predict_historical_xgb  # noqa: E402

NET = 100 / 110  # 0.9091 -- net profit per $1 staked at -110

# Season regular-season date ranges (approximate)
SEASONS = [
    ("2018-19", "2018-10-16", "2019-04-10"),
    ("2019-20", "2019-10-22", "2020-03-11"),  # season cut short by COVID
    ("2020-21", "2020-12-22", "2021-05-16"),
    ("2021-22", "2021-10-19", "2022-04-10"),
    ("2022-23", "2022-10-18", "2023-04-09"),
    ("2023-24", "2023-10-24", "2024-04-14"),
    ("2024-25", "2024-10-22", "2025-04-13"),
    ("2025-26", "2025-10-22", "2026-04-13"),
]


def iter_dates(lo: str, hi: str):
    d = date.fromisoformat(lo)
    end = date.fromisoformat(hi)
    while d <= end:
        yield d
        d += timedelta(days=1)


def kelly_fraction(p: float, b: float = NET) -> float:
    q = 1.0 - p
    f = (b * p - q) / b
    return max(f, 0.0)


def _odds_table_for_season(season_key: str) -> str | None:
    candidates = [season_key, f"odds_{season_key}_new", f"odds_{season_key}"]
    with sqlite3.connect(_ODDS_DB) as con:
        for t in candidates:
            row = con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (t,)
            ).fetchone()
            if row:
                # verify it has graded games in the date window
                cnt = con.execute(f'SELECT COUNT(*) FROM "{t}" WHERE Points > 0').fetchone()[0]
                if cnt > 100:
                    return t
    return None


def collect_picks(season_key: str, start: str, end: str):
    # Monkey-patch main._season_table_for_date to resolve to the right table
    import main as _main
    table = _odds_table_for_season(season_key)
    if not table:
        return None, None
    orig = _main._season_table_for_date
    _main._season_table_for_date = lambda d: table  # type: ignore

    picks = []
    try:
        for d in iter_dates(start, end):
            try:
                preds = _main.predict_historical_xgb(d)
            except Exception:
                continue
            for p in preds:
                if not p.get("ats_is_value"):
                    continue
                if p.get("ats_winner") in (None, "push"):
                    continue
                edge = p.get("ats_value_edge") or 0.0
                picks.append({
                    "date": d.isoformat(),
                    "edge": edge,
                    "win": p.get("ats_model_pick") == p.get("ats_winner"),
                })
    finally:
        _main._season_table_for_date = orig  # restore

    return picks, table


def simulate_quarter_kelly(picks, bankroll, min_bet=1000, max_frac=0.05, kelly_coef=0.25):
    br = bankroll
    peak = bankroll
    max_dd = 0.0
    wins = losses = 0
    for p in picks:
        prob = 0.5 + p["edge"] / 100.0
        f = kelly_fraction(prob) * kelly_coef
        stake = max(min_bet, min(br * f, br * max_frac))
        stake = min(stake, br)
        if stake <= 0:
            break
        if p["win"]:
            br += stake * NET
            wins += 1
        else:
            br -= stake
            losses += 1
        peak = max(peak, br)
        max_dd = max(max_dd, peak - br)
    return {
        "final": br,
        "profit": br - bankroll,
        "roi": (br - bankroll) / bankroll,
        "max_dd": max_dd,
        "wins": wins,
        "losses": losses,
    }


def simulate_flat(picks, bankroll, min_bet=1000):
    br = bankroll
    peak = bankroll
    max_dd = 0.0
    wins = losses = 0
    for p in picks:
        stake = min_bet
        if stake > br:
            break
        if p["win"]:
            br += stake * NET
            wins += 1
        else:
            br -= stake
            losses += 1
        peak = max(peak, br)
        max_dd = max(max_dd, peak - br)
    return {
        "final": br,
        "profit": br - bankroll,
        "roi": (br - bankroll) / bankroll,
        "max_dd": max_dd,
        "wins": wins,
        "losses": losses,
    }


def main():
    BANKROLL = 50_000
    rows = []
    for season, start, end in SEASONS:
        picks, table = collect_picks(season, start, end)
        if picks is None:
            print(f"[{season}] no odds table found — skipped")
            continue
        n = len(picks)
        if n == 0:
            rows.append({
                "season": season, "n_picks": 0, "wr": 0.0,
                "flat_profit": 0, "kelly_profit": 0, "kelly_roi": 0.0, "kelly_maxdd": 0,
            })
            continue
        wr = sum(1 for p in picks if p["win"]) / n
        flat = simulate_flat(picks, BANKROLL)
        kelly = simulate_quarter_kelly(picks, BANKROLL)

        # longest losing streak
        cur = lng = 0
        for p in picks:
            if p["win"]:
                cur = 0
            else:
                cur += 1; lng = max(lng, cur)

        rows.append({
            "season": season, "n_picks": n, "wr": wr,
            "longest_loss_streak": lng,
            "flat_profit": flat["profit"], "flat_maxdd": flat["max_dd"],
            "kelly_profit": kelly["profit"], "kelly_roi": kelly["roi"],
            "kelly_maxdd": kelly["max_dd"], "kelly_final": kelly["final"],
            "odds_table": table,
        })

    df = pd.DataFrame(rows)
    print("\n==== Per-season Quarter-Kelly backtest (bankroll=$50,000) ====\n")
    print(df.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    if not df.empty:
        active = df[df["n_picks"] > 0]
        print("\n==== Aggregated (active seasons) ====")
        print(f"  total picks: {active['n_picks'].sum()}")
        print(f"  avg hit rate: {active['wr'].mean():.1%}")
        print(f"  avg kelly profit: ${active['kelly_profit'].mean():,.0f}")
        print(f"  avg kelly ROI: {active['kelly_roi'].mean():.1%}")
        print(f"  avg max drawdown: ${active['kelly_maxdd'].mean():,.0f}")
        print(f"  worst season profit: ${active['kelly_profit'].min():,.0f}")
        print(f"  best season profit:  ${active['kelly_profit'].max():,.0f}")

    out = REPO / "Data" / "backtest_kelly_per_season.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
