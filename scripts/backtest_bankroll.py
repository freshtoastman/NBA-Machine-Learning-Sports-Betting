"""2024-25 regular season bankroll backtest for ATS value picks.

Runs the production ATS value strategy (asymmetric threshold: home >=8%, away >=9%
Oct-Feb, suppressed Mar/early-Apr, >=10% playoffs) over the 2024-25 season and
simulates several bankroll strategies side-by-side:

  - Flat $1000 per bet (baseline)
  - Flat $1000 with confidence tiers (edge >= 12% => 3x)
  - Quarter-Kelly sizing
  - Half-Kelly sizing
  - Martingale (double after loss)  -- demonstrates ruin

Odds assumed -110 (1.909 decimal) both sides unless a recorded ML is present.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from main import predict_historical_xgb  # noqa: E402


SEASON_START = date(2024, 10, 22)
SEASON_END = date(2025, 4, 13)  # regular season only (playoffs start Apr 15)
PLAYOFF_START = date(2025, 4, 15)
PLAYOFF_END = date(2025, 6, 22)

# User-proposed martingale ladder
USER_LADDER = [1000, 3000, 8000, 24000, 72000, 208000]

# -110 American odds = risk 110 to win 100; decimal = 1.9091; net profit per $1 = 0.9091
NET_WIN_PER_DOLLAR = 100.0 / 110.0  # 0.9091


def iter_dates(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def collect_value_picks(start: date, end: date) -> list[dict]:
    picks = []
    for d in iter_dates(start, end):
        try:
            preds = predict_historical_xgb(d)
        except Exception as e:
            print(f"[warn] {d}: {e}", file=sys.stderr)
            continue
        for p in preds:
            if not p.get("ats_is_value"):
                continue
            if p.get("ats_winner") is None:  # not played / no grade
                continue
            pick_side = p.get("ats_model_pick")
            edge = p.get("ats_value_edge") or 0.0
            cover = p.get("ats_winner")
            # Push = refund; skip for W/L accounting
            if cover == "push":
                continue
            win = (pick_side == cover)
            picks.append({
                "date": d.isoformat(),
                "home": p.get("home_team"),
                "away": p.get("away_team"),
                "pick": pick_side,
                "edge": edge,
                "spread": p.get("spread"),
                "win": win,
            })
    return picks


def kelly_fraction(p_win: float, net_odds: float = NET_WIN_PER_DOLLAR) -> float:
    """Kelly fraction: f* = (bp - q) / b where b = net odds (0.9091 at -110)."""
    q = 1.0 - p_win
    f = (net_odds * p_win - q) / net_odds
    return max(f, 0.0)


def simulate(picks: list[dict], strategy: str, bankroll: float, min_bet: float = 1000.0):
    """Run a strategy over picks list and return per-bet log + final bankroll."""
    br = bankroll
    peak = bankroll
    max_dd = 0.0
    log = []
    losing_streak = 0
    last_loss_stake = 0.0
    cum_loss_for_rule = 0.0

    for i, pk in enumerate(picks):
        edge_pp = pk["edge"]
        # Model prob = 0.5 + edge/100
        p = 0.5 + edge_pp / 100.0

        if strategy == "flat":
            stake = min_bet
        elif strategy == "tiered":
            # edge 8-10%: 1x, 10-12%: 2x, >=12%: 3x
            if edge_pp >= 12.0:
                stake = min_bet * 3
            elif edge_pp >= 10.0:
                stake = min_bet * 2
            else:
                stake = min_bet
        elif strategy == "quarter_kelly":
            f = kelly_fraction(p) * 0.25
            stake = max(br * f, min_bet)
        elif strategy == "half_kelly":
            f = kelly_fraction(p) * 0.50
            stake = max(br * f, min_bet)
        elif strategy == "martingale":
            if losing_streak == 0:
                stake = min_bet
            else:
                # next bet recovers all prior losses + min profit target
                stake = last_loss_stake * 2
        elif strategy == "user_ladder":
            # User-proposed: 1000, 3000, 8000, 24000, 72000, 208000
            level = min(losing_streak, len(USER_LADDER) - 1)
            stake = USER_LADDER[level]
        elif strategy == "sum2x":
            # User's clarified rule: stake = 2 * cumulative losses; start at $1000
            if losing_streak == 0:
                stake = min_bet
                cum_loss_for_rule = 0.0
            else:
                stake = max(2 * cum_loss_for_rule, min_bet)
        else:
            raise ValueError(strategy)

        # Cap stake at bankroll (can't bet what you don't have)
        stake = min(stake, br)
        if stake <= 0:
            # Ruined
            log.append({"i": i, **pk, "stake": 0, "pnl": 0, "br": br, "ruined": True})
            break

        if pk["win"]:
            pnl = stake * NET_WIN_PER_DOLLAR
            losing_streak = 0
            last_loss_stake = 0.0
            cum_loss_for_rule = 0.0
        else:
            pnl = -stake
            losing_streak += 1
            last_loss_stake = stake
            cum_loss_for_rule += stake

        br += pnl
        peak = max(peak, br)
        dd = peak - br
        max_dd = max(max_dd, dd)
        log.append({"i": i, **pk, "stake": round(stake, 0), "pnl": round(pnl, 0), "br": round(br, 0)})

    return {
        "final_br": br,
        "profit": br - bankroll,
        "roi": (br - bankroll) / bankroll,
        "peak": peak,
        "max_drawdown": max_dd,
        "n_bets": len(log),
        "n_wins": sum(1 for r in log if r.get("win") is True),
        "log": log,
    }


def main():
    print("Collecting value picks from 2024-10-22 to 2025-04-13 (regular season)...")
    reg_picks = collect_value_picks(SEASON_START, SEASON_END)
    print(f"  regular season value picks (graded, non-push): {len(reg_picks)}")
    wins = sum(1 for p in reg_picks if p["win"])
    print(f"  W/L: {wins}-{len(reg_picks) - wins}  ({wins / max(len(reg_picks), 1):.1%})")

    # Edge distribution
    df = pd.DataFrame(reg_picks)
    if not df.empty:
        buckets = pd.cut(df["edge"], bins=[0, 9, 10, 12, 15, 100])
        print("\nEdge-bucket hit rate:")
        print(df.groupby(buckets, observed=True)["win"].agg(["count", "mean"]).rename(columns={"mean": "hit_rate"}))

    # Save raw picks
    out = REPO / "Data" / "backtest_2024_25_picks.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    print("\n==== Bankroll simulation (initial = $50,000) ====")
    strategies = ["flat", "quarter_kelly", "sum2x"]
    for s in strategies:
        r = simulate(reg_picks, s, bankroll=50_000, min_bet=1000)
        print(f"\n[{s}]  bets={r['n_bets']}  W={r['n_wins']}  profit=${r['profit']:,.0f}  ROI={r['roi']:.1%}  peak=${r['peak']:,.0f}  maxDD=${r['max_drawdown']:,.0f}")
        if any(row.get("ruined") for row in r["log"]):
            print(f"  !! RUINED at bet {r['n_bets']}")

    # Show longest losing streak
    if reg_picks:
        cur, longest = 0, 0
        for p in reg_picks:
            if p["win"]:
                cur = 0
            else:
                cur += 1
                longest = max(longest, cur)
        print(f"\nLongest losing streak in season: {longest} consecutive losses")

    # Monte Carlo: what happens across many synthetic seasons at same WR,
    # assuming picks are independent Bernoulli(p_hit). 2024-25 had zero
    # back-to-back losses — that's unusually lucky variance.
    import random
    random.seed(42)
    p_hit = wins / max(len(reg_picks), 1)
    n_picks = len(reg_picks)
    print(f"\n==== Monte Carlo: 10,000 synthetic seasons ({n_picks} picks, hit-rate={p_hit:.1%}) ====")
    mc_results = {s: [] for s in ["flat", "sum2x"]}
    mc_ruined = {s: 0 for s in mc_results}
    mc_max_streak = []
    for trial in range(10_000):
        synth = [{"edge": 10.0, "win": random.random() < p_hit} for _ in range(n_picks)]
        cur_streak = 0
        longest_streak = 0
        for s in synth:
            if s["win"]:
                cur_streak = 0
            else:
                cur_streak += 1
                longest_streak = max(longest_streak, cur_streak)
        mc_max_streak.append(longest_streak)
        for strat in mc_results:
            r = simulate(synth, strat, bankroll=50_000, min_bet=1000)
            mc_results[strat].append(r["profit"])
            if any(row.get("ruined") for row in r["log"]) or r["final_br"] < 1000:
                mc_ruined[strat] += 1

    import statistics as st
    streak_buckets = {i: 0 for i in range(0, 10)}
    for ls in mc_max_streak:
        streak_buckets[min(ls, 9)] = streak_buckets.get(min(ls, 9), 0) + 1
    print("Longest-losing-streak distribution (per synthetic season):")
    for k, v in sorted(streak_buckets.items()):
        print(f"  streak={k}: {v / 100:.1f}%")

    for strat, profits in mc_results.items():
        profits.sort()
        print(f"\n[{strat}] MC profit distribution:")
        print(f"  mean=${st.mean(profits):,.0f}  median=${st.median(profits):,.0f}")
        print(f"  5th-pct=${profits[500]:,.0f}  95th-pct=${profits[9500]:,.0f}")
        print(f"  worst=${profits[0]:,.0f}  best=${profits[-1]:,.0f}")
        print(f"  ruin/broke rate: {mc_ruined[strat] / 100:.2f}%")


if __name__ == "__main__":
    main()
