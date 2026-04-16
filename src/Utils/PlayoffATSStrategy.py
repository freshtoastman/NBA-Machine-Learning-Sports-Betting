"""Playoff ATS betting strategy — rule-based signal combiner.

Backtested on 13 seasons (2012-2025) of playoff data.  The signals below
survived walk-forward validation and target the structural inefficiencies
of playoff spreads:

  * Markets overvalue regular-season dominance → favorites only cover 46.5%
  * Elimination-game intensity compresses margins → underdogs cover 57.5%
  * ML model confidence on small spreads is highly predictive (80% WR)
  * ATS cold streaks in regular season reverse in playoffs (regression)

Each signal has a name, a required minimum sample size from backtest, and
a tier (GOLD / SILVER / BRONZE) that maps to suggested Kelly fractions.

Usage:
    from src.Utils.PlayoffATSStrategy import evaluate_playoff_ats
    picks = evaluate_playoff_ats(pred, series_state, team_form)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PlayoffATSPick:
    """A single actionable ATS recommendation."""
    signal_name: str       # human-readable rule name
    side: str              # 'home' | 'away'
    ats_side: str          # '讓分(押fav)' | '受讓(押dog)'
    tier: str              # 'GOLD' | 'SILVER' | 'BRONZE'
    backtest_wr: float     # win rate from backtest (0-1)
    backtest_roi: float    # ROI% from backtest
    backtest_n: int        # sample size
    reason_zh: str         # Chinese explanation for the UI


# ---------------------------------------------------------------------------
# Signal definitions — each returns a PlayoffATSPick or None
# ---------------------------------------------------------------------------

def _ml_high_conf_small_spread(pred, series_state, team_form) -> PlayoffATSPick | None:
    """ML model ≥65% confidence + |spread| ≤ 5 → bet the ML side to cover.

    Backtest: 80% WR, +52.7% ROI, n=15 (3 seasons).
    Full 13-season sample shows consistent edge at 60%+ threshold.
    """
    spread = pred.get("spread")
    if spread is None or abs(spread) > 5:
        return None
    home_conf = pred.get("home_confidence", 50)
    away_conf = pred.get("away_confidence", 50)
    # Which side does the ML model favor with ≥65%?
    if home_conf >= 65:
        side = "home"
    elif away_conf >= 65:
        side = "away"
    else:
        return None
    home_fav = spread > 0
    if side == "home":
        ats_side = "讓分(押fav)" if home_fav else "受讓(押dog)"
    else:
        ats_side = "受讓(押dog)" if home_fav else "讓分(押fav)"
    return PlayoffATSPick(
        signal_name="ML高信心+小讓分",
        side=side,
        ats_side=ats_side,
        tier="GOLD",
        backtest_wr=0.80,
        backtest_roi=52.7,
        backtest_n=15,
        reason_zh=f"模型對{'主' if side == 'home' else '客'}場信心 ≥65%，且讓分 ≤5，回測勝率 80%",
    )


def _ml_moderate_conf_small_spread(pred, series_state, team_form) -> PlayoffATSPick | None:
    """ML model 60-65% confidence + |spread| ≤ 5 → bet the ML side.

    Backtest: 66.7% WR, +27.3% ROI, n=33.
    """
    spread = pred.get("spread")
    if spread is None or abs(spread) > 5:
        return None
    home_conf = pred.get("home_confidence", 50)
    away_conf = pred.get("away_confidence", 50)
    # 60-65 range — not caught by the ≥65 signal above
    if 60 <= home_conf < 65:
        side = "home"
    elif 60 <= away_conf < 65:
        side = "away"
    else:
        return None
    home_fav = spread > 0
    if side == "home":
        ats_side = "讓分(押fav)" if home_fav else "受讓(押dog)"
    else:
        ats_side = "受讓(押dog)" if home_fav else "讓分(押fav)"
    return PlayoffATSPick(
        signal_name="ML中信心+小讓分",
        side=side,
        ats_side=ats_side,
        tier="SILVER",
        backtest_wr=0.667,
        backtest_roi=27.3,
        backtest_n=33,
        reason_zh=f"模型對{'主' if side == 'home' else '客'}場信心 60-65%，讓分 ≤5，回測勝率 66.7%",
    )


def _elimination_underdog(pred, series_state, team_form) -> PlayoffATSPick | None:
    """Elimination game → bet the underdog to cover.

    Backtest: 57.5% WR, +9.8% ROI, n=40 (3 seasons).
    13-season data: favorites cover only 32.7% in Game 6.
    """
    if series_state is None:
        return None
    if series_state.get("round_num", 0) == 0:
        return None  # Play-In: no series elimination context
    if not series_state.get("is_elimination"):
        return None
    spread = pred.get("spread")
    if spread is None:
        return None
    home_fav = spread > 0
    side = "away" if home_fav else "home"
    return PlayoffATSPick(
        signal_name="淘汰局押冷門",
        side=side,
        ats_side="受讓(押dog)",
        tier="SILVER",
        backtest_wr=0.575,
        backtest_roi=9.8,
        backtest_n=40,
        reason_zh="淘汰局中冷門球隊拼命打，回測勝率 57.5%",
    )


def _complacent_leader(pred, series_state, team_form) -> PlayoffATSPick | None:
    """Home team leads series 2+ → away covers (complacency effect).

    Backtest: 69.2% WR, +32.2% ROI, n=13 (3 seasons).
    13-season: when leader is up 2+, the trailing team competes harder ATS.
    """
    if series_state is None:
        return None
    if series_state.get("round_num", 0) == 0:
        return None  # Play-In: no series lead context
    home_wins = series_state.get("home_wins", 0)
    away_wins = series_state.get("away_wins", 0)
    lead = home_wins - away_wins
    if lead < 2:
        return None
    spread = pred.get("spread")
    home_fav = spread is not None and spread > 0
    return PlayoffATSPick(
        signal_name="主場大幅領先鬆懈",
        side="away",
        ats_side="受讓(押dog)" if home_fav else "讓分(押fav)",
        tier="SILVER",
        backtest_wr=0.692,
        backtest_roi=32.2,
        backtest_n=13,
        reason_zh=f"主場領先 {home_wins}-{away_wins}，領先方鬆懈，客場拼搶可 cover",
    )


def _trailing_home_desperation(pred, series_state, team_form) -> PlayoffATSPick | None:
    """Home trails series 2+ → home covers (desperation at home).

    Backtest: 57.6% WR, +9.9% ROI, n=33.
    """
    if series_state is None:
        return None
    if series_state.get("round_num", 0) == 0:
        return None  # Play-In: no series trail context
    home_wins = series_state.get("home_wins", 0)
    away_wins = series_state.get("away_wins", 0)
    lead = home_wins - away_wins
    if lead > -2:
        return None
    spread = pred.get("spread")
    home_fav = spread is not None and spread > 0
    return PlayoffATSPick(
        signal_name="主場落後拼搶",
        side="home",
        ats_side="讓分(押fav)" if home_fav else "受讓(押dog)",
        tier="BRONZE",
        backtest_wr=0.576,
        backtest_roi=9.9,
        backtest_n=33,
        reason_zh=f"主場系列賽落後 {home_wins}-{away_wins}，背水一戰拼搶 cover",
    )


def _ats_cold_bounce(pred, series_state, team_form) -> PlayoffATSPick | None:
    """Team's last 10 regular-season ATS ≤ 30% → bounce back in playoffs.

    Backtest: 75% WR, +43.2% ROI, n=20 (home side).
    Regression to the mean: cold ATS streaks don't persist into playoffs.
    """
    if team_form is None:
        return None
    h_ats = team_form.get("home_ats_l10")
    a_ats = team_form.get("away_ats_l10")
    spread = pred.get("spread")
    # Home team bouncing back
    if h_ats is not None and h_ats <= 0.30:
        home_fav = spread is not None and spread > 0
        return PlayoffATSPick(
            signal_name="主場ATS冷門反彈",
            side="home",
            ats_side="讓分(押fav)" if home_fav else "受讓(押dog)",
            tier="GOLD",
            backtest_wr=0.75,
            backtest_roi=43.2,
            backtest_n=20,
            reason_zh=f"主場近10場 ATS 僅 {h_ats*100:.0f}%，季後賽反彈回測勝率 75%",
        )
    if a_ats is not None and a_ats <= 0.30:
        home_fav = spread is not None and spread > 0
        return PlayoffATSPick(
            signal_name="客場ATS冷門反彈",
            side="away",
            ats_side="受讓(押dog)" if home_fav else "讓分(押fav)",
            tier="SILVER",
            backtest_wr=0.60,
            backtest_roi=14.5,
            backtest_n=15,
            reason_zh=f"客場近10場 ATS 僅 {a_ats*100:.0f}%，季後賽反彈回測勝率 60%",
        )
    return None


def _evenly_matched_home(pred, series_state, team_form) -> PlayoffATSPick | None:
    """When teams are evenly matched (form diff < 5%), home covers.

    Backtest: 63.5% WR, +21.2% ROI, n=52.
    Home court matters more in close matchups.
    """
    if team_form is None:
        return None
    h_form = team_form.get("home_w20", 0.5)
    a_form = team_form.get("away_w20", 0.5)
    if abs(h_form - a_form) >= 0.05:
        return None
    spread = pred.get("spread")
    home_fav = spread is not None and spread > 0
    return PlayoffATSPick(
        signal_name="實力接近主場優勢",
        side="home",
        ats_side="讓分(押fav)" if home_fav else "受讓(押dog)",
        tier="SILVER",
        backtest_wr=0.635,
        backtest_roi=21.2,
        backtest_n=52,
        reason_zh="雙方例行賽戰績差距 <5%，主場優勢在季後賽更關鍵",
    )


def _medium_spread_dog(pred, series_state, team_form) -> PlayoffATSPick | None:
    """5.5 ≤ |spread| < 8 → bet underdog (13-season: 60.9% WR, +16.3% ROI).

    Markets over-adjust medium spreads in playoffs.
    """
    spread = pred.get("spread")
    if spread is None:
        return None
    abs_sp = abs(spread)
    if abs_sp < 5.5 or abs_sp >= 8:
        return None
    home_fav = spread > 0
    side = "away" if home_fav else "home"
    return PlayoffATSPick(
        signal_name="中等讓分押冷門",
        side=side,
        ats_side="受讓(押dog)",
        tier="BRONZE",
        backtest_wr=0.609,
        backtest_roi=16.3,
        backtest_n=64,
        reason_zh=f"讓分 {abs_sp:.1f} 在 5.5-8 區間，市場高估熱門，冷門 cover 率 60.9%",
    )


# Ordered from highest confidence to lowest
_SIGNALS = [
    _ml_high_conf_small_spread,
    _ats_cold_bounce,
    _complacent_leader,
    _ml_moderate_conf_small_spread,
    _evenly_matched_home,
    _elimination_underdog,
    _medium_spread_dog,
    _trailing_home_desperation,
]


def evaluate_playoff_ats(
    pred: dict,
    series_state: dict | None = None,
    team_form: dict | None = None,
) -> list[PlayoffATSPick]:
    """Run all playoff ATS signals and return matching picks.

    Parameters
    ----------
    pred : dict
        Prediction dict from predict_today_xgb / predict_historical_xgb.
        Must contain: home_confidence, away_confidence, spread.
    series_state : dict | None
        From PlayoffContext.get_series_state(). Contains series_game_num,
        home_wins, away_wins, is_elimination, etc.
    team_form : dict | None
        Rolling form stats:
            home_ats_l10  — home team's ATS cover rate in last 10 games
            away_ats_l10  — away team's ATS cover rate in last 10 games
            home_w20      — home team's win % in last 20 games
            away_w20      — away team's win % in last 20 games

    Returns
    -------
    list[PlayoffATSPick] — all matching signals, sorted by tier then WR.
    """
    picks = []
    for signal_fn in _SIGNALS:
        pick = signal_fn(pred, series_state, team_form)
        if pick is not None:
            picks.append(pick)
    # Sort: GOLD > SILVER > BRONZE, then by backtest win rate
    tier_order = {"GOLD": 0, "SILVER": 1, "BRONZE": 2}
    picks.sort(key=lambda p: (tier_order.get(p.tier, 9), -p.backtest_wr))
    return picks


def best_pick(picks: list[PlayoffATSPick]) -> PlayoffATSPick | None:
    """Return the single best pick, or None if no signals fire."""
    return picks[0] if picks else None


def consensus_side(picks: list[PlayoffATSPick]) -> str | None:
    """Return 'home' or 'away' if the majority of signals agree, else None."""
    if not picks:
        return None
    home_votes = sum(1 for p in picks if p.side == "home")
    away_votes = sum(1 for p in picks if p.side == "away")
    if home_votes > away_votes:
        return "home"
    if away_votes > home_votes:
        return "away"
    return None


def picks_to_dict(picks: list[PlayoffATSPick]) -> list[dict]:
    """Serialize picks for JSON export."""
    return [
        {
            "signal": p.signal_name,
            "side": p.side,
            "ats_side": p.ats_side,
            "tier": p.tier,
            "backtest_wr": p.backtest_wr,
            "backtest_roi": p.backtest_roi,
            "backtest_n": p.backtest_n,
            "reason_zh": p.reason_zh,
        }
        for p in picks
    ]
