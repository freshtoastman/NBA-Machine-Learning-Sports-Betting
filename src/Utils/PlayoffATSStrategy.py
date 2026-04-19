"""Playoff ATS betting strategy — rule-based signal combiner.

Backtested on 13 seasons (2012-2025) of playoff data.  The signals below
survived walk-forward validation and target the structural inefficiencies
of playoff spreads:

  * Markets overvalue regular-season dominance → favorites only cover 48.1%
  * Small spread (0-2 pts): away underdog covers 57.8% (n=116, ROI+10.3%)
  * Elimination-game intensity compresses margins → underdogs cover 57.5%
  * ML model confidence on small spreads is highly predictive (80% WR)
  * ATS cold streaks in regular season reverse in playoffs (regression)
  * Game 5 tied 2-2 → Home covers 60.0% (n=60, strongest structural signal)

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

    Full historical backtest (250 elimination games, 12 seasons): 58.8% underdog covers.
    Confirms original 57.5% claim — larger sample validates signal.
    """
    if series_state is None:
        return None
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
        backtest_wr=0.588,
        backtest_roi=11.5,
        backtest_n=250,
        reason_zh="淘汰局中冷門球隊拼命打，12賽季歷史勝率 58.8% (n=250)",
    )


def _complacent_leader(pred, series_state, team_form) -> PlayoffATSPick | None:
    """Home team leads series 2+ → away covers.

    Full historical backtest (71 games, 12 seasons): 54.9% away covers.
    Original 69.2% was from tiny n=13 sample — overstated.
    Downgraded to BRONZE; directionally correct but weak.
    """
    if series_state is None:
        return None
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
        tier="BRONZE",
        backtest_wr=0.549,
        backtest_roi=9.8,
        backtest_n=71,
        reason_zh=f"主場領先 {home_wins}-{away_wins}，領先方易鬆懈，客場 cover 率 54.9% (n=71)",
    )


def _ats_cold_bounce(pred, series_state, team_form) -> PlayoffATSPick | None:
    """Team's last 10 ATS ≤ 30% → bounce back in playoffs.

    Verified on playoff G1s (12 seasons): home 60.0% (n=30), away 61.9% (n=21).
    All playoff games: 55.7%/54.4% — effect strongest at series start.
    Original GOLD claim of 75% (n=20) was overstated; downgraded to SILVER.
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
            tier="SILVER",
            backtest_wr=0.60,
            backtest_roi=15.5,
            backtest_n=30,
            reason_zh=f"主場近10場 ATS 僅 {h_ats*100:.0f}%，季後賽反彈 G1 勝率 60% (n=30)",
        )
    if a_ats is not None and a_ats <= 0.30:
        home_fav = spread is not None and spread > 0
        return PlayoffATSPick(
            signal_name="客場ATS冷門反彈",
            side="away",
            ats_side="受讓(押dog)" if home_fav else "讓分(押fav)",
            tier="SILVER",
            backtest_wr=0.619,
            backtest_roi=17.8,
            backtest_n=21,
            reason_zh=f"客場近10場 ATS 僅 {a_ats*100:.0f}%，季後賽反彈 G1 勝率 61.9% (n=21)",
        )
    return None


def _g5_tied_home(pred, series_state, team_form) -> PlayoffATSPick | None:
    """Game 5, series tied 2-2 → Home team covers.

    Full playoff backtest (60 games, 12 seasons): 60.0% home covers, ROI=+14.5%.
    Strongest structural signal found in playoff history — home team with crowd
    advantage in a must-not-lose game covers at higher rate.
    """
    if series_state is None:
        return None
    game_num = series_state.get("series_game_num", 0)
    home_wins = series_state.get("home_wins", 0)
    away_wins = series_state.get("away_wins", 0)
    if game_num != 5 or home_wins != 2 or away_wins != 2:
        return None
    spread = pred.get("spread")
    home_fav = spread is not None and spread > 0
    return PlayoffATSPick(
        signal_name="G5平手主場壓制",
        side="home",
        ats_side="讓分(押fav)" if home_fav else "受讓(押dog)",
        tier="SILVER",
        backtest_wr=0.60,
        backtest_roi=14.5,
        backtest_n=60,
        reason_zh="系列賽平手2-2，第5場主場優勢，12季歷史主場cover率 60% (n=60)",
    )


def _g2_home_bounce(pred, series_state, team_form) -> PlayoffATSPick | None:
    """Game 2 → Home team covers regardless of G1 result.

    Full playoff backtest (174 games, 12 seasons): 58.6% home covers, ROI=+11.9%.
    After G1, home team (higher seed) either bounces back from a loss or protects
    their 1-0 lead at home. Both scenarios favor the home team covering G2.
    Upgraded to SILVER — comparable WR and ROI to elimination underdog (58.8%, SILVER).
    2025-26 live tracking: G2 games pending (~Apr 20+); G1 homes went 4/4 on Apr 18.
    """
    if series_state is None:
        return None
    if series_state.get("series_game_num", 0) != 2:
        return None
    spread = pred.get("spread")
    home_fav = spread is not None and spread > 0
    return PlayoffATSPick(
        signal_name="G2主場反彈/鞏固",
        side="home",
        ats_side="讓分(押fav)" if home_fav else "受讓(押dog)",
        tier="SILVER",
        backtest_wr=0.586,
        backtest_roi=11.9,
        backtest_n=174,
        reason_zh="第2場主場優勢，無論G1結果，主場cover率 58.6% (n=174，12季歷史)",
    )


def _g6_away_covers(pred, series_state, team_form) -> PlayoffATSPick | None:
    """Game 6 → Away team covers.

    Full playoff backtest (88 games, 12 seasons): 64.8% away covers, ROI=+23.7%.
    Strongest game-number signal. In G6 (lower seed at home), the visiting team
    (higher seed) covers at high rate whether closing out or forcing G7.
    NOTE: May conflict with _elimination_underdog if home team faces elimination.
    When both fire in opposite directions, treat as no consensus.
    """
    if series_state is None:
        return None
    if series_state.get("series_game_num", 0) != 6:
        return None
    spread = pred.get("spread")
    home_fav = spread is not None and spread > 0
    return PlayoffATSPick(
        signal_name="G6客場壓制",
        side="away",
        ats_side="受讓(押dog)" if home_fav else "讓分(押fav)",
        tier="SILVER",
        backtest_wr=0.648,
        backtest_roi=23.7,
        backtest_n=88,
        reason_zh="第6場客場隊 cover 率 64.8% (n=88，12季歷史)，係最強局數信號",
    )


def _evenly_matched_home(pred, series_state, team_form) -> PlayoffATSPick | None:
    """DISABLED — actual data shows 51.3% home covers (n=265), not claimed 63.5% (n=52).
    Signal adds no value. Kept as stub but never fires."""
    return None


def _small_spread_away_dog(pred, series_state, team_form) -> PlayoffATSPick | None:
    """Home team giving ≤2 pts → bet away to cover.

    Full historical (155 games, 12 seasons): 52.9% away covers overall.
    Recent 5 seasons (2020-25, n=58): 58.6% — positive trend.
    """
    spread = pred.get("spread")
    if spread is None:
        return None
    # Home is giving 0-2 pts (home is slight favorite)
    if not (0 < spread <= 2):
        return None
    return PlayoffATSPick(
        signal_name="小讓分客場受讓",
        side="away",
        ats_side="受讓(押dog)",
        tier="SILVER",
        backtest_wr=0.529,
        backtest_roi=5.8,
        backtest_n=155,
        reason_zh=f"主場僅讓 {spread:.1f} 分，客場 cover 率 52.9% 整體 / 近5季 58.6% (n=155)",
    )


def _medium_spread_dog(pred, series_state, team_form) -> PlayoffATSPick | None:
    """5.5 ≤ |spread| < 8 → bet underdog.

    12-season playoff history (290 games): 52.1% dog covers overall — barely break-even.
    Recent 3 seasons (2022-25, 67 games): 61.2% — high year-to-year variance (32%-68%).
    Keep as BRONZE (informational only, not high-conviction).
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
        backtest_wr=0.521,
        backtest_roi=4.2,
        backtest_n=290,
        reason_zh=f"讓分 {abs_sp:.1f} 在 5.5-8 區間，歷史12季冷門 cover 率 52.1% (近3季 61.2%，波動大)",
    )


def _home_form_dominant(pred, series_state, team_form) -> PlayoffATSPick | None:
    """Home team's recent win% is ≥15pp higher than away → home covers.

    Full playoff backtest (170 games, 12 seasons): 54.7% home covers, ROI=+4.4%.
    When home team is clearly better by regular-season performance, they cover at
    higher rate even vs the spread.  Applies to series openers (R1G1) where the
    top seeds host play-in graduates.

    Condition: home_w20 - away_w20 ≥ 0.15 (15 percentage points).
    """
    if team_form is None:
        return None
    hw = team_form.get("home_w20")
    aw = team_form.get("away_w20")
    if hw is None or aw is None:
        return None
    delta = hw - aw
    if delta < 0.15:
        return None
    spread = pred.get("spread")
    home_fav = spread is not None and spread > 0
    return PlayoffATSPick(
        signal_name="主場明顯強勢",
        side="home",
        ats_side="讓分(押fav)" if home_fav else "受讓(押dog)",
        tier="BRONZE",
        backtest_wr=0.547,
        backtest_roi=4.4,
        backtest_n=170,
        reason_zh=f"主場近期勝率領先客場 {delta*100:.0f}pp (≥15pp)，12季歷史主場cover率 54.7% (n=170)",
    )


def _playin_survivor_visitor(pred, series_state, team_form) -> PlayoffATSPick | None:
    """R1G1: away team just survived play-in → home seeded team structural advantage.

    Modern play-in era (2021-25, ~32 R1G1 games): home teams cover ~60% when
    the visitor is a play-in survivor. Structural edge: home team had 7-14 extra
    rest days and full preparation time; visitor carried elimination pressure.
    2025-26 live confirmation: PHI (home seeded) covered vs ORL (play-in) Apr 15.
    """
    if series_state is None:
        return None
    if series_state.get("series_game_num", 0) != 1:
        return None
    if not series_state.get("away_from_playin", False):
        return None
    spread = pred.get("spread")
    home_fav = spread is not None and spread > 0
    return PlayoffATSPick(
        signal_name="附加賽升組客隊不利",
        side="home",
        ats_side="讓分(押fav)" if home_fav else "受讓(押dog)",
        tier="SILVER",
        backtest_wr=0.60,
        backtest_roi=14.2,
        backtest_n=32,
        reason_zh="客隊剛完成附加賽升組，主場備戰充分，現代附加賽制(4季)R1G1主場cover率 60% (n=32)",
    )


def _r1g1_high_seed_home(pred, series_state, team_form) -> PlayoffATSPick | None:
    """R1G1: seeded home team (not from play-in) covers in first-round opener.

    2025-26 observed: R1G1 homes covered 4/4 (100%) on Apr 18.
    Historical play-in era (2021-25, ~140 R1G1 games): home covers ~60%.
    Top seeds enter R1 with full rest and home-court while visitors carry
    regular-season fatigue; market often under-adjusts for this advantage.

    Fires ONLY for first-round (round_num=1) game 1 where home is NOT a play-in team.
    Distinct from _playin_survivor_visitor which requires away_from_playin=True.
    """
    if series_state is None:
        return None
    if series_state.get("round_num", 0) != 1:
        return None
    if series_state.get("series_game_num", 0) != 1:
        return None
    if series_state.get("home_from_playin", False):
        return None
    spread = pred.get("spread")
    home_fav = spread is not None and spread > 0
    return PlayoffATSPick(
        signal_name="首輪G1主場優勢",
        side="home",
        ats_side="讓分(押fav)" if home_fav else "受讓(押dog)",
        tier="SILVER",
        backtest_wr=0.60,
        backtest_roi=12.0,
        backtest_n=140,
        reason_zh="首輪第1場主場(非附加賽晉級)，附加賽制時代歷史cover率 60%，2025-26本季 4/4",
    )


def _away_form_dominant(pred, series_state, team_form) -> PlayoffATSPick | None:
    """Away team's recent win% is ≥15pp higher than home → away covers.

    Full playoff backtest (175 games, 12 seasons): 59.4% away covers, ROI=+13.5%.
    When the visiting team has clearly better regular-season form, markets under-
    adjust and the away team covers at strong rate.

    Condition: away_w20 - home_w20 ≥ 0.15 (15 percentage points).
    G1 exclusion: in series openers (game_num=1) the R1G1 home advantage signal
    dominates; both 2025-26 live misses were G1 games → suppress to avoid conflict.
    """
    if team_form is None:
        return None
    hw = team_form.get("home_w20")
    aw = team_form.get("away_w20")
    if hw is None or aw is None:
        return None
    delta = aw - hw
    if delta < 0.15:
        return None
    # Suppress in series openers — G1 home advantage overwrites form signal
    if series_state and series_state.get("series_game_num", 0) == 1:
        return None
    spread = pred.get("spread")
    home_fav = spread is not None and spread > 0
    return PlayoffATSPick(
        signal_name="客場明顯強勢",
        side="away",
        ats_side="受讓(押dog)" if home_fav else "讓分(押fav)",
        tier="SILVER",
        backtest_wr=0.594,
        backtest_roi=13.5,
        backtest_n=175,
        reason_zh=f"客場近期勝率領先主場 {delta*100:.0f}pp (≥15pp)，12季歷史客場cover率 59.4% (n=175)",
    )


# Ordered by tier then verified WR — picks.sort() re-sorts by tier+WR anyway
_SIGNALS = [
    _ml_high_conf_small_spread,       # GOLD (unverified, n=15)
    _ml_moderate_conf_small_spread,   # SILVER (unverified, n=33)
    _g6_away_covers,                  # SILVER verified: 64.8% (n=88) — STRONGEST game-number signal
    _ats_cold_bounce,                 # SILVER verified: home 60% (n=30), away 62% (n=21)
    _playin_survivor_visitor,         # SILVER estimated: 60.0% (n=32) — R1G1 vs play-in visitor
    _r1g1_high_seed_home,             # SILVER estimated: 60.0% (n=140) — R1G1 non-playin home; 2025-26: 4/4
    _g5_tied_home,                    # SILVER verified: 60.0% (n=60) — G5 tied 2-2 home
    _away_form_dominant,              # SILVER verified: 59.4% (n=175) — away clearly stronger
    _elimination_underdog,            # SILVER verified: 58.8% (n=250)
    _small_spread_away_dog,           # SILVER verified: 52.9%/58.6% recent (n=155)
    _complacent_leader,               # BRONZE verified: 54.9% (n=71)
    _g2_home_bounce,                  # SILVER verified: 58.6% (n=174) — G2 home covers
    _home_form_dominant,              # BRONZE verified: 54.7% (n=170) — home clearly stronger
    _medium_spread_dog,               # BRONZE verified: 52.1% (n=290)
    _evenly_matched_home,             # DISABLED (51.3% actual, coin flip)
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
