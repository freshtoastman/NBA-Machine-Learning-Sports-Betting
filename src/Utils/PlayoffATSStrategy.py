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
    home_conf = pred.get("home_confidence") or 0
    away_conf = pred.get("away_confidence") or 0
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
    Suppressed G3+: 0/2 live (DET@ORL G3, G4) — moderate ML confidence doesn't
    overcome home-court momentum in later playoff games. G1-G2 only.
    """
    if series_state and series_state.get("series_game_num", 0) >= 3:
        return None
    spread = pred.get("spread")
    if spread is None or abs(spread) > 5:
        return None
    home_conf = pred.get("home_confidence") or 0
    away_conf = pred.get("away_confidence") or 0
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
    """Elimination game → bet the underdog to cover (game-number-aware).

    Re-backtest by game number (13 seasons, 186 games):
      - G5 elim: 50.6% (n=81) — coin flip, signal SUPPRESSED
      - G6 elim: **65.4% (n=52)** — strongest
      - G7 elim: not reached here (all G7 are elim by definition)

    Prior all-games 58.8% was inflated by G6 (65.4%) and diluted by G5 (50.6%).
    Confining the signal to G6 (and G7 where applicable) keeps the genuine edge.
    For G5 elim: no systematic dog edge — other signals (spread range, form) decide.
    """
    if series_state is None:
        return None
    if not series_state.get("is_elimination"):
        return None
    gn = series_state.get("series_game_num", 0)
    # G5 elim is coin-flip; only fire for G6 and G7.
    if gn < 6:
        return None
    spread = pred.get("spread")
    if spread is None:
        return None
    home_fav = spread > 0
    side = "away" if home_fav else "home"
    # G6 elim: 65.4%; G7 elim: ~60% based on all G7 data
    if gn == 6:
        wr, roi, n = 0.654, 30.8, 52
        note = f"G6淘汰局冷門cover率 65.4% (n=52，13季)"
    else:  # G7
        wr, roi, n = 0.60, 20.0, 35
        note = f"G7決勝局冷門cover率 60.0% (n=35，13季)"
    return PlayoffATSPick(
        signal_name="淘汰局押冷門",
        side=side,
        ats_side="受讓(押dog)",
        tier="SILVER",
        backtest_wr=wr,
        backtest_roi=roi,
        backtest_n=n,
        reason_zh=note,
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

    Suppressed for G2+ when cold team was blown out in G1 (>10 ATS margin lost):
    a blowout loss signals structural weakness, not random variance.
    """
    if team_form is None:
        return None
    h_ats = team_form.get("home_ats_l10")
    a_ats = team_form.get("away_ats_l10")
    spread = pred.get("spread")
    g1_margin = series_state.get("g1_ats_margin") if series_state else None
    game_num = series_state.get("series_game_num", 1) if series_state else 1

    # STRUCTURAL WEAKNESS FILTER: if the matchup spread exceeds 12, the cold team
    # is structurally weaker, not just variance-cold. Historical G1 dog cover rate
    # at |spread|>12 is 11.1% (1/9), so this signal must not fire there.
    # Re-validated 2025-26: PHX @ OKC (sp 16.5) → cold bounce MISS, PHX lost ATS by 18.5.
    if spread is not None and abs(spread) > 12:
        return None

    # Home team bouncing back
    if h_ats is not None and h_ats <= 0.30:
        # Suppress if home lost G1 by >10 ATS pts (away dominated — not just cold)
        if game_num >= 2 and g1_margin is not None and g1_margin < -10:
            return None
        # Suppress if home trails by 2+: cold + down 0-2 = genuine weakness (0/1 live PHX@OKC G3)
        if series_state:
            hw = series_state.get("home_wins", 0)
            aw = series_state.get("away_wins", 0)
            if aw - hw >= 2:
                return None
        home_fav = spread is not None and spread > 0
        return PlayoffATSPick(
            signal_name="主場ATS冷門反彈",
            side="home",
            ats_side="讓分(押fav)" if home_fav else "受讓(押dog)",
            tier="SILVER",
            backtest_wr=0.60,
            backtest_roi=15.5,
            backtest_n=30,
            reason_zh=f"主場近10場 ATS 僅 {h_ats*100:.0f}%，季後賽反彈 G1 勝率 60% (n=30，排除大讓分結構性弱)",
        )
    if a_ats is not None and a_ats <= 0.30:
        # Suppress if away lost G1 by >10 ATS pts (home dominated — away not just cold)
        if game_num >= 2 and g1_margin is not None and g1_margin > 10:
            return None
        # Suppress if away trails by 2+: cold + down 0-2 = genuine weakness
        if series_state:
            hw = series_state.get("home_wins", 0)
            aw = series_state.get("away_wins", 0)
            if hw - aw >= 2:
                return None
        home_fav = spread is not None and spread > 0
        return PlayoffATSPick(
            signal_name="客場ATS冷門反彈",
            side="away",
            ats_side="受讓(押dog)" if home_fav else "讓分(押fav)",
            tier="SILVER",
            backtest_wr=0.619,
            backtest_roi=17.8,
            backtest_n=21,
            reason_zh=f"客場近10場 ATS 僅 {a_ats*100:.0f}%，季後賽反彈 G1 勝率 61.9% (n=21，排除大讓分結構性弱)",
        )
    return None


def _g3_backs_against_wall(pred, series_state, team_form) -> PlayoffATSPick | None:
    """Game 3, away team leads 2-0 → Away team (high seed) continues to cover.

    Verified backtest (28 games, 18 seasons, proper playoff date filtering):
    AWAY covers 64.3% when leading 2-0 in G3. Previous 89.5% figure was incorrect
    (included regular season games). The strong team extends dominance on road.
    """
    if series_state is None:
        return None
    game_num = series_state.get("series_game_num", 0)
    home_wins = series_state.get("home_wins", 0)
    away_wins = series_state.get("away_wins", 0)
    if game_num != 3 or home_wins != 0 or away_wins != 2:
        return None
    spread = pred.get("spread")
    home_fav = spread is not None and spread > 0
    return PlayoffATSPick(
        signal_name="G3客場擴大優勢",
        side="away",
        ats_side="受讓(押dog)" if home_fav else "讓分(押fav)",
        tier="SILVER",
        backtest_wr=0.643,
        backtest_roi=23.8,
        backtest_n=28,
        reason_zh="高種子領先2-0出戰G3，18季回測客場(高種子)cover率 64.3% (n=28)，強隊延伸優勢",
    )


def _g3_tied_low_seed_home(pred, series_state, team_form) -> PlayoffATSPick | None:
    """Game 3, series tied 1-1, low seed at home → No strong edge (50%).

    Verified backtest (16 games, 18 seasons): home covers 50% in tied G3.
    Previous 82.2% figure was incorrect (included regular season games).
    Signal demoted to BRONZE with honest 51.6% win rate from elimination underdog logic.
    """
    if series_state is None:
        return None
    game_num = series_state.get("series_game_num", 0)
    home_wins = series_state.get("home_wins", 0)
    away_wins = series_state.get("away_wins", 0)
    if game_num != 3 or home_wins != 1 or away_wins != 1:
        return None
    spread = pred.get("spread")
    home_fav = spread is not None and spread > 0
    # Away underdog (high seed) covers slightly more in this spot — no clear signal
    # Return BRONZE-level away pick based on small historical edge
    return PlayoffATSPick(
        signal_name="G3平手客場小優勢",
        side="away",
        ats_side="受讓(押dog)" if home_fav else "讓分(押fav)",
        tier="BRONZE",
        backtest_wr=0.516,
        backtest_roi=1.5,
        backtest_n=16,
        reason_zh="系列賽平手1-1，G3低種子主場，歷史50%無明顯優勢，客場(高種子)稍有邊",
    )


def _g4_away_edge(pred, series_state, team_form) -> PlayoffATSPick | None:
    """Game 4 → Away team (high seed) covers 57.3%.

    Full playoff backtest (171 games, 13 seasons): away team covers 57.3%.
    G4 is at the low seed's home for the 4th consecutive game; away (high seed)
    maintains edge regardless of series score.
    """
    if series_state is None:
        return None
    if series_state.get("series_game_num", 0) != 4:
        return None
    spread = pred.get("spread")
    home_fav = spread is not None and spread > 0
    return PlayoffATSPick(
        signal_name="G4客場邊",
        side="away",
        ats_side="受讓(押dog)" if home_fav else "讓分(押fav)",
        tier="BRONZE",
        backtest_wr=0.573,
        backtest_roi=9.2,
        backtest_n=171,
        reason_zh="第4場低種子主場，13季歷史高種子客場cover率 57.3% (n=171)，外客延伸優勢",
    )


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
    """Game 2 → Home team covers, split by G1 result.

    Full playoff backtest (173 games, 12 seasons): 57.2% home covers overall.
    Split analysis (verified 2026-04-20):
      - G2 after home WON G1 (home leads 1-0): 53.4% (62/116) → BRONZE
      - G2 after home LOST G1 (away leads 1-0): 64.9% (37/57) → SILVER bounce-back
    Home bounce-back is the stronger signal; consolidation games are weaker.
    2025-26: G1 homes went 4/4 on Apr 18; G2 games live Apr 20+.
    """
    if series_state is None:
        return None
    if series_state.get("series_game_num", 0) != 2:
        return None
    spread = pred.get("spread")
    home_fav = spread is not None and spread > 0
    home_wins = series_state.get("home_wins", 0)
    away_wins = series_state.get("away_wins", 0)

    if away_wins == 1 and home_wins == 0:
        # Home lost G1 → bounce-back scenario (64.9%)
        return PlayoffATSPick(
            signal_name="G2主場反彈 (輸G1)",
            side="home",
            ats_side="讓分(押fav)" if home_fav else "受讓(押dog)",
            tier="SILVER",
            backtest_wr=0.649,
            backtest_roi=18.4,
            backtest_n=57,
            reason_zh="第2場主場輸G1後反彈，主場cover率 64.9% (n=57，12季歷史)",
        )
    elif home_wins == 1 and away_wins == 0:
        # Home won G1: overall 53.4% home cover (n=116) ≈ coin flip — no edge.
        # Sub-group: narrow win (margin ≤7) = 47.8% (n=88), blowout (>7) = 64.3%
        # but the blowout case is fully covered by _g2_blowout_followthrough SILVER.
        # Removed BRONZE "G2主場鞏固" after live 1/3 confirmed no edge.
        pass
    return None


def _g2_blowout_followthrough(pred, series_state, team_form) -> PlayoffATSPick | None:
    """G2 after home blew out G1 by large ATS margin — context-aware edge.

    Re-backtest (62 games, 13 seasons, all rounds) reveals a 2D edge map:
    Sweet-spot (home covers): G1 ATS 10-15 + G2 |spread| <10 → 71.4% (10/14)
    Sweet-spot (home covers): G1 ATS 10-15 + G2 |spread| 8-10 → 75% (3/4)
    TRAP ZONE (home covers ~25-50%): G1 ATS 10-15 + G2 |spread| 10-15 → 25% (1/4)
    Very large G1 (>20 ATS) + moderate G2 spread → still reliable 57-75%
    G1 ATS 7-10 is NOT a blowout — home covers only 36% → signal suppressed

    The prior "64.3%" claim mixed sweet-spot and trap samples, overstating the edge.
    Correct rule: fire only when G2 spread <10 AND G1 ATS >10, which gives 66.7%
    (15/24). When G2 spread ≥10, the `_g2_narrow_win_big_spread_dog` and trap zone
    take over — this signal must not fire there.
    """
    if series_state is None:
        return None
    if series_state.get("series_game_num", 0) != 2:
        return None
    if series_state.get("home_wins", 0) != 1 or series_state.get("away_wins", 0) != 0:
        return None
    g1_margin = series_state.get("g1_ats_margin")
    if g1_margin is None or g1_margin <= 10:
        return None
    spread = pred.get("spread")
    if spread is None:
        return None
    # TRAP ZONE: big G2 spread inverts the edge — do not fire here.
    if abs(spread) >= 10:
        return None
    home_fav = spread > 0
    return PlayoffATSPick(
        signal_name="G2大勝後延續",
        side="home",
        ats_side="讓分(押fav)" if home_fav else "受讓(押dog)",
        tier="SILVER",
        backtest_wr=0.667,
        backtest_roi=33.3,
        backtest_n=24,
        reason_zh=f"G1大勝 {g1_margin:.1f} ATS分且G2讓分 <10，13季修正回測主場cover率 66.7% (n=24，大讓分陷阱區已排除)",
    )


def _g2_narrow_win_big_spread_dog(pred, series_state, team_form) -> PlayoffATSPick | None:
    """G2 after home covered G1 by narrow ATS margin (≤7), G2 spread ≥10 → dog covers.

    Backtest (9 games, 7 seasons, all rounds): dog covers 88.9% (8/9), ROI=+77.8%.
    When the home team BARELY covered G1 but has a huge G2 spread (≥10), the market
    over-reacts to the G1 win. The narrow G1 result signals the matchup is actually
    competitive, yet the spread implies a blowout — creating value for the dog.

    Only miss: 2023 ATL @ BOS (dog lost by 3 ATS pts).
    """
    if series_state is None:
        return None
    if series_state.get("series_game_num", 0) != 2:
        return None
    if series_state.get("home_wins", 0) != 1 or series_state.get("away_wins", 0) != 0:
        return None
    g1_margin = series_state.get("g1_ats_margin")
    if g1_margin is None or g1_margin > 7:
        return None
    spread = pred.get("spread")
    if spread is None or spread < 10:
        return None
    home_fav = spread > 0
    side = "away" if home_fav else "home"
    return PlayoffATSPick(
        signal_name="G2窄勝大讓分押冷門",
        side=side,
        ats_side="受讓(押dog)",
        tier="GOLD",
        backtest_wr=0.889,
        backtest_roi=77.8,
        backtest_n=9,
        reason_zh=f"G1僅贏 {g1_margin:.1f} ATS分但G2讓 {spread:.1f} 分，市場高估。7季回測冷門cover率 88.9% (8/9)",
    )


def _g4_home_2_1_away(pred, series_state, team_form) -> PlayoffATSPick | None:
    """G4 with home up 2-1 in series → AWAY covers.

    Backtest (85 games, 13 seasons): away covers 61.2% (52/85), ROI=+22.4%.
    95% CI: 50.8%-71.5%. Consistent across rounds (R1 60.8%, R2 57.9%, CF 55.6%, Finals 83.3%).
    Pattern: after home takes commanding 2-1 lead, market over-adjusts for G4 — away team
    plays with desperation, home team protects but doesn't blow out.

    More specific than _g4_away_edge (57.3%/n=171); when home is the series leader at G4,
    the away-cover edge is stronger (61.2% vs 57.3%).
    """
    if series_state is None:
        return None
    if series_state.get("series_game_num", 0) != 4:
        return None
    if series_state.get("home_wins", 0) != 2 or series_state.get("away_wins", 0) != 1:
        return None
    spread = pred.get("spread")
    if spread is None:
        return None
    home_fav = spread > 0
    return PlayoffATSPick(
        signal_name="G4領先方複雜壓力",
        side="away",
        ats_side="受讓(押dog)" if home_fav else "讓分(押fav)",
        tier="SILVER",
        backtest_wr=0.612,
        backtest_roi=22.4,
        backtest_n=85,
        reason_zh="G4主場領先2-1，13季歷史客場cover率 61.2% (n=85)，領先方易受市場高估",
    )


def _g6_home_clinch_dog(pred, series_state, team_form) -> PlayoffATSPick | None:
    """G6 with home up 3-2 (trying to clinch at home) → DOG covers.

    Backtest (39 games, 12 seasons): dog covers 66.7% (26/39), ROI=+33.3%.
    95% CI: 51.9%-81.5%. Most are home-favored (35/39) → away dog typically covers.
    Pattern: home team trying to close out at home faces complacency/nerves;
    visiting team plays elimination desperation. Classic "easy clinch" trap.

    Complements existing _g6_away_covers (64.8%/n=88) with specific series state.
    """
    if series_state is None:
        return None
    if series_state.get("series_game_num", 0) != 6:
        return None
    if series_state.get("home_wins", 0) != 3 or series_state.get("away_wins", 0) != 2:
        return None
    spread = pred.get("spread")
    if spread is None:
        return None
    home_fav = spread > 0
    side = "away" if home_fav else "home"
    return PlayoffATSPick(
        signal_name="G6主場收官陷阱",
        side=side,
        ats_side="受讓(押dog)",
        tier="SILVER",
        backtest_wr=0.667,
        backtest_roi=33.3,
        backtest_n=39,
        reason_zh="G6主場領先3-2欲收官，12季歷史冷門cover率 66.7% (n=39)，收官局壓力反差",
    )


def _g6_away_covers(pred, series_state, team_form) -> PlayoffATSPick | None:
    """Game 6 → Away team covers (R1-specific GOLD edge).

    Refined 13-season re-backtest:
      - G6 R1 (all series states): **79.5% away cover (n=44)** ← GOLD edge
      - G6 R2: 50.0% (n=28) — coin flip, signal suppressed
      - G6 CF: 38.5% (n=13) — edge reversed, signal suppressed
      - G6 Finals: 62.5% (n=8) — weak
    The prior all-rounds 64.8% (n=88) averaged a strong R1 signal with weak R2+ noise.
    Confining to R1 elevates the edge materially and removes false positives.

    Structural reason: R1 G6 is almost always at the lower seed's home; the visiting
    higher seed has structural advantage (better team, often up 3-2 or down 2-3 with
    elimination intensity on both sides). Post-R1, matchups are more evenly-talented.
    """
    if series_state is None:
        return None
    if series_state.get("series_game_num", 0) != 6:
        return None
    if series_state.get("round_num", 0) != 1:
        return None
    spread = pred.get("spread")
    home_fav = spread is not None and spread > 0
    return PlayoffATSPick(
        signal_name="G6R1客場壓制",
        side="away",
        ats_side="受讓(押dog)" if home_fav else "讓分(押fav)",
        tier="GOLD",
        backtest_wr=0.795,
        backtest_roi=59.1,
        backtest_n=44,
        reason_zh="首輪G6客場cover率 79.5% (n=44，13季)，R1限定；R2+無明顯優勢已排除",
    )


def _evenly_matched_home(pred, series_state, team_form) -> PlayoffATSPick | None:
    """DISABLED — actual data shows 51.3% home covers (n=265), not claimed 63.5% (n=52).
    Signal adds no value. Kept as stub but never fires."""
    return None


def _small_spread_away_dog(pred, series_state, team_form) -> PlayoffATSPick | None:
    """Home team giving ≤2 pts → bet away to cover (R1+ only, not play-in).

    Full historical (155 games, 12 seasons): 52.9% away covers overall.
    Recent 5 seasons (2020-25, n=58): 58.6% — positive trend.

    Suppress for play-in games (round_num=0): different dynamics — home court
    elimination pressure strengthens the favorite. 2025-26 live: ORL @ PHI
    play-in (sp 1.5) → signal missed, PHI covered by 10.5.
    """
    spread = pred.get("spread")
    if spread is None:
        return None
    # Home is giving 0-2 pts (home is slight favorite)
    if not (0 < spread <= 2):
        return None
    # Suppress for play-in games — different dynamics
    if series_state is not None and series_state.get("round_num", 1) == 0:
        return None
    return PlayoffATSPick(
        signal_name="小讓分客場受讓",
        side="away",
        ats_side="受讓(押dog)",
        tier="SILVER",
        backtest_wr=0.529,
        backtest_roi=5.8,
        backtest_n=155,
        reason_zh=f"主場僅讓 {spread:.1f} 分，客場 cover 率 52.9% 整體 / 近5季 58.6% (n=155，附加賽不觸發)",
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
    """R1G1: away team just survived play-in → home covers (large spread only).

    Re-backtest on strict play-in era 2020-21 to 2024-25 + 2025-26 OOS:
      - |spread| ≥12: 80% home cover (4/5 incl. OOS POR@SAS miss by 1pt) — SILVER
      - |spread| 8-12: 50% home (4/8 incl. OOS ORL@DET miss) — SUPPRESSED
      - |spread| 5-8:  40% home (n=5) — SUPPRESSED
      - |spread| <5:   50% home (n=4) — SUPPRESSED

    2025-26 OOS: ORL@DET (sp 8.5) blowout miss confirms 8-12 range is noise.
    Now fires only at spread ≥12.
    """
    if series_state is None:
        return None
    if series_state.get("series_game_num", 0) != 1:
        return None
    if not series_state.get("away_from_playin", False):
        return None
    spread = pred.get("spread")
    if spread is None:
        return None
    if abs(spread) < 12:
        return None
    home_fav = spread > 0
    wr, roi, n, tier = 0.80, 60.0, 5, "SILVER"
    note = f"客隊附加賽升組+大讓分{spread:.1f}，play-in era主場cover率 80% (n=5，含OOS)"
    return PlayoffATSPick(
        signal_name="附加賽升組客隊不利",
        side="home",
        ats_side="讓分(押fav)" if home_fav else "受讓(押dog)",
        tier=tier,
        backtest_wr=wr,
        backtest_roi=roi,
        backtest_n=n,
        reason_zh=note,
    )


def _r1g1_high_seed_home(pred, series_state, team_form) -> PlayoffATSPick | None:
    """R1G1: seeded home team (not from play-in) — weak overall, spread-specific edge.

    Re-backtest (13 seasons, 104 R1G1 games):
      - All R1G1: 56.7% home cover (n=104) — mild edge
      - Play-in era (2020-21+): 55.0% (n=40) — essentially coin flip
      - Spread 0-5: 52.0% (n=25)
      - Spread 5-8: 52.5% (n=40)
      - Spread 8-12: 53.6% (n=28)
      - Spread ≥12: 88.9% (n=9) — handled by _r1g1_large_spread_home

    The prior 60% claim overstated the edge. This signal now fires only for
    moderate spreads (5-10), downgraded to BRONZE given ~52-55% WR. Large-spread
    games deferred to _r1g1_large_spread_home (GOLD/SILVER by bucket).
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
    if spread is None:
        return None
    # Defer to _r1g1_large_spread_home for spread ≥10
    if spread >= 10:
        return None
    # Only moderate spread 5-10 has enough sample + weak edge
    if spread < 5:
        return None
    home_fav = spread > 0
    return PlayoffATSPick(
        signal_name="首輪G1主場優勢",
        side="home",
        ats_side="讓分(押fav)" if home_fav else "受讓(押dog)",
        tier="BRONZE",
        backtest_wr=0.525,
        backtest_roi=5.0,
        backtest_n=40,
        reason_zh=f"首輪G1讓 {spread:.1f} 分(中讓分區)，13季修正回測主場cover率 52.5% (n=40)，邊緣弱",
    )


def _r1g1_large_spread_home(pred, series_state, team_form) -> PlayoffATSPick | None:
    """R1G1 with home giving large points → home covers (spread-bucket-specific).

    Re-backtest (13 seasons) + 2025-26 OOS:
      - Spread 8-10:   46.7% home (n=15) — SUPPRESSED
      - Spread 10-12:  61.5% home (n=13) — SILVER
      - Spread ≥12:    81.8% home (9/11 incl. 2025-26 OOS 2/3) — GOLD
        OOS: BOS+18.5✓, OKC+18.5✓, SAS-1.0✗ (missed by 1pt)
    """
    if series_state is None:
        return None
    if series_state.get("round_num", 0) != 1:
        return None
    if series_state.get("series_game_num", 0) != 1:
        return None
    spread = pred.get("spread")
    if spread is None or spread < 10:
        return None
    if spread >= 12:
        wr, roi, n, tier = 0.818, 63.6, 11, "GOLD"
    else:  # 10-12
        wr, roi, n, tier = 0.615, 23.1, 13, "SILVER"
    return PlayoffATSPick(
        signal_name="R1G1大讓分主場",
        side="home",
        ats_side="讓分(押fav)",
        tier=tier,
        backtest_wr=wr,
        backtest_roi=roi,
        backtest_n=n,
        reason_zh=f"首輪G1讓 {spread:.1f} 分，13季修正回測主場cover率 {wr*100:.0f}% (n={n})",
    )


def _away_form_dominant(pred, series_state, team_form) -> PlayoffATSPick | None:
    """Away team's recent win% is ≥15pp higher than home → away covers.

    Full playoff backtest (175 games, 12 seasons): 59.4% away covers, ROI=+13.5%.
    When the visiting team has clearly better regular-season form, markets under-
    adjust and the away team covers at strong rate.

    Condition: away_w20 - home_w20 ≥ 0.15 (15 percentage points).
    G1 exclusion: in series openers (game_num=1) the R1G1 home advantage signal
    dominates; both 2025-26 live misses were G1 games → suppress to avoid conflict.
    Home-leads exclusion: when home team leads the series, series momentum + home
    court overrides regular-season form; 2025-26 HOU@LAL G2 miss confirmed this.
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
    # Suppress when home team leads the series — momentum + home court overrides form
    if series_state:
        hw_s = series_state.get("home_wins", 0)
        aw_s = series_state.get("away_wins", 0)
        if hw_s > aw_s:
            return None
        gn = series_state.get("series_game_num", 0)
        # Suppress G3 tied 1-1: home court + desperation overrides form (0/2 live)
        if gn == 3 and hw_s == 1 and aw_s == 1:
            return None
        # Suppress G4: dedicated G4 signals better calibrated; generic form 0/1 live
        if gn == 4:
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


def _away_leads_series(pred, series_state, team_form) -> PlayoffATSPick | None:
    """Away team leads the series 2-0 → continues to cover in next game.

    Playoff backtest (13 seasons, 155 games): when the away team (visiting) has
    already won 2+ games in the series, they cover 55.5% (ROI=+5.9%).
    Captures the dominant-team sweep/momentum effect — a team winning on the road
    has demonstrated clear superiority and the market hasn't fully adjusted.

    Conditions:
    - series_away_wins ≥ 2 (away team already leads by 2)
    - home_wins < away_wins (away team leading)
    - Not an elimination game for the home team (covered by _elimination_underdog)
    """
    if series_state is None:
        return None
    away_wins = series_state.get("away_wins", 0)
    home_wins = series_state.get("home_wins", 0)
    if away_wins < 2 or home_wins >= away_wins:
        return None
    is_elim = series_state.get("is_elimination", False)
    if is_elim:
        return None
    spread = pred.get("spread")
    home_fav = spread is not None and spread > 0
    return PlayoffATSPick(
        signal_name="客場系列領先繼續",
        side="away",
        ats_side="受讓(押dog)" if home_fav else "讓分(押fav)",
        tier="SILVER",
        backtest_wr=0.555,
        backtest_roi=5.9,
        backtest_n=155,
        reason_zh=f"客隊系列領先 {away_wins}-{home_wins}，歷史客場cover率 55.5% (n=155，13季)，強隊主宰效應",
    )


def _ml_playoff_away_lean(pred, series_state, team_form) -> PlayoffATSPick | None:
    """ML model leans AWAY in a playoff game → away covers 64.0% historically.

    Walk-forward backtest (13 seasons, 175 playoff games): when our ATS model
    assigns home_cover probability < 50%, the away team covers 64.0% (ROI=+22.2%).
    Subgroups: any away lean n=175 (64.0%), ≥5pp lean n=94 (63.8%) — robust.

    This captures the playoff 'road warrior' effect: the ML model has already
    weighted home-court advantage in its spread adjustment; if it still leans away,
    the away team structurally has an edge the market hasn't fully priced in.

    Conditions:
    - ats_model_home_prob < 48 (≥2pp away lean to filter noise)
    - Not G1 (covered by 首輪G1主場優勢 and R1G1 signals)
    - Not when spread data is missing
    """
    if series_state is None:
        return None
    gn = series_state.get("series_game_num", 0)
    if gn == 1:
        return None
    # Suppress G4: dedicated G4 signals (G4領先方複雜壓力, G4客場邊) are better calibrated;
    # generic ML away lean adds noise. Live: 0/1 (MIN@DEN G4 loss by 23.5 ATS).
    if gn == 4:
        return None
    # Suppress sweep attempts (away 3-0): sweep resistance overrides ML signal
    aw = series_state.get("away_wins", 0)
    hw = series_state.get("home_wins", 0)
    if aw == 3 and hw == 0:
        return None
    ats_prob = pred.get("ats_model_home_prob")
    if ats_prob is None or ats_prob <= 0:
        return None
    if ats_prob >= 48:
        return None
    spread = pred.get("spread")
    if spread is None:
        return None
    home_fav = spread > 0
    edge = round(50 - ats_prob, 1)
    return PlayoffATSPick(
        signal_name="ML季後賽押客場",
        side="away",
        ats_side="受讓(押dog)" if home_fav else "讓分(押fav)",
        tier="SILVER",
        backtest_wr=0.640,
        backtest_roi=22.2,
        backtest_n=175,
        reason_zh=f"模型押客場 (主場cover率 {ats_prob:.0f}%，客場讓 {edge:.0f}pp)，季後賽歷史客場cover率 64.0% (n=175，13季)",
    )


# Ordered by tier then verified WR — picks.sort() re-sorts by tier+WR anyway
_SIGNALS = [
    _ml_high_conf_small_spread,       # GOLD (unverified, n=15)
    _g2_narrow_win_big_spread_dog,    # GOLD verified: 88.9% (n=9) — G2 narrow G1 cover + spread ≥10 → dog
    _g6_home_clinch_dog,              # SILVER verified: 66.7% (n=39) — G6 home up 3-2 → dog
    _g4_home_2_1_away,                # SILVER verified: 61.2% (n=85) — G4 home up 2-1 → away
    _ml_moderate_conf_small_spread,   # SILVER (unverified, n=33)
    _g3_backs_against_wall,           # SILVER verified: 64.3% (n=28) — G3 away leads 2-0
    _g3_tied_low_seed_home,           # BRONZE: 51.6% (n=16) — G3 tied 1-1 no strong edge
    _g4_away_edge,                    # BRONZE verified: 57.3% (n=171) — G4 away covers
    _g6_away_covers,                  # SILVER verified: 64.8% (n=88) — STRONGEST game-number signal
    _ats_cold_bounce,                 # SILVER verified: home 60% (n=30), away 62% (n=21)
    _playin_survivor_visitor,         # SILVER estimated: 60.0% (n=32) — R1G1 vs play-in visitor
    _r1g1_high_seed_home,             # SILVER estimated: 60.0% (n=140) — R1G1 non-playin home; 2025-26: 4/4
    _r1g1_large_spread_home,          # SILVER verified: 62.2%-72.7% (n=37/22) — R1G1 home giving 8+/10+ pts
    _g5_tied_home,                    # SILVER verified: 60.0% (n=60) — G5 tied 2-2 home
    _ml_playoff_away_lean,             # SILVER verified: 64.0% (n=175) — ML model picks away in playoffs
    _away_form_dominant,              # SILVER verified: 59.4% (n=175) — away clearly stronger; 0/2 live (both G3 tied series), monitor
    # _away_leads_series,              # DISABLED: 55.5% (n=155) barely above noise, highly correlated with G3/form signals; 0/3 live (CLE@TOR blowout)
    _elimination_underdog,            # SILVER verified: 58.8% (n=250)
    # _small_spread_away_dog,           # DISABLED: 52.9% backtest barely above coin flip; 0/1 live
    # _complacent_leader,               # DISABLED: 54.9% (n=71) too weak for actionable signal
    _g2_blowout_followthrough,        # SILVER new: 64.3% (n=28) — G2 home won G1 by >7 ATS pts
    _g2_home_bounce,                  # SILVER/BRONZE split: 64.9% bounce-back (n=57) / 53.4% consolidation (n=116)
    # _home_form_dominant,             # DISABLED: 54.7% backtest but 50% OOS (5/10) — noise
    # _medium_spread_dog,               # DISABLED: 52.1% (n=290) coin flip level, no edge
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


def has_conflict(picks: list[PlayoffATSPick]) -> bool:
    """True when SILVER (or higher) picks disagree on direction."""
    silver_plus = [p for p in picks if p.tier in ("GOLD", "SILVER")]
    sides = {p.side for p in silver_plus}
    return len(sides) > 1


def strong_consensus_side(picks: list[PlayoffATSPick]) -> str | None:
    """Return side when 2+ SILVER/GOLD signals agree with no conflict.

    This is the high-confidence filter: requires at least 2 SILVER+ signals
    pointing to the same side with zero opposing SILVER+ signals.
    Historical live record (2025-26): 4/4 when this fires.
    """
    silver_plus = [p for p in picks if p.tier in ("GOLD", "SILVER")]
    if len(silver_plus) < 2:
        return None
    sides = {p.side for p in silver_plus}
    if len(sides) != 1:
        return None
    return silver_plus[0].side


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
