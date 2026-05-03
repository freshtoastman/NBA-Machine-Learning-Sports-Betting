"""Self-contained backtest signal builder for the Vercel deployment.

Mirrors `src/Utils/AIAnalysis._build_backtest_signals` but has no
dependencies outside the `web/` directory, because the Vercel lambda
only ships `web/` — it cannot import from the repo-level `src/` package.

The game dict already carries `home_team_zh` / `away_team_zh` from the
export pipeline, so we read those instead of calling `team_name_zh`.
"""
from __future__ import annotations


def build_backtest_signals(game: dict) -> list[dict]:
    """Compute backtest-proven ATS strategy signals for a game.

    Each signal: {'text': str, 'tier': 'hot'|'warm'|'base'|'danger'}
    """
    signals: list[dict] = []
    spread = game.get("spread")
    if spread is None:
        return signals
    abs_sp = abs(float(spread))
    home_fav = float(spread) > 0

    h_conf = float(game.get("home_confidence", 50))
    a_conf = float(game.get("away_confidence", 50))
    model_conf = max(h_conf, a_conf)
    model_side = "home" if h_conf >= a_conf else "away"

    h_prof = game.get("home_profile") or {}
    a_prof = game.get("away_profile") or {}
    h_wr_1m = (h_prof.get("last_1m", {}).get("overall", {}).get("win_rate") or 50)
    a_wr_1m = (a_prof.get("last_1m", {}).get("overall", {}).get("win_rate") or 50)
    h_ats_3m = (h_prof.get("last_3m", {}).get("overall", {}).get("ats_rate") or 50)
    a_ats_3m = (a_prof.get("last_3m", {}).get("overall", {}).get("ats_rate") or 50)
    h_season_wr = (h_prof.get("season", {}).get("overall", {}).get("win_rate") or 50)
    a_season_wr = (a_prof.get("season", {}).get("overall", {}).get("win_rate") or 50)

    is_po = bool(game.get("is_playoff"))
    series_gn = int(game.get("series_game_num") or 0)
    series_lead = int(game.get("series_lead_for_home") or 0)
    is_elim = bool(game.get("series_is_elimination"))

    h_zh = game.get("home_team_zh") or game.get("home_team") or "?"
    a_zh = game.get("away_team_zh") or game.get("away_team") or "?"

    # ── Regular season ──
    if not is_po:
        if abs_sp >= 5 and h_wr_1m <= 35:
            signals.append({"text": "主場弱隊大讓分 → 押客場/受讓（歷史 70% WR）", "tier": "hot"})
        if abs_sp >= 5 and abs_sp < 10 and h_wr_1m <= 35:
            signals.append({"text": "讓分 5-10 + 主場弱隊 → 押冷門（70.1%）", "tier": "hot"})
        if h_ats_3m <= 35:
            signals.append({"text": f"{h_zh}近期 ATS 差 → 押客場 cover（61.7%）", "tier": "warm"})
        if a_wr_1m >= 65:
            signals.append({"text": f"{a_zh} L10 強隊 → 押客場 cover（60.1%）", "tier": "warm"})
        if a_ats_3m >= 65:
            signals.append({"text": f"{a_zh} ATS 近期熱 → 押客場 cover（59.0%）", "tier": "warm"})
        if not home_fav and h_season_wr >= 55:
            signals.append({"text": "主場受讓 + 例行賽強隊 → 押主場 cover（61.5%）", "tier": "warm"})
        if abs_sp >= 5 and abs_sp < 10:
            signals.append({"text": "中讓分(5-10) → 冷門 cover 傾向（58.9%）", "tier": "base"})

    # ── Playoffs ──
    # Round-specific base rates (re-measured 2026-04-20 across 13 seasons 2012-2025, n=1053):
    #   R1 G1: home 58.0% (n=112)    R1 G2: home 55.4% (n=112)
    #   R1 G3: away 59.6% (n=104)    R1 G4: away 60.8% (n=102)
    #   R1 G5 2-2: home 53.1% (all)/60% (recent 2017+, n=35)
    #   R1 G6: AWAY 80.4% (n=46) ★ strongest structural signal
    #   R2 G1: away 60.4% (n=48)     R2+ G6: away 53.8% (n=52)
    round_num = int(game.get("round_num") or 0)
    if is_po:
        # R1 G6 — GOLD tier hot signal
        if round_num == 1 and series_gn == 6:
            signals.append({"text": "R1 G6 → 押客場（高種子）歷史 cover 率 80.4%（n=46，近5年 84.6%）", "tier": "hot"})
        elif series_gn == 6 and round_num >= 2:
            signals.append({"text": f"R{round_num} G6 → 押客場（53.8%，次輪後效果減弱）", "tier": "base"})
        elif series_gn == 7:
            if abs_sp >= 7:
                signals.append({"text": f"G7 大讓分（{abs_sp}）→ 歷史主場 cover 66.7%（n=6，樣本小，冷門信號已抑制）", "tier": "base"})
            else:
                signals.append({"text": f"G7 小讓分（{abs_sp}）→ 冷門 cover 61.3%（n=31，13季）", "tier": "warm"})
        # R1 G3/G4: away team (high seed) covers 60%
        if round_num == 1 and series_gn in (3, 4):
            signals.append({"text": f"R1 G{series_gn} → 押客場（高種子）歷史 cover 率 ~60%（n~100）", "tier": "warm"})
        # G5 tied 2-2 home
        if series_gn == 5 and series_lead == 0:
            signals.append({"text": "Game 5（2-2 平手）→ 押主場 cover（60%，近7年）", "tier": "hot"})
        # R2 G1 away
        if round_num == 2 and series_gn == 1:
            signals.append({"text": "R2 G1 → 押客場歷史 cover 率 60.4%（n=48）", "tier": "warm"})
        if is_elim and not (series_gn == 7 and abs_sp >= 7):
            signals.append({"text": "淘汰局 → 押冷門 cover（56.7%）", "tier": "warm"})
        if is_elim and abs_sp <= 8 and not (series_gn == 7 and abs_sp >= 7):
            signals.append({"text": "小讓分淘汰局 → 押冷門（56.6%）", "tier": "warm"})
        if model_conf >= 65 and not is_elim:
            side_zh = "主場" if model_side == "home" else "客場"
            signals.append({"text": f"模型信心 ≥65% + 非淘汰局 → 押{side_zh}（57.1%）", "tier": "warm"})
        if a_season_wr >= 60 and home_fav:
            signals.append({"text": f"例行賽強隊受讓（{a_zh}）→ 押客場 cover（54.7%）", "tier": "base"})
        if h_season_wr >= 60 and not home_fav:
            signals.append({"text": f"例行賽強隊受讓（{h_zh}）→ 押主場 cover（58.8%）", "tier": "warm"})
        if series_gn <= 2:
            signals.append({"text": "系列賽前兩場 → 主場略有 ATS 優勢（54%）", "tier": "base"})
        if series_lead <= -2:
            signals.append({"text": "主場大幅落後 → 客場 cover 傾向（55.5%）", "tier": "base"})
        if is_elim and series_lead < 0:
            signals.append({"text": "⚠ 主場淘汰局（落後方）→ 避免押主場（僅 39.4%）", "tier": "danger"})

    if model_conf >= 70:
        signals.append({"text": "ML 模型高信心 ≥70% → 勝負盤可靠", "tier": "base"})

    return signals
