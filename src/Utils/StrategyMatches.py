"""Regular-season ATS backtest strategies: one source of truth for the card
checklist, the AI prompt and the season-wide "which games matched" list.

Each strategy names the side to buy (home / away / underdog) so the card can
print the actual team and line ("押 湖人 +5.5") instead of "押客場".
`build_strategy_matches` scans the whole season (OddsData.sqlite) with the
pre-game team profile of each game date, so the list on the site is the
same test a reader could run by hand: it never uses the finished game's
own result in its own filter.

`hist_wr` is the pooled record over the 19 past seasons 2007-08 → 2025-26
(scripts/validate_strategies.py, 2026-11-17); `claimed_wr` is the figure the old
checklist quoted from a single-season backtest. None of the seven rules, nor the
weighted verdict (49.5%, n=6,496), beats the 52.6–53.5% break-even of a 0.87–0.90
payout over those seasons, so every card carries that warning; this season's
record is recomputed on every export and shown beside it.
"""
from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from datetime import datetime

import pandas as pd

from src.Utils.OddsTables import load_season_odds
from src.Utils.SeasonStats import ODDS_DB
from src.Utils.TeamProfile import grade_spread, team_profile_for_date
from src.Utils.Teams import team_name_zh

STRATEGIES: list[dict] = [
    {"key": "home_weak_big_spread", "icon": "🔥", "tier": "hot", "hist_wr": 50.6, "hist_n": 2752, "claimed_wr": 70.0,
     "name_zh": "主場弱隊大讓分", "rule_zh": "讓分 ≥5 且主隊近 30 天勝率 ≤35%",
     # scripts/backtest_ats.py analyze_combined measured this as side="dog" (押冷門);
     # the old card text "押客場/受讓" was ambiguous and contradicted the 5–10 rule
     # whenever the weak home team was the underdog.
     "pick_rule": "dog", "pick_rule_zh": "押受讓方"},
    {"key": "mid_spread_home_weak", "icon": "🔥", "tier": "hot", "hist_wr": 49.9, "hist_n": 2167, "claimed_wr": 70.1,
     "name_zh": "讓分 5–10 ＋ 主場弱隊", "rule_zh": "讓分 5–10 且主隊近 30 天勝率 ≤35%",
     "pick_rule": "dog", "pick_rule_zh": "押受讓方"},
    {"key": "home_ats_cold", "icon": "📉", "tier": "warm", "hist_wr": 50.5, "hist_n": 1828, "claimed_wr": 61.7,
     "name_zh": "主隊近期 ATS 差", "rule_zh": "主隊近 90 天過盤率 ≤35%",
     "pick_rule": "away", "pick_rule_zh": "押客隊"},
    {"key": "away_hot_1m", "icon": "📈", "tier": "warm", "hist_wr": 49.6, "hist_n": 5679, "claimed_wr": 60.1,
     "name_zh": "客隊近期強勢", "rule_zh": "客隊近 30 天勝率 ≥65%",
     "pick_rule": "away", "pick_rule_zh": "押客隊"},
    {"key": "away_ats_hot", "icon": "📈", "tier": "warm", "hist_wr": 50.8, "hist_n": 1818, "claimed_wr": 59.0,
     "name_zh": "客隊 ATS 近期熱", "rule_zh": "客隊近 90 天過盤率 ≥65%",
     "pick_rule": "away", "pick_rule_zh": "押客隊"},
    {"key": "home_dog_strong", "icon": "💡", "tier": "warm", "hist_wr": 49.4, "hist_n": 1232, "claimed_wr": 61.5,
     "name_zh": "主場受讓 ＋ 例行賽強隊", "rule_zh": "主隊受讓且本季勝率 ≥55%",
     "pick_rule": "home", "pick_rule_zh": "押主隊"},
    {"key": "mid_spread_dog", "icon": "📊", "tier": "base", "hist_wr": 49.4, "hist_n": 8957, "claimed_wr": 58.9,
     "name_zh": "中讓分（5–10）", "rule_zh": "讓分 5–10（不看戰績）",
     "pick_rule": "dog", "pick_rule_zh": "押受讓方"},
]
STRATEGY_BY_KEY = {s["key"]: s for s in STRATEGIES}


def _num(v, default=50.0) -> float:
    """None / NaN → default (no games yet); a real 0.0 stays 0.0 — the card's
    Jinja `|default(50)` behaves the same, the old python `or 50` did not."""
    if v is None:
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(f) else f


def _window(prof: dict | None, window: str, field: str) -> float:
    prof = prof or {}
    return _num(((prof.get(window) or {}).get("overall") or {}).get(field))


def matched_keys(spread, home_profile, away_profile) -> list[str]:
    """Strategy keys that fire for one regular-season game."""
    if spread is None:
        return []
    try:
        sp = float(spread)
    except (TypeError, ValueError):
        return []
    if math.isnan(sp):
        return []
    abs_sp = abs(sp)
    home_fav = sp > 0
    h_wr_1m = _window(home_profile, "last_1m", "win_rate")
    a_wr_1m = _window(away_profile, "last_1m", "win_rate")
    h_ats_3m = _window(home_profile, "last_3m", "ats_rate")
    a_ats_3m = _window(away_profile, "last_3m", "ats_rate")
    h_season_wr = _window(home_profile, "season", "win_rate")

    keys = []
    if abs_sp >= 5 and h_wr_1m <= 35:
        keys.append("home_weak_big_spread")
    if 5 <= abs_sp < 10 and h_wr_1m <= 35:
        keys.append("mid_spread_home_weak")
    if h_ats_3m <= 35:
        keys.append("home_ats_cold")
    if a_wr_1m >= 65:
        keys.append("away_hot_1m")
    if a_ats_3m >= 65:
        keys.append("away_ats_hot")
    if not home_fav and h_season_wr >= 55:
        keys.append("home_dog_strong")
    if 5 <= abs_sp < 10:
        keys.append("mid_spread_dog")
    return keys


def pick_side(key: str, spread: float) -> str:
    rule = STRATEGY_BY_KEY[key]["pick_rule"]
    if rule == "dog":
        return "away" if float(spread) > 0 else "home"
    return rule


def side_line(side: str, spread: float) -> str:
    """Line from the picked side's point of view. `spread` is home-referenced
    (positive = home favoured), so home gets −spread and away gets +spread."""
    v = -float(spread) if side == "home" else float(spread)
    if v == 0:
        return "PK"
    v = round(v, 1)
    txt = f"{abs(v):g}"
    return f"+{txt}" if v > 0 else f"−{txt}"


def _signal(key: str, spread: float, home: str, away: str, home_zh: str, away_zh: str) -> dict:
    s = STRATEGY_BY_KEY[key]
    side = pick_side(key, spread)
    team = home if side == "home" else away
    team_zh = home_zh if side == "home" else away_zh
    line = side_line(side, spread)
    return {
        "key": key, "icon": s["icon"], "tier": s["tier"], "hist_wr": s["hist_wr"],
        "name_zh": s["name_zh"], "rule_zh": s["rule_zh"],
        "pick_side": side, "pick_team": team, "pick_team_zh": team_zh, "pick_line": line,
        "text": f"{s['name_zh']} → 押 {team_zh} {line}（19 季驗證 {s['hist_wr']:g}%）",
    }


def signals_for_game(game: dict) -> list[dict]:
    """Card signals for one exported game dict (uses its pre-game profiles)."""
    if game.get("is_playoff"):
        return []
    spread = game.get("spread")
    keys = matched_keys(spread, game.get("home_profile"), game.get("away_profile"))
    if not keys:
        return []
    home = game.get("home_team") or "?"
    away = game.get("away_team") or "?"
    home_zh = game.get("home_team_zh") or team_name_zh(home)
    away_zh = game.get("away_team_zh") or team_name_zh(away)
    return [_signal(k, float(spread), home, away, home_zh, away_zh) for k in keys]


def _result_for(side: str, spread, win_margin) -> tuple[str, float | None]:
    winner, margin = grade_spread(spread, win_margin)
    if winner is None:
        return "pending", None
    if winner == "push":
        return "push", 0.0
    return ("win" if winner == side else "loss"), (margin if side == "home" else -margin)


# ── Single verdict when several rules fire on one game ──────────────────────
# Weight of a rule = its hit rate above 50%, where the rate pools the 19-season
# validated figure (worth PRIOR_N games) with this season's record BEFORE the
# game date (walk-forward: a game never votes with its own result). A rule
# running under 50% gets weight 0. The three "buy the underdog on a 5+ spread"
# rules are nested, so only the strongest of them votes. The side with the
# larger total is the verdict; equal totals or all-zero weights → no bet.
PRIOR_N = 20
DOG_FAMILY = ("home_weak_big_spread", "mid_spread_home_weak", "mid_spread_dog")
VERDICT_KEY = "verdict"
VERDICT_META = {
    "key": VERDICT_KEY, "icon": "🎯", "tier": "hot", "hist_wr": None,
    "name_zh": "綜合結論（策略衝突時的單一答案）",
    "rule_zh": f"每條策略的票重＝(19 季驗證勝率×{PRIOR_N} 場＋本季賽前戰績) 的合併勝率減 50；受讓方三條巢狀規則只取最強一條；票重高的一邊勝出，平手或全為 0 不下注",
    "pick_rule": "weighted", "pick_rule_zh": "買票重較高的一隊",
}


def rule_weight(key: str, wins: int, losses: int) -> float:
    hist = STRATEGY_BY_KEY[key]["hist_wr"] / 100.0
    pooled = (wins + PRIOR_N * hist) / (wins + losses + PRIOR_N)
    return round(max(0.0, pooled * 100 - 50), 1)


def verdict_for(signals: list[dict], records: dict[str, tuple[int, int]]) -> dict | None:
    """One side to buy for a game, or side=None (no bet). `records` maps a
    strategy key to its (wins, losses) before this game's date."""
    if not signals:
        return None
    votes = []
    for sg in signals:
        w, l = records.get(sg["key"], (0, 0))
        votes.append({"key": sg["key"], "name_zh": sg["name_zh"], "side": sg["pick_side"],
                      "weight": rule_weight(sg["key"], w, l), "record": f"{w}–{l}"})
    dog = [v for v in votes if v["key"] in DOG_FAMILY]
    if len(dog) > 1:
        keep = max(dog, key=lambda v: v["weight"])
        for v in dog:
            if v is not keep:
                v["weight"], v["nested"] = 0.0, True
    score = {"home": 0.0, "away": 0.0}
    for v in votes:
        score[v["side"]] += v["weight"]
    score = {k: round(x, 1) for k, x in score.items()}
    side = None
    if score["home"] != score["away"]:
        side = "home" if score["home"] > score["away"] else "away"
    conflict = len({v["side"] for v in votes}) > 1
    margin = round(abs(score["home"] - score["away"]), 1)
    out = {"side": side, "conflict": conflict, "score_home": score["home"], "score_away": score["away"],
           "margin": margin, "tier": verdict_tier(margin) if side else None, "votes": votes}
    ref = next((sg for sg in signals if sg["pick_side"] == side), None)
    if ref:
        out.update({"team": ref["pick_team"], "team_zh": ref["pick_team_zh"], "line": ref["pick_line"]})
        other = "away" if side == "home" else "home"
        out["text"] = f"買 {ref['pick_team_zh']} {ref['pick_line']}（票重 {score[side]:g} 對 {score[other]:g}）"
    else:
        out["text"] = f"不下注（兩邊票重 {score['home']:g} 對 {score['away']:g}）"
    return out


# ── Payout check: a spread bet pays about 0.87–0.90 per unit staked ─────────
PAYOUT_LOW, PAYOUT_HIGH = 0.87, 0.90
BREAKEVEN_LOW = round(100 / (1 + PAYOUT_HIGH), 1)   # 52.6% at 0.90
BREAKEVEN_HIGH = round(100 / (1 + PAYOUT_LOW), 1)   # 53.5% at 0.87
EDGE_CEILING = 58.0   # under this a few results flip the sign of the ROI
VERDICT_TIERS = [(0.0, 6.0, "票重差 <6"), (6.0, 15.0, "票重差 6–15"), (15.0, 1e9, "票重差 ≥15")]
RISK_LABEL = {
    "lose": "⚠ 容易虧錢", "edge": "≈ 接近打平", "ok": "目前為正", "na": "尚無結果",
}


def payout_profile(wins: int, losses: int) -> dict:
    """ROI per unit at both ends of the payout range, plus a risk flag:
    lose = hit rate under the 0.87 break-even (53.5%); edge = above it but
    under 58%, where the profit is smaller than ordinary variance."""
    n = wins + losses
    if n == 0:
        return {"roi_low": None, "roi_high": None, "risk": "na", "risk_label": RISK_LABEL["na"]}
    rate = wins / n * 100
    roi = lambda pay: round((wins * pay - losses) / n * 100, 1)
    risk = "lose" if rate < BREAKEVEN_HIGH else ("edge" if rate < EDGE_CEILING else "ok")
    return {"roi_low": roi(PAYOUT_LOW), "roi_high": roi(PAYOUT_HIGH), "risk": risk, "risk_label": RISK_LABEL[risk]}


def verdict_tier(margin: float) -> int:
    for i, (lo, hi, _) in enumerate(VERDICT_TIERS):
        if lo <= margin < hi:
            return i
    return len(VERDICT_TIERS) - 1


def _load_validation() -> dict | None:
    p = Path(__file__).resolve().parents[2] / "web" / "data" / "strategy_validation.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def build_strategy_matches(season_key: str, today_iso: str) -> dict:
    """Season-wide match list per strategy plus the single verdict per game,
    newest game first."""
    with sqlite3.connect(ODDS_DB) as con:
        df = load_season_odds(
            con, season_key,
            ["Date", "Home", "Away", "Spread", "ML_Home", "ML_Away", "Points", "Win_Margin"],
        )
    per_key: dict[str, list[dict]] = {s["key"]: [] for s in STRATEGIES}
    verdict_games: list[dict] = []
    verdicts: dict[str, dict] = {}
    running = {s["key"]: [0, 0] for s in STRATEGIES}

    def _entry(row, date_iso, side, team, team_zh, line, hp, ap, played):
        spread = row.Spread
        result, cover = _result_for(side, spread, row.Win_Margin if played else None)
        e = {
            "date": date_iso, "home": row.Home, "away": row.Away,
            "home_zh": team_name_zh(row.Home), "away_zh": team_name_zh(row.Away),
            "spread": float(spread),
            "home_line": side_line("home", spread), "away_line": side_line("away", spread),
            "pick_side": side, "pick_team": team, "pick_team_zh": team_zh, "pick_line": line,
            "result": result,
            "cover_margin": None if cover is None else round(float(cover), 1),
            "home_score": None, "away_score": None,
            "h_wr_1m": _window(hp, "last_1m", "win_rate"),
            "a_wr_1m": _window(ap, "last_1m", "win_rate"),
            "h_ats_3m": _window(hp, "last_3m", "ats_rate"),
            "a_ats_3m": _window(ap, "last_3m", "ats_rate"),
            "h_season_wr": _window(hp, "season", "win_rate"),
        }
        if played:
            pts, wm = float(row.Points), float(row.Win_Margin)
            e["home_score"] = int(round((pts + wm) / 2))
            e["away_score"] = int(round((pts - wm) / 2))
        return e

    if df is not None and not df.empty:
        df = df.sort_values("Date").reset_index(drop=True)
        for date_val, day in df.groupby("Date", sort=True):
            date_iso = str(date_val)[:10]
            before = {k: tuple(v) for k, v in running.items()}  # records before this date
            day_results = []
            for row in day.itertuples(index=False):
                spread = row.Spread
                if spread is None or (isinstance(spread, float) and math.isnan(spread)):
                    continue
                hp = team_profile_for_date(row.Home, date_iso)
                ap = team_profile_for_date(row.Away, date_iso)
                keys = matched_keys(spread, hp, ap)
                if not keys:
                    continue
                played = _num(row.Points, 0.0) > 0
                home_zh, away_zh = team_name_zh(row.Home), team_name_zh(row.Away)
                sigs = [_signal(k, float(spread), row.Home, row.Away, home_zh, away_zh) for k in keys]
                for sg in sigs:
                    e = _entry(row, date_iso, sg["pick_side"], sg["pick_team"], sg["pick_team_zh"], sg["pick_line"], hp, ap, played)
                    per_key[sg["key"]].append(e)
                    day_results.append((sg["key"], e["result"]))
                v = verdict_for(sigs, before)
                verdicts[f"{date_iso}|{row.Home}|{row.Away}"] = v
                if v and v.get("side"):
                    e = _entry(row, date_iso, v["side"], v["team"], v["team_zh"], v["line"], hp, ap, played)
                    e["conflict"] = v["conflict"]
                    e["margin"], e["tier"] = v["margin"], v["tier"]
                    e["reason_zh"] = "；".join(
                        f"{x['name_zh']}→{'主' if x['side'] == 'home' else '客'} {x['weight']:g}" for x in v["votes"])
                    v["result"] = e["result"]
                    verdict_games.append(e)
            for k, res in day_results:
                if res == "win":
                    running[k][0] += 1
                elif res == "loss":
                    running[k][1] += 1

    def _season(games):
        wins = sum(1 for g in games if g["result"] == "win")
        losses = sum(1 for g in games if g["result"] == "loss")
        decided = wins + losses
        return {
            **payout_profile(wins, losses),
            "wins": wins, "losses": losses,
            "pushes": sum(1 for g in games if g["result"] == "push"),
            "pending": sum(1 for g in games if g["result"] == "pending"),
            "n": decided, "hit_rate": round(wins / decided * 100, 1) if decided else None,
        }

    meta_keys = ("key", "icon", "tier", "hist_wr", "hist_n", "claimed_wr", "name_zh", "rule_zh", "pick_rule", "pick_rule_zh")
    verdict_games.reverse()
    conflict_games = [g for g in verdict_games if g.get("conflict")]
    strategies = [{**VERDICT_META, "season": _season(verdict_games),
                   "season_conflict_only": _season(conflict_games), "games": verdict_games}]
    for s in STRATEGIES:
        games = per_key[s["key"]][::-1]
        w, l = running[s["key"]]
        strategies.append({**{k: s[k] for k in meta_keys}, "season": _season(games),
                           "weight_now": rule_weight(s["key"], w, l), "games": games})
    tiers = []
    for i, (lo, hi, label) in enumerate(VERDICT_TIERS):
        tiers.append({"index": i, "label_zh": label, "season": _season([g for g in verdict_games if g.get("tier") == i])})
    for v in verdicts.values():
        if v and v.get("tier") is not None:
            t = tiers[v["tier"]]
            v["tier_label_zh"] = t["label_zh"]
            v["tier_record"] = f"{t['season']['wins']}–{t['season']['losses']}"
            v["tier_hit_rate"] = t["season"]["hit_rate"]
            v["risk"], v["risk_label"] = t["season"]["risk"], t["season"]["risk_label"]
            v["tier_roi_low"], v["tier_roi_high"] = t["season"]["roi_low"], t["season"]["roi_high"]
    validation = _load_validation()
    if validation:
        vall = validation["verdict_walk_forward"]["all"]
        for st in strategies:
            past = (validation["strategies"].get(st["key"]) or {}).get("past")
            if st["key"] == VERDICT_KEY:
                past = vall
            if past:
                st["validated"] = {k: past[k] for k in ("wins", "losses", "n", "hit_rate", "ci95", "roi_low", "roi_high", "risk", "risk_label")}
        if vall["risk"] == "lose":
            # 19 seasons outweigh a few dozen games: the long-run flag wins.
            for v in verdicts.values():
                if v and v.get("side"):
                    v["season_tier_risk"] = v.get("risk")
                    v["risk"], v["risk_label"] = "lose", RISK_LABEL["lose"]
                    v["validated_hit_rate"], v["validated_n"] = vall["hit_rate"], vall["n"]
    return {
        "validation": None if not validation else {
            "seasons": len(validation["past_seasons"]), "first": validation["past_seasons"][0],
            "last": validation["past_seasons"][-1], "verdict": validation["verdict_walk_forward"]["all"],
            "note_zh": f"{len(validation['past_seasons'])} 季驗證（{validation['past_seasons'][0]} → {validation['past_seasons'][-1]}）：七條策略合併勝率都在 49.4%–50.8%，綜合結論 {validation['verdict_walk_forward']['all']['wins']}–{validation['verdict_walk_forward']['all']['losses']}（{validation['verdict_walk_forward']['all']['hit_rate']}%），沒有任何一條能過 1 賠 0.87–0.90 的打平線。舊版寫的「歷史 59%–70%」是單季回測，不成立。本季戰績較好的幾條目前應視為短期波動。",
        },
        "payout": {"low": PAYOUT_LOW, "high": PAYOUT_HIGH, "breakeven_low": BREAKEVEN_LOW,
                   "breakeven_high": BREAKEVEN_HIGH, "edge_ceiling": EDGE_CEILING,
                   "note_zh": f"讓分賠率約 1 賠 {PAYOUT_LOW}–{PAYOUT_HIGH}：過盤率要 {BREAKEVEN_LOW}%–{BREAKEVEN_HIGH}% 才打平。低於 {BREAKEVEN_HIGH}% 標「容易虧錢」，{BREAKEVEN_HIGH}%–{EDGE_CEILING:g}% 標「接近打平」（獲利小於正常波動）。"},
        "verdict_tiers": tiers,
        "season": season_key,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "as_of": today_iso,
        "note_zh": "條件用該場開賽前的球隊輪廓（嚴格早於比賽日）計算；盤口為資料庫最後一次刷新的讓分，正值＝主隊讓分。「19 季驗證」是 2007-08 → 2025-26 全部例行賽的合併勝率，本季戰績每次匯出重算，僅供參考、不列為推薦。",
        "strategies": strategies,
        "verdicts": verdicts,
    }
