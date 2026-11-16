"""Regular-season ATS backtest strategies: one source of truth for the card
checklist, the AI prompt and the season-wide "which games matched" list.

Each strategy names the side to buy (home / away / underdog) so the card can
print the actual team and line ("押 湖人 +5.5") instead of "押客場".
`build_strategy_matches` scans the whole season (OddsData.sqlite) with the
pre-game team profile of each game date, so the list on the site is the
same test a reader could run by hand: it never uses the finished game's
own result in its own filter.

Historical hit rates in `hist_wr` are the 2025-26 backtest figures quoted in
the original checklist; this season's record is recomputed on every export.
"""
from __future__ import annotations

import math
import sqlite3
from datetime import datetime

import pandas as pd

from src.Utils.OddsTables import load_season_odds
from src.Utils.SeasonStats import ODDS_DB
from src.Utils.TeamProfile import grade_spread, team_profile_for_date
from src.Utils.Teams import team_name_zh

STRATEGIES: list[dict] = [
    {"key": "home_weak_big_spread", "icon": "🔥", "tier": "hot", "hist_wr": 70.0,
     "name_zh": "主場弱隊大讓分", "rule_zh": "讓分 ≥5 且主隊近 30 天勝率 ≤35%",
     # scripts/backtest_ats.py analyze_combined measured this as side="dog" (押冷門);
     # the old card text "押客場/受讓" was ambiguous and contradicted the 5–10 rule
     # whenever the weak home team was the underdog.
     "pick_rule": "dog", "pick_rule_zh": "押受讓方"},
    {"key": "mid_spread_home_weak", "icon": "🔥", "tier": "hot", "hist_wr": 70.1,
     "name_zh": "讓分 5–10 ＋ 主場弱隊", "rule_zh": "讓分 5–10 且主隊近 30 天勝率 ≤35%",
     "pick_rule": "dog", "pick_rule_zh": "押受讓方"},
    {"key": "home_ats_cold", "icon": "📉", "tier": "warm", "hist_wr": 61.7,
     "name_zh": "主隊近期 ATS 差", "rule_zh": "主隊近 90 天過盤率 ≤35%",
     "pick_rule": "away", "pick_rule_zh": "押客隊"},
    {"key": "away_hot_1m", "icon": "📈", "tier": "warm", "hist_wr": 60.1,
     "name_zh": "客隊近期強勢", "rule_zh": "客隊近 30 天勝率 ≥65%",
     "pick_rule": "away", "pick_rule_zh": "押客隊"},
    {"key": "away_ats_hot", "icon": "📈", "tier": "warm", "hist_wr": 59.0,
     "name_zh": "客隊 ATS 近期熱", "rule_zh": "客隊近 90 天過盤率 ≥65%",
     "pick_rule": "away", "pick_rule_zh": "押客隊"},
    {"key": "home_dog_strong", "icon": "💡", "tier": "warm", "hist_wr": 61.5,
     "name_zh": "主場受讓 ＋ 例行賽強隊", "rule_zh": "主隊受讓且本季勝率 ≥55%",
     "pick_rule": "home", "pick_rule_zh": "押主隊"},
    {"key": "mid_spread_dog", "icon": "📊", "tier": "base", "hist_wr": 58.9,
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
        "text": f"{s['name_zh']} → 押 {team_zh} {line}（歷史 {s['hist_wr']:g}%）",
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


def build_strategy_matches(season_key: str, today_iso: str) -> dict:
    """Season-wide match list per strategy, newest game first."""
    with sqlite3.connect(ODDS_DB) as con:
        df = load_season_odds(
            con, season_key,
            ["Date", "Home", "Away", "Spread", "ML_Home", "ML_Away", "Points", "Win_Margin"],
        )
    per_key: dict[str, list[dict]] = {s["key"]: [] for s in STRATEGIES}
    if df is not None and not df.empty:
        df = df.sort_values("Date", ascending=False).reset_index(drop=True)
        for row in df.itertuples(index=False):
            spread = row.Spread
            if spread is None or (isinstance(spread, float) and math.isnan(spread)):
                continue
            date_iso = str(row.Date)[:10]
            hp = team_profile_for_date(row.Home, date_iso)
            ap = team_profile_for_date(row.Away, date_iso)
            keys = matched_keys(spread, hp, ap)
            if not keys:
                continue
            played = _num(row.Points, 0.0) > 0
            win_margin = row.Win_Margin if played else None
            home_zh, away_zh = team_name_zh(row.Home), team_name_zh(row.Away)
            for k in keys:
                sig = _signal(k, float(spread), row.Home, row.Away, home_zh, away_zh)
                result, cover = _result_for(sig["pick_side"], spread, win_margin)
                entry = {
                    "date": date_iso, "home": row.Home, "away": row.Away,
                    "home_zh": home_zh, "away_zh": away_zh,
                    "spread": float(spread),
                    "home_line": side_line("home", spread), "away_line": side_line("away", spread),
                    "pick_side": sig["pick_side"], "pick_team": sig["pick_team"],
                    "pick_team_zh": sig["pick_team_zh"], "pick_line": sig["pick_line"],
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
                    entry["home_score"] = int(round((pts + wm) / 2))
                    entry["away_score"] = int(round((pts - wm) / 2))
                per_key[k].append(entry)

    strategies = []
    for s in STRATEGIES:
        games = per_key[s["key"]]
        wins = sum(1 for g in games if g["result"] == "win")
        losses = sum(1 for g in games if g["result"] == "loss")
        pushes = sum(1 for g in games if g["result"] == "push")
        pending = sum(1 for g in games if g["result"] == "pending")
        decided = wins + losses
        strategies.append({
            **{k: s[k] for k in ("key", "icon", "tier", "hist_wr", "name_zh", "rule_zh", "pick_rule", "pick_rule_zh")},
            "season": {
                "wins": wins, "losses": losses, "pushes": pushes, "pending": pending,
                "n": decided, "hit_rate": round(wins / decided * 100, 1) if decided else None,
            },
            "games": games,
        })
    return {
        "season": season_key,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "as_of": today_iso,
        "note_zh": "條件用該場開賽前的球隊輪廓（嚴格早於比賽日）計算；盤口為資料庫最後一次刷新的讓分，正值＝主隊讓分。歷史勝率是 2025-26 回測數字，本季戰績每次匯出重算，僅供參考、不列為推薦。",
        "strategies": strategies,
    }
