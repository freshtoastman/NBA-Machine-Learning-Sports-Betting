"""Per-game AI analysis using Gemini (with Google Search grounding for
live injury/news data) + structured output format for frontend rendering.

Fallback chain: Gemini w/ search → Gemini w/o search → Claude → rule-based.
Results cached in SQLite by (date, home, away).
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import date, datetime
from pathlib import Path

try:
    from google import genai
except ImportError:
    genai = None

try:
    import anthropic
except ImportError:
    anthropic = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DB = PROJECT_ROOT / "Data" / "ai_analysis_cache.sqlite"


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _init_cache():
    with sqlite3.connect(CACHE_DB) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS analysis_v2 (
                cache_key TEXT PRIMARY KEY,
                game_date TEXT,
                home_team TEXT,
                away_team TEXT,
                analysis_json TEXT,
                created_at TEXT
            )
        """)


def _cache_key(game_date: str, home: str, away: str) -> str:
    return hashlib.md5(f"{game_date}:{home}:{away}".encode()).hexdigest()


def _get_cached(game_date: str, home: str, away: str) -> dict | None:
    _init_cache()
    key = _cache_key(game_date, home, away)
    with sqlite3.connect(CACHE_DB) as con:
        row = con.execute(
            "SELECT analysis_json FROM analysis_v2 WHERE cache_key = ?", (key,)
        ).fetchone()
    if row:
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return {"raw": row[0]}
    return None


def _set_cache(game_date: str, home: str, away: str, data: dict):
    _init_cache()
    key = _cache_key(game_date, home, away)
    with sqlite3.connect(CACHE_DB) as con:
        con.execute(
            """INSERT OR REPLACE INTO analysis_v2
               (cache_key, game_date, home_team, away_team, analysis_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (key, game_date, home, away, json.dumps(data, ensure_ascii=False),
             datetime.now().isoformat()),
        )


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def _build_game_context(game: dict, game_date: str) -> str:
    """Build structured context string from our model data."""
    from src.Utils.Teams import team_name_zh
    home = game.get("home_team", "?")
    away = game.get("away_team", "?")
    h_zh = team_name_zh(home)
    a_zh = team_name_zh(away)

    ml_pick = home if game.get("winner") == "home" else away
    ml_conf = game.get("home_confidence") if game.get("winner") == "home" else game.get("away_confidence", 0)
    ou_pick = game.get("ou_pick", "?")
    ou_val = game.get("ou_value", "?")
    ou_conf = game.get("ou_confidence", "?")

    ats_pick_side = game.get("ats_model_pick")
    ats_pick = home if ats_pick_side == "home" else away if ats_pick_side == "away" else "N/A"
    ats_conf = game.get("ats_model_confidence", "?")
    ats_edge = game.get("ats_value_edge", 0)

    spread = game.get("spread")
    spread_str = f"{spread:+.1f}" if spread is not None else "N/A"

    value_side = game.get("value_side")
    value_team = home if value_side == "home" else away if value_side else "N/A"
    value_edge = game.get("value_edge", 0)

    def _fmt_profile(profile, name):
        if not profile:
            return f"{name}: 無數據"
        parts = []
        for key, label in [("season", "本賽季"), ("last_1m", "近1月")]:
            w = profile.get(key, {}).get("overall", {})
            if w.get("games", 0) > 0:
                parts.append(f"{label} {w['wins']}-{w['losses']}({w['win_rate']}%) "
                             f"得{w['avg_for']}/失{w['avg_against']} "
                             f"ATS {w.get('ats_wins',0)}-{w.get('ats_losses',0)}")
        return f"{name}: " + " | ".join(parts) if parts else f"{name}: 無數據"

    return f"""{a_zh}({away}) @ {h_zh}({home})
讓分: {h_zh} {spread_str}
ML模型: {ml_pick} ({ml_conf}%)
UO模型: {ou_pick} {ou_val} ({ou_conf}%)
ATS模型: {ats_pick} 蓋 ({ats_conf}%, edge {ats_edge}pp)
鑽石場次: {'是 — ' + value_team + f' (edge +{value_edge}pp)' if game.get('is_value') else '否'}
共識: {'🔥 是' if game.get('is_consensus') else '否'}
{_fmt_profile(game.get('home_profile'), h_zh)}
{_fmt_profile(game.get('away_profile'), a_zh)}"""


SINGLE_GAME_PROMPT = """你是 NBA 專業分析師。根據以下資料和你搜尋到的最新傷兵/新聞，對這場比賽做分析。

比賽日期: {game_date}
{context}

請用以下 JSON 格式回答（不要加 markdown code block）：
{{
  "injuries_home": ["球員名 - 狀態(OUT/GTD/Available)", ...],
  "injuries_away": ["球員名 - 狀態(OUT/GTD/Available)", ...],
  "key_factors": ["因素1", "因素2", "因素3"],
  "ml_verdict": "看好哪隊勝（隊名）",
  "ml_reason": "一句話理由",
  "ats_verdict": "押哪邊讓分（隊名 +/- 分數）",
  "ats_reason": "一句話理由",
  "ou_verdict": "大分/小分",
  "ou_reason": "一句話理由",
  "confidence": "高/中/低",
  "summary": "50字內總結建議"
}}

重要：
1. 先搜尋兩隊今天的傷兵報告(injury report)
2. injuries 欄位要列出實際查到的傷兵，不要編造
3. key_factors 要包含傷兵影響、賽程疲勞、季後賽動機等模型看不到的因素
4. 用繁體中文回答"""


DAILY_REPORT_PROMPT = """你是 NBA 專業分析師。以下是今天所有比賽的模型預測和數據。請產出一份當日分析報告。

日期: {game_date}
比賽數量: {num_games}

{all_contexts}

請用以下 JSON 格式回答（不要加 markdown code block）：
{{
  "headline": "今日 NBA 一句話標題",
  "top_picks": [
    {{"game": "隊A vs 隊B", "pick": "建議", "confidence": "高/中/低", "reason": "理由"}},
    ...最多3場
  ],
  "avoid_games": [
    {{"game": "隊A vs 隊B", "reason": "為什麼要避開"}}
  ],
  "injury_impact": ["重大傷兵影響1", "重大傷兵影響2"],
  "daily_summary": "100字內的當日總結，包含整體策略建議"
}}

重要：
1. 搜尋今天所有重大傷兵消息
2. top_picks 只選最有把握的場次（鑽石場次 + 高信心）
3. avoid_games 列出不確定性高、應避開的場次
4. 用繁體中文回答"""


# ---------------------------------------------------------------------------
# AI calls
# ---------------------------------------------------------------------------

def _call_gemini(prompt: str, use_search: bool = True) -> str | None:
    key = os.environ.get("GEMINI_API_KEY")
    if not key or genai is None:
        return None
    try:
        client = genai.Client(api_key=key)
        config = {}
        if use_search:
            config["tools"] = [{"google_search": {}}]
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=config,
        )
        return response.text
    except Exception:
        return None


def _call_claude(prompt: str) -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key or anthropic is None:
        return None
    try:
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except Exception:
        return None


def _parse_json_response(text: str) -> dict | None:
    """Extract JSON from AI response, handling markdown fences."""
    if not text:
        return None
    # Strip markdown code fences if present.
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON object in text.
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass
    return None


def _rule_based_analysis(game: dict, game_date: str) -> dict:
    """Generate structured analysis without API calls."""
    from src.Utils.Teams import team_name_zh
    home = game.get("home_team", "?")
    away = game.get("away_team", "?")
    h_zh = team_name_zh(home)
    a_zh = team_name_zh(away)

    ml_pick = h_zh if game.get("winner") == "home" else a_zh
    ml_conf = game.get("home_confidence") if game.get("winner") == "home" else game.get("away_confidence", 0)

    ats_pick_side = game.get("ats_model_pick")
    ats_team = h_zh if ats_pick_side == "home" else a_zh if ats_pick_side else "N/A"
    spread = game.get("spread")
    spread_str = f"{spread:+.1f}" if spread is not None else ""

    ou_pick = game.get("ou_pick", "?")
    ou_val = game.get("ou_value", "?")

    factors = []
    if ml_conf and ml_conf >= 70:
        factors.append(f"模型高信心看好 {ml_pick}")
    if game.get("ats_is_value"):
        factors.append(f"ATS 模型看好 {ats_team} 蓋過讓分")
    if game.get("is_consensus"):
        factors.append("雙模型共識，最強訊號")

    for side, zh, prof in [("home", h_zh, game.get("home_profile")),
                            ("away", a_zh, game.get("away_profile"))]:
        m = (prof or {}).get("last_1m", {}).get("overall", {})
        if m.get("games", 0) >= 5:
            if m.get("win_rate", 50) >= 70:
                factors.append(f"{zh} 近一個月火燙 ({m['win_rate']}%)")
            elif m.get("win_rate", 50) <= 35:
                factors.append(f"{zh} 近一個月低迷 ({m['win_rate']}%)")

    conf = "高" if (ml_conf or 0) >= 70 or game.get("is_consensus") else "中" if (ml_conf or 0) >= 60 else "低"

    return {
        "injuries_home": ["無法查詢（離線模式）"],
        "injuries_away": ["無法查詢（離線模式）"],
        "key_factors": factors or ["無特殊因素"],
        "ml_verdict": ml_pick,
        "ml_reason": f"模型信心 {ml_conf}%",
        "ats_verdict": f"{ats_team} {spread_str}",
        "ats_reason": f"ATS 模型 edge {game.get('ats_value_edge', 0)}pp" if game.get("ats_model_pick") else "無 ATS 訊號",
        "ou_verdict": f"{'大分' if ou_pick == 'OVER' else '小分'} {ou_val}",
        "ou_reason": f"模型信心 {game.get('ou_confidence', '?')}%",
        "confidence": conf,
        "summary": f"{'🔥 雙模型共識' if game.get('is_consensus') else '💎 鑽石場次' if game.get('is_value') else '常規場次'}，建議{'積極跟進' if conf == '高' else '留意觀察' if conf == '中' else '觀望為主'}。",
        "source": "rule-based",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_game(game: dict, game_date: str, force: bool = False) -> dict:
    """Generate structured AI analysis for a single game.

    Returns a dict with keys: injuries_home, injuries_away, key_factors,
    ml_verdict, ml_reason, ats_verdict, ats_reason, ou_verdict, ou_reason,
    confidence, summary, source.
    """
    home = game.get("home_team", "")
    away = game.get("away_team", "")

    if not force:
        cached = _get_cached(game_date, home, away)
        if cached:
            return cached

    context = _build_game_context(game, game_date)
    prompt = SINGLE_GAME_PROMPT.format(game_date=game_date, context=context)

    result = None

    # Try Gemini with search (best: gets live injury data).
    raw = _call_gemini(prompt, use_search=True)
    if raw:
        result = _parse_json_response(raw)
        if result:
            result["source"] = "gemini+search"

    # Fallback: Gemini without search.
    if result is None:
        raw = _call_gemini(prompt, use_search=False)
        if raw:
            result = _parse_json_response(raw)
            if result:
                result["source"] = "gemini"

    # Fallback: Claude.
    if result is None:
        raw = _call_claude(prompt)
        if raw:
            result = _parse_json_response(raw)
            if result:
                result["source"] = "claude"

    # Fallback: rule-based.
    if result is None:
        result = _rule_based_analysis(game, game_date)

    _set_cache(game_date, home, away, result)
    return result


def generate_daily_report(games: dict, game_date: str, force: bool = False) -> dict:
    """Generate a daily summary report for all games on a date.

    Returns dict with: headline, top_picks, avoid_games, injury_impact, daily_summary.
    """
    cache_key_date = f"__daily__{game_date}"
    if not force:
        cached = _get_cached(game_date, cache_key_date, "")
        if cached:
            return cached

    all_contexts = []
    for i, (game_key, game) in enumerate(games.items(), 1):
        all_contexts.append(f"=== 第{i}場 ===\n{_build_game_context(game, game_date)}")

    prompt = DAILY_REPORT_PROMPT.format(
        game_date=game_date,
        num_games=len(games),
        all_contexts="\n\n".join(all_contexts),
    )

    result = None
    raw = _call_gemini(prompt, use_search=True)
    if raw:
        result = _parse_json_response(raw)

    if result is None:
        raw = _call_gemini(prompt, use_search=False)
        if raw:
            result = _parse_json_response(raw)

    if result is None:
        raw = _call_claude(prompt)
        if raw:
            result = _parse_json_response(raw)

    if result is None:
        # Minimal fallback.
        value_games = [g for g in games.values() if g.get("is_value")]
        result = {
            "headline": f"今日 {len(games)} 場比賽，{len(value_games)} 場鑽石",
            "top_picks": [],
            "avoid_games": [],
            "injury_impact": ["無法取得傷兵資訊（離線模式）"],
            "daily_summary": f"共 {len(games)} 場比賽，{len(value_games)} 場觸發鑽石訊號。建議集中在鑽石場次。",
            "source": "rule-based",
        }

    _set_cache(game_date, cache_key_date, "", result)
    return result
