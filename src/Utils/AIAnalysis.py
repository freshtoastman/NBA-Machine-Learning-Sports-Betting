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

    # Spread convention here: positive => home is favorite (gives points).
    # We must render BOTH sides explicitly so the LLM doesn't flip the sign.
    spread = game.get("spread")
    if spread is None:
        spread_str = "N/A"
        pick_option_a = pick_option_b = None
    elif spread > 0:
        spread_str = f"{h_zh} -{spread:.1f} / {a_zh} +{spread:.1f}（主隊讓 {spread:.1f} 分）"
        pick_option_a = f"押 {h_zh} -{spread:.1f} 分"
        pick_option_b = f"押 {a_zh} +{spread:.1f} 分"
    elif spread < 0:
        spread_str = f"{h_zh} +{-spread:.1f} / {a_zh} -{-spread:.1f}（客隊讓 {-spread:.1f} 分）"
        pick_option_a = f"押 {a_zh} -{-spread:.1f} 分"
        pick_option_b = f"押 {h_zh} +{-spread:.1f} 分"
    else:
        spread_str = "PK（無讓分）"
        pick_option_a = f"押 {h_zh}（PK）"
        pick_option_b = f"押 {a_zh}（PK）"

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

    pick_options_block = ""
    if pick_option_a and pick_option_b:
        pick_options_block = (
            f"\n讓分推薦只能從以下兩個字串擇一（複製貼上，不要改任何字）：\n"
            f"  選項A: {pick_option_a}\n"
            f"  選項B: {pick_option_b}"
        )

    return f"""{a_zh}（主場：{h_zh}）
讓分盤: {spread_str}
ML模型: {team_name_zh(ml_pick)} ({ml_conf}%)
ATS模型: {team_name_zh(ats_pick) if ats_pick != 'N/A' else 'N/A'}（{ats_conf}%, edge {ats_edge}pp）
鑽石場次: {'是 — ' + team_name_zh(value_team) + f' (edge +{value_edge}pp)' if game.get('is_value') else '否'}
共識: {'🔥 是' if game.get('is_consensus') else '否'}
{_fmt_profile(game.get('home_profile'), h_zh)}
{_fmt_profile(game.get('away_profile'), a_zh)}{pick_options_block}"""


SINGLE_GAME_PROMPT = """你是職業 NBA 讓分盤分析師。你的讀者是認真的體育投注者，他們需要明確的讓分推薦。

比賽日期: {game_date}
{context}

=== 你的任務 ===
1. 搜尋兩隊今天的 injury report（傷兵報告），列出每個傷兵狀態
2. 評估傷兵對讓分盤的影響（缺主力 = 可能蓋不過讓分）
3. 綜合模型 + 傷兵 + 賽程 + 動機，給出明確讓分推薦
4. 勝負盤（moneyline）賠率通常太低不推薦 — 除非冷門值得小注

=== 回答格式（JSON，不要 code block）===
{{
  "injuries_home": ["球員名 - OUT/GTD/Available（傷勢）", ...],
  "injuries_away": ["球員名 - OUT/GTD/Available（傷勢）", ...],
  "injury_impact": "傷兵對讓分盤的具體影響",
  "key_factors": ["因素1", "因素2", "因素3"],
  "ats_pick": "從上方提供的『選項A』或『選項B』中挑一個，整個字串原封不動複製",
  "ats_reason": "為什麼選這邊（2-3 句）",
  "ats_units": 1到5的整數,
  "ml_note": "勝負盤備註（通常寫'賠率過低不建議'）",
  "risk_warning": "翻車風險",
  "summary": "一句話結論"
}}

=== 嚴格規則 ===
- **ats_pick 必須從『讓分推薦只能從以下兩個字串擇一』那段中複製整段字串，不能改任何字、不能換符號、不能用英文隊名。**
- 如果讓分盤沒有優勢，ats_pick 寫「不推薦」，ats_units 寫 0
- ats_units 必須是 1-5 的整數
- 隊名一律用繁體中文，不要用英文
- injuries 必須是實際搜尋到的，不能編造"""


DAILY_REPORT_PROMPT = """你是職業 NBA 讓分盤分析師團隊主管。你的任務是產出一份讓投注者能直接照著操作的當日報告。

日期: {game_date}
比賽數量: {num_games}

{all_contexts}

=== 你的任務 ===
1. 搜尋今天所有重大傷兵消息
2. 從每場比賽提供的『選項A / 選項B』中，挑出值得下注的讓分推薦
3. 標出應該避開的比賽
4. 寫出整體策略

=== 回答格式（JSON，不要 code block）===
{{
  "headline": "今日 NBA 讓分盤精選（一句吸引眼球的標題）",
  "best_bets": [
    {{
      "game": "客隊 @ 主隊（繁體中文隊名）",
      "spread": "從該場 context 複製『讓分盤:』那行的內容",
      "pick": "從該場 context 提供的『選項A 或 選項B』中挑一個，整段原封不動複製",
      "units": 1到5的整數,
      "reason": "2-3 句具體分析（含傷兵影響）",
      "model_support": "模型資料佐證（ML 信心、ATS edge 等）"
    }}
  ],
  "lean_picks": [
    {{
      "game": "客隊 @ 主隊",
      "pick": "從該場 context 的『選項A 或 選項B』中挑一個，整段原封不動複製",
      "units": 1到2的整數,
      "reason": "一句話理由"
    }}
  ],
  "avoid_games": [
    {{"game": "客隊 @ 主隊", "reason": "為什麼避開"}}
  ],
  "injury_alerts": ["重大傷兵消息1", "重大傷兵消息2"],
  "bankroll_plan": "今日資金分配建議",
  "daily_summary": "100字策略總結"
}}

=== 嚴格規則 ===
- **best_bets 與 lean_picks 的 pick 欄位必須從該場提供的『選項A』或『選項B』整段原封不動複製。不能改任何字、不能換符號、不能用英文隊名。**
- best_bets 最多 3 場，units 必須 >= 3
- lean_picks 是次級推薦，units 1-2
- units 必須是數字
- 如果今天沒有好機會，best_bets 可以是空陣列
- 隊名一律用繁體中文
- injury_alerts 必須真實，不能編造
- 用繁體中文回答"""


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

    # If we don't have a spread, the AI cannot recommend a meaningful ATS pick;
    # fall back to a structured "no odds" response so the front-end can show it.
    if game.get("spread") is None:
        from src.Utils.Teams import team_name_zh as _zh
        return {
            "injuries_home": [],
            "injuries_away": [],
            "injury_impact": "尚無資料",
            "key_factors": ["運彩公司尚未開盤"],
            "ats_pick": "不推薦（尚無讓分盤口）",
            "ats_reason": "今天 sbrscrape 抓不到讓分資料，請賽前 2-4 小時再試。",
            "ats_units": 0,
            "ou_pick": "不推薦",
            "ou_reason": "尚無總分盤口",
            "ml_note": "無金錢線資料",
            "risk_warning": "等待盤口公布後再下注。",
            "summary": "今日尚無賠率，無法生成讓分推薦。",
            "source": "no-odds-fallback",
        }

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

    # Pre-flight: only include games that actually have spread data. AI cannot
    # write a meaningful `pick` like "押 X 隊 -5 分" without a spread.
    games_with_spread = {k: g for k, g in games.items() if g.get("spread") is not None}
    if not games_with_spread:
        from src.Utils.Teams import team_name_zh as _zh
        result = {
            "headline": f"⚠️ {game_date} 今日尚無賠率資料",
            "best_bets": [],
            "lean_picks": [],
            "avoid_games": [],
            "injury_alerts": [],
            "bankroll_plan": "等待運彩公司開盤後再下注。",
            "daily_summary": (
                f"今日 {len(games)} 場比賽，但運彩公司尚未公布讓分/總分/金錢線盤口"
                f"（sbrscrape 抓不到資料）。模型的勝負預測仍可參考個別比賽卡。"
                f"請賽前 2-4 小時再重新生成報告。"
            ),
            "source": "no-odds-fallback",
        }
        # Don't cache this — let it regenerate when odds become available.
        return result

    all_contexts = []
    for i, (game_key, game) in enumerate(games_with_spread.items(), 1):
        all_contexts.append(f"=== 第{i}場 ===\n{_build_game_context(game, game_date)}")

    prompt = DAILY_REPORT_PROMPT.format(
        game_date=game_date,
        num_games=len(games_with_spread),
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
