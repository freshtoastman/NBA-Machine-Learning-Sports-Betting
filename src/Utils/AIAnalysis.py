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

def _build_playoff_context_block(game: dict, h_zh: str, a_zh: str, game_date: str) -> str:
    """Return a multi-line playoff context block, or empty string if not playoff."""
    try:
        from src.Utils.PlayoffContext import get_series_state
    except Exception:
        return ""
    home = game.get("home_team", "")
    away = game.get("away_team", "")
    state = get_series_state(home, away, game_date)
    if not state:
        return ""

    must_win_team = None
    if state["is_must_win_for"] == "home":
        must_win_team = h_zh
    elif state["is_must_win_for"] == "away":
        must_win_team = a_zh

    leader_zh = h_zh if state["leader"] == home else a_zh if state["leader"] == away else None
    if leader_zh:
        status = f"{leader_zh}領先 {max(state['home_wins'], state['away_wins'])}-{min(state['home_wins'], state['away_wins'])}"
    else:
        status = f"平手 {state['home_wins']}-{state['away_wins']}"

    elim_line = ""
    if state["is_elimination"]:
        elim_line = f"\n  🔴 淘汰局: {must_win_team} 今晚輸了即遭淘汰"

    return f"""
季後賽脈絡:
  輪次: {state['round_label']}（系列賽第 {state['series_game_num']} 場 / 7 戰 4 勝制）
  系列賽狀態: {status}
  主場: {h_zh}（系列賽中目前主場記錄略，整體系列賽 {state['home_wins']}-{state['away_wins']}）{elim_line}
"""


def _build_backtest_signals(game: dict) -> list[dict]:
    """Compute backtest-proven ATS strategy signals for a game.

    Each signal: {'text': str, 'tier': 'hot'|'warm'|'base'|'danger'}
    Same logic as the Jinja2 checklist in index.html.
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

    # Profile-derived form
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

    from src.Utils.Teams import team_name_zh
    h_zh = team_name_zh(game.get("home_team", "?"))
    a_zh = team_name_zh(game.get("away_team", "?"))

    # ── Regular season strategies ──
    if not is_po:
        if abs_sp >= 5 and h_wr_1m <= 35:
            signals.append({"text": f"主場弱隊大讓分 → 押客場/受讓（歷史 70% WR）", "tier": "hot"})
        if abs_sp >= 5 and abs_sp < 10 and h_wr_1m <= 35:
            signals.append({"text": f"讓分 5-10 + 主場弱隊 → 押冷門（70.1%）", "tier": "hot"})
        if h_ats_3m <= 35:
            signals.append({"text": f"{h_zh}近期 ATS 差 → 押客場 cover（61.7%）", "tier": "warm"})
        if a_wr_1m >= 65:
            signals.append({"text": f"{a_zh} L10 強隊 → 押客場 cover（60.1%）", "tier": "warm"})
        if a_ats_3m >= 65:
            signals.append({"text": f"{a_zh} ATS 近期熱 → 押客場 cover（59.0%）", "tier": "warm"})
        if not home_fav and h_season_wr >= 55:
            signals.append({"text": f"主場受讓 + 例行賽強隊 → 押主場 cover（61.5%）", "tier": "warm"})
        if abs_sp >= 5 and abs_sp < 10:
            signals.append({"text": "中讓分(5-10) → 冷門 cover 傾向（58.9%）", "tier": "base"})

    # ── Playoff strategies: use live PlayoffATSStrategy picks when available ──
    if is_po:
        po_picks = game.get("playoff_ats_picks") or []
        if po_picks:
            # Convert PlayoffATSStrategy picks → signal dicts for the AI prompt.
            tier_to_ai = {"GOLD": "hot", "SILVER": "warm", "BRONZE": "base"}
            for pick in po_picks:
                wr_pct = int(round(pick.get("backtest_wr", 0) * 100))
                roi = pick.get("backtest_roi", 0)
                n = pick.get("backtest_n", 0)
                reason = pick.get("reason_zh", "")
                signal_name = pick.get("signal", "")
                ats_side = pick.get("ats_side", "")
                tier = tier_to_ai.get(pick.get("tier", "BRONZE"), "base")
                text = (
                    f"【季後賽回測】{signal_name}：{reason} "
                    f"（回測 {wr_pct}% WR, +{roi}% ROI, n={n}）→ {ats_side}"
                )
                signals.append({"text": text, "tier": tier})
        else:
            # Fallback static rules when playoff_ats_picks not yet computed.
            if series_gn >= 6:
                signals.append({"text": f"Game {series_gn} → 押客場/冷門（63% WR, +20% ROI）", "tier": "hot"})
            if series_gn == 5 and series_lead == 0:
                signals.append({"text": "Game 5（2-2 平手）→ 押主場 cover（60%）", "tier": "hot"})
            if is_elim:
                signals.append({"text": "淘汰局 → 押冷門 cover（56.7%）", "tier": "warm"})
            if is_elim and abs_sp <= 8:
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
    # We avoid +/- sign syntax in the pick options because the LLM keeps
    # flipping them. Instead we use explicit Chinese wording:
    #   「押 A 讓 N 分」    = A is favored by N (gives up N points)
    #   「押 A 受讓 N 分」  = A is the underdog getting N points
    spread = game.get("spread")
    if spread is None:
        spread_str = "N/A"
        pick_option_a = pick_option_b = None
    elif spread > 0:
        abs_sp = f"{spread:.1f}"
        spread_str = f"{h_zh} 讓 {abs_sp} 分 / {a_zh} 受讓 {abs_sp} 分"
        pick_option_a = f"押 {h_zh} 讓 {abs_sp} 分"
        pick_option_b = f"押 {a_zh} 受讓 {abs_sp} 分"
    elif spread < 0:
        abs_sp = f"{-spread:.1f}"
        spread_str = f"{a_zh} 讓 {abs_sp} 分 / {h_zh} 受讓 {abs_sp} 分"
        pick_option_a = f"押 {a_zh} 讓 {abs_sp} 分"
        pick_option_b = f"押 {h_zh} 受讓 {abs_sp} 分"
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

    playoff_block = _build_playoff_context_block(game, h_zh, a_zh, game_date)

    # Backtest strategy signals
    signals = _build_backtest_signals(game)
    if signals:
        tier_label = {"hot": "🔥強", "warm": "⚡中", "base": "📊參考", "danger": "🚫警告"}
        signal_lines = "\n".join(
            f"  [{tier_label.get(s['tier'], '📊')}] {s['text']}" for s in signals
        )
        backtest_block = f"\n回測策略訊號（根據歷史數據回測結果，勝率已驗證）:\n{signal_lines}"
    else:
        backtest_block = ""

    return f"""{a_zh}（主場：{h_zh}）
讓分盤: {spread_str}{playoff_block}
ML模型: {team_name_zh(ml_pick)} ({ml_conf}%)
ATS模型: {team_name_zh(ats_pick) if ats_pick != 'N/A' else 'N/A'}（{ats_conf}%, edge {ats_edge}pp）
鑽石場次: {'是 — ' + team_name_zh(value_team) + f' (edge +{value_edge}pp)' if game.get('is_value') else '否'}
共識: {'🔥 是' if game.get('is_consensus') else '否'}
{_fmt_profile(game.get('home_profile'), h_zh)}
{_fmt_profile(game.get('away_profile'), a_zh)}{pick_options_block}{backtest_block}"""


SINGLE_GAME_PROMPT = """你是職業 NBA 讓分盤分析師。你的讀者是認真的體育投注者，他們需要明確的讓分推薦。

比賽日期: {game_date}
{context}

=== 你的任務 ===
1. 搜尋兩隊今天的 injury report（傷兵報告），列出每個傷兵狀態
2. 評估傷兵對讓分盤的影響（缺主力 = 可能蓋不過讓分）
3. **仔細閱讀「回測策略訊號」**：這些是根據 2007-2026 年 24,000+ 場歷史比賽回測驗證的高勝率情境。🔥強 = 歷史勝率 ≥60%，⚡中 = 勝率 55-60%，🚫警告 = 歷史上此情境勝率極低應避免。你的推薦應該與回測訊號的方向一致，除非有明確的傷兵或特殊消息推翻。
4. 綜合模型 + 回測策略 + 傷兵 + 賽程 + 動機，給出明確讓分推薦
5. 勝負盤（moneyline）賠率通常太低不推薦 — 除非冷門值得小注

=== 回答格式（JSON，不要 code block）===
{{
  "injuries_home": ["球員名 - OUT/GTD/Available（傷勢）", ...],
  "injuries_away": ["球員名 - OUT/GTD/Available（傷勢）", ...],
  "injury_impact": "傷兵對讓分盤的具體影響",
  "key_factors": ["因素1", "因素2", "因素3"],
  "backtest_alignment": "回測訊號支持/反對哪一方（用一句話說明你的推薦與回測訊號的關係）",
  "ats_pick": "從上方提供的『選項A』或『選項B』中挑一個，整個字串原封不動複製",
  "ats_reason": "為什麼選這邊（2-3 句，必須提到回測策略訊號的佐證或為何不採用）",
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
- injuries 必須是實際搜尋到的，不能編造
- **backtest_alignment 為必填欄位**：必須引用上方「回測策略訊號」中至少一條訊號，說明你的推薦與該訊號是一致還是相反。若無訊號則填「無相關回測訊號」。
- **ats_reason 中必須提到回測**：在理由中明確引用回測訊號名稱與勝率（如「回測顯示主場弱隊大讓分歷史 70% 冷門 cover，與本場推薦一致」），否則回答視為不完整。"""


PLAYOFF_GAME_PROMPT = """你是職業 NBA 季後賽讓分盤分析師。季後賽 ≠ 常規賽 — 你必須把以下季後賽特有的因素納入分析：

比賽日期: {game_date}
{context}

=== 季後賽特殊考量（必須評估）===
1. **系列賽動能**：誰拿下上一場？領先方通常在保護優勢，落後方背水一戰
2. **淘汰局壓力**：若是淘汰局，落後方戰意爆棚但體能也接近極限；領先方可能有「rest mode」現象
3. **球員輪換縮減 (rotation tightening)**：季後賽核心球員上場時間 ↑，板凳貢獻 ↓ — 主力傷勢影響被放大
4. **裁判尺度**：季後賽判罰通常較鬆，肉搏戰 — 利於防守強隊與低位中鋒
5. **主場優勢加成**：季後賽主場勝率歷史平均 ~62%，比常規賽 ~57% 高
6. **教練調整**：第 3 場以後雙方都會做戰術調整（雙人包夾、更換對位）
7. **Game 1 警訊**：首場樣本太雜（球隊還在試對位），units 應降一級

=== 你的任務 ===
1. 搜尋兩隊今天的 injury report，特別關注關鍵球員（Top 3 rotation player）
2. 評估系列賽當前狀態對讓分盤的影響
3. **仔細閱讀「回測策略訊號」**：這些標有【季後賽回測】的訊號是由機器學習系統根據 13 季（2012-2025）、996 場季後賽實際資料驗證得出的高勝率策略，每條都附有真實回測勝率（WR）、ROI、樣本數（n）。🔥強 = 歷史勝率 ≥60%，⚡中 = 勝率 55-60%，🚫警告 = 此情境歷史上勝率極低應避免。**這些訊號是你最重要的參考依據**，你的推薦必須與訊號方向一致，除非有確鑿的傷兵或突發消息能推翻。
4. 綜合模型 + 回測策略 + 傷兵 + 系列賽脈絡，給出明確讓分推薦

=== 回答格式（JSON，不要 code block）===
{{
  "injuries_home": ["球員名 - OUT/GTD/Available（傷勢）", ...],
  "injuries_away": ["球員名 - OUT/GTD/Available（傷勢）", ...],
  "injury_impact": "傷兵對讓分盤的具體影響",
  "series_context": "用 1-2 句描述系列賽動能（例：湖人 3-1 領先，Game 5 主場想直接收尾）",
  "key_factors": ["淘汰壓力 / rotation 縮減 / 傷情 / 動能 / 回測策略 等因素，至少 3 個"],
  "backtest_alignment": "回測訊號支持/反對哪一方（一句話說明推薦與回測訊號的關係）",
  "ats_pick": "從上方提供的『選項A』或『選項B』中挑一個，整個字串原封不動複製",
  "ats_reason": "為什麼選這邊（2-3 句，必須提到系列賽脈絡與回測策略佐證）",
  "ats_units": 1到5的整數,
  "ml_note": "勝負盤備註（季後賽冷門較少，通常賠率偏低）",
  "risk_warning": "翻車風險",
  "summary": "一句話結論"
}}

=== 嚴格規則 ===
- **ats_pick 必須從『讓分推薦只能從以下兩個字串擇一』那段中複製整段字串。**
- 若是 Game 1，ats_units 上限 3
- 若是淘汰局且模型 edge < 8pp，ats_units 上限 2（淘汰局變數太大）
- 隊名一律用繁體中文
- injuries 必須是實際搜尋到的，不能編造
- **backtest_alignment 為必填欄位**：必須引用上方「回測策略訊號」中至少一條【季後賽回測】訊號，說明推薦與訊號的關係（例如「小讓分客場受讓訊號支持押客場，與本場推薦一致」）。
- **ats_reason 中必須提到回測**：明確引用訊號名稱與其回測 WR（例如「季後賽回測顯示此情境 57.8% WR，支持押客場受讓」），不能泛泛而談。"""


DAILY_REPORT_PROMPT = """你是職業 NBA 讓分盤分析師團隊主管。你的任務是產出一份讓投注者能直接照著操作的當日報告。

日期: {game_date}
比賽數量: {num_games}

{all_contexts}

=== 你的任務 ===
1. 搜尋今天所有重大傷兵消息
2. **仔細閱讀每場比賽的「回測策略訊號」**：這些是歷史回測驗證的高勝率情境。🔥強 = 歷史 ≥60%，⚡中 = 55-60%，🚫警告 = 應避免。優先挑選有多個 🔥/⚡ 訊號支持的場次作為 best_bets。
3. 從每場比賽提供的『選項A / 選項B』中，挑出值得下注的讓分推薦
4. 標出應該避開的比賽（特別留意有 🚫警告 訊號的場次）
5. 寫出整體策略

=== 回答格式（JSON，不要 code block）===
{{
  "headline": "今日 NBA 讓分盤精選（一句吸引眼球的標題）",
  "best_bets": [
    {{
      "game": "客隊 @ 主隊（繁體中文隊名）",
      "spread": "從該場 context 複製『讓分盤:』那行的內容",
      "pick": "從該場 context 提供的『選項A 或 選項B』中挑一個，整段原封不動複製",
      "units": 1到5的整數,
      "reason": "2-3 句具體分析（含傷兵影響 + 回測策略佐證）",
      "model_support": "模型 + 回測佐證（ML 信心、ATS edge、回測訊號方向）"
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

    # Include backtest signals in rule-based factors
    signals = _build_backtest_signals(game)
    for sig in signals:
        if sig["tier"] in ("hot", "warm", "danger"):
            factors.append(f"[回測] {sig['text']}")

    for side, zh, prof in [("home", h_zh, game.get("home_profile")),
                            ("away", a_zh, game.get("away_profile"))]:
        m = (prof or {}).get("last_1m", {}).get("overall", {})
        if m.get("games", 0) >= 5:
            if m.get("win_rate", 50) >= 70:
                factors.append(f"{zh} 近一個月火燙 ({m['win_rate']}%)")
            elif m.get("win_rate", 50) <= 35:
                factors.append(f"{zh} 近一個月低迷 ({m['win_rate']}%)")

    conf = "高" if (ml_conf or 0) >= 70 or game.get("is_consensus") else "中" if (ml_conf or 0) >= 60 else "低"

    # Build backtest_alignment summary
    bt_hot = [s for s in signals if s["tier"] == "hot"]
    bt_warn = [s for s in signals if s["tier"] == "danger"]
    if bt_hot:
        bt_align = "回測強訊號: " + "; ".join(s["text"] for s in bt_hot)
    elif bt_warn:
        bt_align = "回測警告: " + "; ".join(s["text"] for s in bt_warn)
    elif signals:
        bt_align = "回測參考: " + signals[0]["text"]
    else:
        bt_align = ""

    return {
        "injuries_home": ["無法查詢（離線模式）"],
        "injuries_away": ["無法查詢（離線模式）"],
        "key_factors": factors or ["無特殊因素"],
        "backtest_alignment": bt_align,
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

    # Branch on playoff vs regular season prompt.
    is_playoff_game = False
    try:
        from src.Utils.PlayoffContext import is_playoff_date, get_series_state
        is_playoff_game = bool(
            is_playoff_date(game_date)
            and get_series_state(home, away, game_date) is not None
        )
    except Exception:
        is_playoff_game = False

    template = PLAYOFF_GAME_PROMPT if is_playoff_game else SINGLE_GAME_PROMPT
    prompt = template.format(game_date=game_date, context=context)

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
