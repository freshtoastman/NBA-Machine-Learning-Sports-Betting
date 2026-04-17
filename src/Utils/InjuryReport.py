"""Fetch today's NBA injury report via AI web search.

Provides a standalone injury lookup that doesn't depend on the full
AIAnalysis pipeline. Uses Gemini w/ Google Search grounding (preferred)
or falls back to Claude. Results are cached per (date, team) pair.

Usage:
    from src.Utils.InjuryReport import get_injury_report
    report = get_injury_report("Los Angeles Lakers", "2026-04-17")
    # => {"players": [{"name": "LeBron James", "status": "OUT", "injury": "left knee"}], ...}
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime
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
            CREATE TABLE IF NOT EXISTS injury_reports (
                cache_key TEXT PRIMARY KEY,
                game_date TEXT,
                team_name TEXT,
                report_json TEXT,
                created_at TEXT
            )
        """)


def _cache_key(game_date: str, team: str) -> str:
    return hashlib.md5(f"injury:{game_date}:{team}".encode()).hexdigest()


def _get_cached(game_date: str, team: str) -> dict | None:
    _init_cache()
    key = _cache_key(game_date, team)
    with sqlite3.connect(CACHE_DB) as con:
        row = con.execute(
            "SELECT report_json FROM injury_reports WHERE cache_key = ?", (key,)
        ).fetchone()
    if row:
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return None
    return None


def _set_cache(game_date: str, team: str, data: dict):
    _init_cache()
    key = _cache_key(game_date, team)
    with sqlite3.connect(CACHE_DB) as con:
        con.execute(
            """INSERT OR REPLACE INTO injury_reports
               (cache_key, game_date, team_name, report_json, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (key, game_date, team, json.dumps(data, ensure_ascii=False),
             datetime.now().isoformat()),
        )


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

INJURY_PROMPT = """Search for the latest NBA injury report for the {team} as of {game_date}.

Return ONLY a JSON object (no code block, no explanation) with this exact format:
{{
  "team": "{team}",
  "date": "{game_date}",
  "players": [
    {{
      "name": "Player Name",
      "status": "OUT" or "Doubtful" or "Questionable" or "Probable" or "Available",
      "injury": "brief injury description",
      "impact": "star" or "rotation" or "bench"
    }}
  ],
  "summary": "One sentence impact summary in Traditional Chinese",
  "source": "where you found this info"
}}

Rules:
- Only include players who are actually injured, questionable, or recently returned
- "impact" = "star" for top-3 players, "rotation" for regular rotation, "bench" for deep bench
- If no injuries found, return empty players array with summary "全員健康"
- Do NOT fabricate injuries — only report what you find from recent sources
"""


# ---------------------------------------------------------------------------
# AI calls
# ---------------------------------------------------------------------------

def _call_gemini(prompt: str) -> str | None:
    key = os.environ.get("GEMINI_API_KEY")
    if not key or genai is None:
        return None
    try:
        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={"tools": [{"google_search": {}}]},
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
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except Exception:
        return None


def _parse_json(text: str) -> dict | None:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_injury_report(team: str, game_date: str, force: bool = False) -> dict:
    """Get injury report for a team on a specific date.

    Returns dict with: team, date, players[], summary, source.
    """
    if not force:
        cached = _get_cached(game_date, team)
        if cached:
            return cached

    prompt = INJURY_PROMPT.format(team=team, game_date=game_date)
    result = None

    # Try Gemini with Google Search grounding (best for live data).
    raw = _call_gemini(prompt)
    if raw:
        result = _parse_json(raw)
        if result:
            result["source"] = result.get("source", "gemini+search")

    # Fallback: Claude (no web access, but can use training data).
    if result is None:
        raw = _call_claude(prompt)
        if raw:
            result = _parse_json(raw)
            if result:
                result["source"] = "claude (may be outdated)"

    # Fallback: empty report.
    if result is None:
        result = {
            "team": team,
            "date": game_date,
            "players": [],
            "summary": "無法取得傷兵資訊（API 離線）",
            "source": "unavailable",
        }

    _set_cache(game_date, team, result)
    return result


def get_game_injuries(home_team: str, away_team: str, game_date: str,
                      force: bool = False) -> dict:
    """Get injury reports for both teams in a matchup.

    Returns dict with: home_injuries, away_injuries, impact_summary.
    """
    home_report = get_injury_report(home_team, game_date, force=force)
    away_report = get_injury_report(away_team, game_date, force=force)

    # Count star/rotation players out.
    home_stars_out = sum(
        1 for p in home_report.get("players", [])
        if p.get("impact") == "star" and p.get("status") in ("OUT", "Doubtful")
    )
    away_stars_out = sum(
        1 for p in away_report.get("players", [])
        if p.get("impact") == "star" and p.get("status") in ("OUT", "Doubtful")
    )

    if home_stars_out > away_stars_out:
        impact = f"主場 {home_team} 缺少更多主力，客場有利"
    elif away_stars_out > home_stars_out:
        impact = f"客場 {away_team} 缺少更多主力，主場有利"
    elif home_stars_out == 0 and away_stars_out == 0:
        impact = "雙方主力無重大缺陣"
    else:
        impact = "雙方主力缺陣程度相近"

    return {
        "home_injuries": home_report,
        "away_injuries": away_report,
        "impact_summary": impact,
        "home_stars_out": home_stars_out,
        "away_stars_out": away_stars_out,
    }
