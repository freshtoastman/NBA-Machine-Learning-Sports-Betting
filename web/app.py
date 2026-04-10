"""Lightweight Flask app for Vercel deployment.

Reads pre-computed JSON from data/ directory — no model inference,
no tensorflow/xgboost/pandas dependencies. Only external API calls
are AI analysis (Gemini) and player data (RapidAPI).
"""
import hashlib
import json
import os
import secrets
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv
from flask import (
    Flask, render_template, jsonify, request,
    redirect, url_for, session, make_response,
)

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))
app.jinja_env.add_extension("jinja2.ext.loopcontrols")


# ---------------------------------------------------------------------------
# Auth: whitelist-based access control
# ---------------------------------------------------------------------------
# Whitelist stored as env var AUTHORIZED_EMAILS (comma-separated).
# Each user gets an access token (sha256 of email + salt).
# Login via /login?token=xxx or /login with email form.

AUTH_SALT = os.environ.get("AUTH_SALT", "nba-ml-2026")
ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "lchao@example.com")


def _get_whitelist() -> set[str]:
    raw = os.environ.get("AUTHORIZED_EMAILS", "")
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


def _make_token(email: str) -> str:
    return hashlib.sha256(f"{email.lower().strip()}:{AUTH_SALT}".encode()).hexdigest()[:16]


def _is_authenticated() -> bool:
    return session.get("authenticated") is True


@app.before_request
def require_auth():
    """Gate all pages behind auth except /login and /subscribe."""
    allowed = {"/login", "/subscribe", "/favicon.ico"}
    if request.path in allowed or request.path.startswith("/static"):
        return
    if _is_authenticated():
        return
    # Check token in query string (magic link).
    token = request.args.get("token")
    if token:
        whitelist = _get_whitelist()
        for email in whitelist:
            if _make_token(email) == token:
                session["authenticated"] = True
                session["email"] = email
                return redirect(request.path)
        return redirect(url_for("login", error="invalid"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = request.args.get("error")
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        whitelist = _get_whitelist()
        if email in whitelist:
            session["authenticated"] = True
            session["email"] = email
            return redirect(url_for("index"))
        error = "unauthorized"
    return render_template("login.html", error=error, admin_email=ADMIN_EMAIL)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/subscribe")
def subscribe():
    return render_template("subscribe.html", admin_email=ADMIN_EMAIL)

DATA_DIR = Path(__file__).parent / "data"
TAIPEI_TZ = ZoneInfo("Asia/Taipei")

WEEKDAY_ZH = ["週一", "週二", "週三", "週四", "週五", "週六", "週日"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def today_taipei():
    return datetime.now(TAIPEI_TZ).date()


def load_dates_index():
    p = DATA_DIR / "dates.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"dates": [], "today": date.today().isoformat(), "season": "2025-26"}


def load_date_data(iso_date: str) -> dict | None:
    p = DATA_DIR / f"{iso_date}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def load_season_stats() -> dict | None:
    p = DATA_DIR / "season_stats.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def build_date_chips(selected_date, days=7):
    today = today_taipei()
    chips = []
    for offset in range(days + 1):
        d = today - timedelta(days=offset)
        chips.append({
            "iso": d.isoformat(),
            "label": "今天" if offset == 0 else f"{d.month}/{d.day}",
            "weekday": WEEKDAY_ZH[d.weekday()],
            "is_today": offset == 0,
            "is_selected": d == selected_date,
        })
    return chips


def team_logo_url(team_name):
    """Provided in exported JSON; this is a fallback."""
    return None


def team_name_zh(team_name):
    """Provided in exported JSON; this is a fallback."""
    return team_name


# Register as Jinja globals.
app.jinja_env.globals.update(
    team_logo_url=team_logo_url,
    team_name_zh=team_name_zh,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    today = today_taipei()
    selected_str = request.args.get("date")
    if selected_str:
        try:
            selected_date = date.fromisoformat(selected_str)
        except ValueError:
            selected_date = today
    else:
        selected_date = today

    data = load_date_data(selected_date.isoformat())
    is_today = selected_date == today
    games = data.get("games", {}) if data else {}
    summary = data.get("summary", {"games": 0}) if data else {"games": 0}
    season_stats = load_season_stats()
    idx = load_dates_index()

    # Override team_name_zh / logo from exported data.
    def _zh(name):
        for g in games.values():
            if g.get("home_team") == name:
                return g.get("home_team_zh", name)
            if g.get("away_team") == name:
                return g.get("away_team_zh", name)
        return name

    def _logo(name):
        for g in games.values():
            if g.get("home_team") == name:
                return g.get("home_logo")
            if g.get("away_team") == name:
                return g.get("away_logo")
        return None

    app.jinja_env.globals["team_name_zh"] = _zh
    app.jinja_env.globals["team_logo_url"] = _logo

    return render_template(
        "index.html",
        today=today,
        selected_date=selected_date,
        is_today=is_today,
        date_chips=build_date_chips(selected_date),
        summary=summary,
        season_key=idx.get("season", "2025-26"),
        season_stats=season_stats,
        data={"fanduel": games, "draftkings": {}, "betmgm": {}},
    )


# ---------------------------------------------------------------------------
# API: AI Analysis (calls Gemini, lightweight)
# ---------------------------------------------------------------------------

def _call_gemini(prompt, use_search=True):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    try:
        from google import genai
        client = genai.Client(api_key=key)
        config = {}
        if use_search:
            config["tools"] = [{"google_search": {}}]
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=prompt, config=config,
        )
        return response.text
    except Exception:
        return None


def _parse_json(text):
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = [l for l in cleaned.split("\n") if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(cleaned[start:end])
        except json.JSONDecodeError:
            pass
    return None


@app.route("/api/analysis")
def api_analysis():
    game_date = request.args.get("date")
    home = request.args.get("home")
    away = request.args.get("away")
    if not game_date or not home or not away:
        return jsonify({"error": "Missing params"}), 400

    data = load_date_data(game_date)
    if not data:
        return jsonify({"error": "No data for date"}), 404
    game = data["games"].get(f"{away}:{home}")
    if not game:
        return jsonify({"error": "Game not found"}), 404

    prompt = f"""你是NBA專業分析師。分析這場比賽，搜尋最新傷兵消息。
{away} @ {home} ({game_date})
讓分: {game.get('spread')}
ML模型: {'主隊' if game.get('winner')=='home' else '客隊'} ({game.get('home_confidence') if game.get('winner')=='home' else game.get('away_confidence')}%)
ATS模型: {'主隊' if game.get('ats_model_pick')=='home' else '客隊'} 蓋 ({game.get('ats_model_confidence')}%)
鑽石: {'是' if game.get('is_value') else '否'}

用JSON回答(不要code block):
{{"injuries_home":["球員-狀態",...],"injuries_away":["球員-狀態",...],"key_factors":["因素1","因素2"],"ml_verdict":"看好哪隊","ml_reason":"理由","ats_verdict":"押哪邊讓分","ats_reason":"理由","ou_verdict":"大/小分","ou_reason":"理由","confidence":"高/中/低","summary":"50字總結"}}
用繁體中文。"""

    raw = _call_gemini(prompt, use_search=True) or _call_gemini(prompt, use_search=False)
    result = _parse_json(raw)
    if result:
        result["source"] = "gemini"
    else:
        result = {
            "injuries_home": [], "injuries_away": [],
            "key_factors": ["無法取得即時資料"],
            "ml_verdict": game.get("home_team") if game.get("winner") == "home" else game.get("away_team"),
            "ml_reason": f"模型信心 {game.get('home_confidence') if game.get('winner')=='home' else game.get('away_confidence')}%",
            "ats_verdict": "見讓分模型",
            "ats_reason": f"ATS edge {game.get('ats_value_edge', 0)}pp",
            "ou_verdict": game.get("ou_pick", "?"),
            "ou_reason": f"信心 {game.get('ou_confidence', '?')}%",
            "confidence": "中",
            "summary": "AI 分析暫時無法使用，請參考模型預測。",
            "source": "fallback",
        }
    return jsonify(result)


@app.route("/api/daily-report")
def api_daily_report():
    game_date = request.args.get("date")
    if not game_date:
        return jsonify({"error": "Missing date"}), 400
    data = load_date_data(game_date)
    if not data:
        return jsonify({"error": "No data"}), 404

    games = data["games"]
    contexts = []
    for i, (k, g) in enumerate(games.items(), 1):
        contexts.append(f"第{i}場: {g.get('away_team')} @ {g.get('home_team')} 讓分{g.get('spread')} ML{'主' if g.get('winner')=='home' else '客'}({g.get('home_confidence') if g.get('winner')=='home' else g.get('away_confidence')}%) 鑽石:{'是' if g.get('is_value') else '否'}")

    prompt = f"""NBA分析師，產出{game_date}當日報告。搜尋傷兵消息。
{len(games)}場比賽:
{chr(10).join(contexts)}
用JSON回答(不要code block):
{{"headline":"一句標題","top_picks":[{{"game":"隊vs隊","pick":"建議","confidence":"高/中/低","reason":"理由"}}],"avoid_games":[{{"game":"隊vs隊","reason":"原因"}}],"injury_impact":["傷兵影響1"],"daily_summary":"100字策略建議"}}
用繁體中文。"""

    raw = _call_gemini(prompt, use_search=True) or _call_gemini(prompt, use_search=False)
    result = _parse_json(raw)
    if not result:
        result = {
            "headline": f"今日 {len(games)} 場比賽",
            "top_picks": [],
            "avoid_games": [],
            "injury_impact": ["無法取得傷兵資訊"],
            "daily_summary": f"{len(games)} 場比賽，請參考各場模型預測。",
        }
    return jsonify(result)


# ---------------------------------------------------------------------------
# API: Player data (RapidAPI, lightweight)
# ---------------------------------------------------------------------------

TEAM_ABBREVIATIONS = {
    "Orlando Magic": "ORL", "Minnesota Timberwolves": "MIN", "Miami Heat": "MIA",
    "Boston Celtics": "BOS", "LA Clippers": "LAC", "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET", "Atlanta Hawks": "ATL", "Cleveland Cavaliers": "CLE",
    "Toronto Raptors": "TOR", "Washington Wizards": "WAS", "Phoenix Suns": "PHO",
    "San Antonio Spurs": "SA", "Chicago Bulls": "CHI", "Charlotte Hornets": "CHA",
    "Philadelphia 76ers": "PHI", "New Orleans Pelicans": "NO", "Sacramento Kings": "SAC",
    "Dallas Mavericks": "DAL", "Houston Rockets": "HOU", "Brooklyn Nets": "BKN",
    "New York Knicks": "NY", "Utah Jazz": "UTA", "Oklahoma City Thunder": "OKC",
    "Portland Trail Blazers": "POR", "Indiana Pacers": "IND", "Milwaukee Bucks": "MIL",
    "Golden State Warriors": "GS", "Memphis Grizzlies": "MEM", "Los Angeles Lakers": "LAL",
}


def _rapidapi_headers():
    return {
        "x-rapidapi-key": os.environ.get("RAPIDAPI_KEY", ""),
        "x-rapidapi-host": "tank01-fantasy-stats.p.rapidapi.com",
    }


@app.route("/team-data/<path:team_name>")
def team_data(team_name):
    from urllib.parse import unquote
    team_name = unquote(team_name)
    abv = TEAM_ABBREVIATIONS.get(team_name)
    if not abv:
        return jsonify({"success": False, "error": f"Unknown team: {team_name}"})
    try:
        r = requests.get(
            "https://tank01-fantasy-stats.p.rapidapi.com/getNBATeamRoster",
            headers=_rapidapi_headers(), params={"teamAbv": abv},
        )
        data = r.json()
        if data.get("statusCode") == 200:
            roster = data.get("body", {}).get("roster", [])
            players = [{
                "name": p.get("longName"), "shortName": p.get("shortName"),
                "headshot": p.get("nbaComHeadshot"),
                "injury": (p.get("injury", {}).get("designation", "Healthy")
                           if p.get("injury") else "Healthy"),
                "position": p.get("pos"), "height": p.get("height"),
                "weight": p.get("weight"), "jerseyNum": p.get("jerseyNum"),
                "playerId": p.get("playerID"),
            } for p in roster]
            return jsonify({"success": True, "players": players})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)})
    return jsonify({"success": False, "error": "API error"})


@app.route("/player-stats/<path:player_id>")
def player_stats(player_id):
    headers = _rapidapi_headers()
    try:
        info_r = requests.get(
            "https://tank01-fantasy-stats.p.rapidapi.com/getNBAPlayerInfo",
            headers=headers, params={"playerID": player_id},
        )
        games_r = requests.get(
            "https://tank01-fantasy-stats.p.rapidapi.com/getNBAGamesForPlayer",
            headers=headers, params={"playerID": player_id, "season": "2025"},
        )
        info = info_r.json()
        games_data = games_r.json()
        if info.get("statusCode") == 200 and games_data.get("statusCode") == 200:
            games = sorted(games_data["body"].values(), key=lambda x: x["gameID"], reverse=True)[:10]
            player = info["body"]
            return jsonify({
                "success": True, "games": games,
                "player": {
                    "name": player.get("longName"), "position": player.get("pos"),
                    "number": player.get("jerseyNum"), "height": player.get("height"),
                    "weight": player.get("weight"), "team": player.get("team"),
                    "headshot": player.get("nbaComHeadshot"),
                    "injury": player.get("injury") or "Healthy",
                },
            })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)})
    return jsonify({"success": False, "error": "API error"})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
