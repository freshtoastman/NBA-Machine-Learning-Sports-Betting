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

    # If "today" requested has no exported JSON, auto-fall back to the most
    # recent date that does (so the user lands on real content instead of an
    # empty page).
    no_games_today = False
    data = load_date_data(selected_date.isoformat())
    if data is None and selected_date == today:
        idx_tmp = load_dates_index()
        latest_iso = (idx_tmp.get("dates") or [None])[0]
        if latest_iso:
            try:
                selected_date = date.fromisoformat(latest_iso)
                data = load_date_data(latest_iso)
                no_games_today = True
            except ValueError:
                pass

    is_today = selected_date == today
    games = data.get("games", {}) if data else {}
    summary = data.get("summary", {"games": 0}) if data else {"games": 0}
    active_series = data.get("active_series", []) if data else []
    is_playoff_view = data.get("is_playoff_view", False) if data else False
    season_stats = load_season_stats()
    idx = load_dates_index()

    # Build a name→game lookup so team_name_zh / team_logo_url can resolve any
    # team referenced from active_series (which has team names not in `games`).
    name_to_game = {}
    for g in games.values():
        for side in ("home_team", "away_team"):
            if g.get(side):
                name_to_game[g[side]] = g

    # Override team_name_zh / logo from exported data. For names not in this
    # day's games (e.g. teams in active_series tracker), return None so the
    # template's `{% if team_logo_url(...) %}` skips the image.
    def _zh(name):
        g = name_to_game.get(name)
        if g:
            return g.get("home_team_zh") if g.get("home_team") == name else g.get("away_team_zh") or name
        return name

    def _logo(name):
        g = name_to_game.get(name)
        if g:
            return g.get("home_logo") if g.get("home_team") == name else g.get("away_logo")
        return None

    app.jinja_env.globals["team_name_zh"] = _zh
    app.jinja_env.globals["team_logo_url"] = _logo

    return render_template(
        "index.html",
        today=today,
        selected_date=selected_date,
        is_today=is_today,
        no_games_today=no_games_today,
        date_chips=build_date_chips(selected_date),
        summary=summary,
        season_key=idx.get("season", "2025-26"),
        season_stats=season_stats,
        active_series=active_series,
        is_playoff_view=is_playoff_view,
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

    ctx = _build_game_context(game)
    prompt = f"""你是職業 NBA 讓分盤分析師。你的讀者是認真的體育投注者。

比賽日期: {game_date}
{ctx}

=== 任務 ===
1. 搜尋兩隊今天的 injury report（傷兵報告）
2. 評估傷兵對讓分盤的影響
3. 給出明確的讓分推薦（從上方『選項A/選項B』中擇一）

=== 回答格式（JSON，不要 code block）===
{{
  "injuries_home": ["球員名 - OUT/GTD/Available（傷勢）", ...],
  "injuries_away": ["球員名 - OUT/GTD/Available（傷勢）", ...],
  "injury_impact": "傷兵對讓分盤的具體影響",
  "key_factors": ["因素1", "因素2", "因素3"],
  "ats_pick": "從上方『選項A』或『選項B』中挑一個整段原封不動複製",
  "ats_reason": "為什麼選這一邊（2-3句具體分析）",
  "ats_units": 數字1到5,
  "ou_pick": "大分 N / 小分 N / 不推薦",
  "ou_reason": "一句話理由",
  "ml_note": "勝負盤備註（通常'賠率過低不建議'）",
  "risk_warning": "這場的風險",
  "summary": "一句話結論：押什麼、幾個單位"
}}

嚴格規則：
- ats_pick 必須從『選項A』或『選項B』原封不動複製，不能改任何字、不能用 +/- 符號、不能用英文隊名
- ats_units 必須是數字 1-5
- 用繁體中文回答"""

    raw = _call_gemini(prompt, use_search=True) or _call_gemini(prompt, use_search=False)
    result = _parse_json(raw)
    if result:
        result["source"] = "gemini"
    else:
        home_zh = game.get("home_team_zh") or game.get("home_team", "?")
        away_zh = game.get("away_team_zh") or game.get("away_team", "?")
        ml_pick = home_zh if game.get("winner") == "home" else away_zh
        ats_side = game.get("ats_model_pick")
        ats_team = home_zh if ats_side == "home" else away_zh if ats_side == "away" else "N/A"
        sp = game.get("spread")
        # Rule-based ATS pick using 讓/受讓 wording.
        if ats_team != "N/A" and sp is not None:
            abs_sp = f"{abs(sp):.1f}"
            # Model picks the favorite iff (home picked AND home favored) OR (away picked AND away favored).
            is_fav = (ats_side == "home" and sp > 0) or (ats_side == "away" and sp < 0)
            ats_pick_str = f"押 {ats_team} {'讓' if is_fav else '受讓'} {abs_sp} 分"
        else:
            ats_pick_str = "不推薦"
        result = {
            "injuries_home": [], "injuries_away": [],
            "injury_impact": "無法查詢傷兵（離線模式）",
            "key_factors": ["請參考模型預測"],
            "ats_pick": ats_pick_str,
            "ats_reason": f"ATS 模型 edge {game.get('ats_value_edge', 0)}pp",
            "ats_units": 2 if game.get("ats_is_value") else 1,
            "ou_pick": f"{'大分' if game.get('ou_pick')=='OVER' else '小分'} {game.get('ou_value','?')}",
            "ou_reason": f"模型信心 {game.get('ou_confidence','?')}%",
            "ml_note": "賠率過低不建議",
            "risk_warning": "AI 分析暫時不可用，僅供參考",
            "summary": f"參考 ATS 模型方向下注，{2 if game.get('ats_is_value') else 1} 個單位。",
            "source": "fallback",
        }
    return jsonify(result)


@app.route("/api/analyze-pinned", methods=["POST"])
def api_analyze_pinned():
    """Analyze user's pinned games as a group — find correlations and parlays."""
    body = request.get_json(silent=True) or {}
    game_date = body.get("date")
    pinned_keys = body.get("pinned", [])
    if not game_date or not pinned_keys:
        return jsonify({"error": "Missing date or pinned list"}), 400

    data = load_date_data(game_date)
    if not data:
        return jsonify({"error": "No data"}), 404

    pinned_games = []
    for key in pinned_keys:
        g = data["games"].get(key)
        if g:
            pinned_games.append(g)
    if not pinned_games:
        return jsonify({"error": "No valid pinned games found"}), 404

    contexts = []
    for i, g in enumerate(pinned_games, 1):
        contexts.append(f"=== 置頂場次 {i} ===\n{_build_game_context(g)}")

    prompt = f"""你是職業 NBA 讓分盤分析師。用戶從今天 {len(data['games'])} 場比賽中置頂了 {len(pinned_games)} 場要你重點分析。

日期: {game_date}

{chr(10).join(contexts)}

=== 任務 ===
1. 搜尋這 {len(pinned_games)} 場的傷兵
2. 每場給出明確讓分推薦（從該場提供的『選項A/選項B』中擇一）
3. 分析這幾場之間有沒有關聯（同一隊連戰、跨場串關機會）
4. 給出這 {len(pinned_games)} 場的整體下注策略

=== 回答格式（JSON，不要 code block）===
{{
  "picks": [
    {{
      "game": "客隊 @ 主隊（繁體中文隊名）",
      "pick": "從該場『選項A』或『選項B』中挑一個整段原封不動複製",
      "units": 1到5,
      "reason": "2-3句分析（含傷兵）",
      "injuries": ["重要傷兵1", "重要傷兵2"]
    }}
  ],
  "parlay_suggestion": "串關建議（如果有合適的 2-3 場串關機會就寫，沒有就寫'不建議串關'）",
  "total_units": 數字（這幾場總共建議下多少單位）,
  "strategy": "整體策略（100字）"
}}

嚴格規則：
- 每個 pick 必須從該場提供的『選項A』或『選項B』原封不動複製，不能改任何字、不能用 +/- 符號
- units 是數字 1-5
- 隊名一律用繁體中文
- 用繁體中文回答"""

    raw = _call_gemini(prompt, use_search=True) or _call_gemini(prompt, use_search=False)
    result = _parse_json(raw)
    if not result:
        result = {
            "picks": [{"game": f"{g.get('away_team')} @ {g.get('home_team')}", "pick": "參考模型", "units": 1, "reason": "AI 暫時不可用", "injuries": []} for g in pinned_games],
            "parlay_suggestion": "不建議串關",
            "total_units": len(pinned_games),
            "strategy": "AI 分析暫時不可用，請參考各場模型預測。",
        }
    return jsonify(result)


def _build_game_context(g):
    """Build detailed context string for a single game.

    Produces two pre-formatted pick-option strings using 讓/受讓 wording
    (never +/- signs) so the LLM can only copy one verbatim without flipping.
    """
    home_zh = g.get("home_team_zh") or g.get("home_team", "?")
    away_zh = g.get("away_team_zh") or g.get("away_team", "?")
    ml_side = g.get("winner")
    ml_pick_zh = home_zh if ml_side == "home" else away_zh
    ml_conf = g.get("home_confidence") if ml_side == "home" else g.get("away_confidence", 0)
    spread = g.get("spread")

    # Spread convention: positive => home is favorite (gives points).
    if spread is None:
        spread_str = "N/A"
        pick_option_a = pick_option_b = None
    elif spread > 0:
        abs_sp = f"{spread:.1f}"
        spread_str = f"{home_zh} 讓 {abs_sp} 分 / {away_zh} 受讓 {abs_sp} 分"
        pick_option_a = f"押 {home_zh} 讓 {abs_sp} 分"
        pick_option_b = f"押 {away_zh} 受讓 {abs_sp} 分"
    elif spread < 0:
        abs_sp = f"{-spread:.1f}"
        spread_str = f"{away_zh} 讓 {abs_sp} 分 / {home_zh} 受讓 {abs_sp} 分"
        pick_option_a = f"押 {away_zh} 讓 {abs_sp} 分"
        pick_option_b = f"押 {home_zh} 受讓 {abs_sp} 分"
    else:
        spread_str = "PK（無讓分）"
        pick_option_a = f"押 {home_zh}（PK）"
        pick_option_b = f"押 {away_zh}（PK）"

    ats_side = g.get("ats_model_pick")
    ats_team = home_zh if ats_side == "home" else away_zh if ats_side == "away" else "N/A"
    ats_conf = g.get("ats_model_confidence", "?")
    ats_edge = g.get("ats_value_edge", 0)
    ou_pick = g.get("ou_pick", "?")
    ou_val = g.get("ou_value", "?")
    value_side = g.get("value_side")
    value_team = home_zh if value_side == "home" else away_zh if value_side else "無"
    value_edge = g.get("value_edge", 0)

    # Profile summary
    def _prof(p, name):
        if not p:
            return ""
        s = p.get("season", {}).get("overall", {})
        m = p.get("last_1m", {}).get("overall", {})
        parts = []
        if s.get("games"):
            parts.append(f"賽季{s['wins']}-{s['losses']}({s['win_rate']}%)")
        if m.get("games"):
            parts.append(f"近月{m['wins']}-{m['losses']}({m['win_rate']}%)")
        return f"{name}: {' '.join(parts)}" if parts else ""
    hp = _prof(g.get("home_profile"), home_zh)
    ap = _prof(g.get("away_profile"), away_zh)

    pick_block = ""
    if pick_option_a and pick_option_b:
        pick_block = (
            f"\n讓分推薦只能從以下兩個字串擇一（複製貼上，不要改任何字）：\n"
            f"  選項A: {pick_option_a}\n"
            f"  選項B: {pick_option_b}"
        )

    return f"""{away_zh} @ {home_zh} | 讓分盤: {spread_str}
ML: {ml_pick_zh}({ml_conf}%) | ATS: {ats_team}({ats_conf}%,edge{ats_edge}pp) | OU: {ou_pick} {ou_val}
鑽石ML:{'是 '+value_team+f'(+{value_edge}pp)' if g.get('is_value') else '否'} | ATS鑽石:{'是' if g.get('ats_is_value') else '否'} | 共識:{'🔥是' if g.get('is_consensus') else '否'}
{hp}
{ap}{pick_block}"""


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
        contexts.append(f"=== 第{i}場 ===\n{_build_game_context(g)}")

    prompt = f"""你是職業 NBA 讓分盤分析師團隊主管。產出一份讓投注者能直接照著操作的當日報告。

日期: {game_date}
比賽數量: {len(games)}

{chr(10).join(contexts)}

=== 任務 ===
1. 搜尋今天所有重大傷兵消息
2. 從每場比賽提供的『選項A/選項B』中挑出值得下注的讓分推薦
3. 標出應該避開的比賽
4. 寫出整體策略

=== 回答格式（JSON，不要 code block）===
{{
  "headline": "今日 NBA 讓分盤精選（一句吸引眼球的標題）",
  "best_bets": [
    {{
      "game": "客隊 @ 主隊（繁體中文隊名）",
      "spread": "從該場『讓分盤:』那行複製",
      "pick": "從該場『選項A』或『選項B』中挑一個整段原封不動複製",
      "units": 1到5的數字,
      "reason": "2-3句具體分析（含傷兵影響）",
      "model_support": "模型資料佐證（ML信心、ATS edge等）"
    }}
  ],
  "lean_picks": [
    {{
      "game": "客隊 @ 主隊",
      "pick": "從該場『選項A』或『選項B』中挑一個整段原封不動複製",
      "units": 1到2,
      "reason": "一句話理由"
    }}
  ],
  "avoid_games": [
    {{"game": "客隊 @ 主隊", "reason": "為什麼避開（傷兵不確定/盤口合理/五五波）"}}
  ],
  "injury_alerts": ["重大傷兵消息1", "重大傷兵消息2"],
  "bankroll_plan": "今日資金分配建議（總共幾個單位、怎麼分配）",
  "daily_summary": "150字策略總結：今天哪些盤口有機會、整體市場觀察、風險提醒"
}}

=== 嚴格規則 ===
- best_bets: 最多 3 場，units >= 3
- lean_picks: 次級推薦，units 1-2
- **每個 pick 必須從該場提供的『選項A』或『選項B』原封不動複製，不能改任何字、不能用 +/- 符號、不能用英文隊名**
- units 必須是數字 1-5
- 勝負盤（moneyline）只有在大冷門有價值時才提及
- injury_alerts 必須是搜尋到的真實傷兵消息
- 隊名一律用繁體中文
- 用繁體中文回答"""

    raw = _call_gemini(prompt, use_search=True) or _call_gemini(prompt, use_search=False)
    result = _parse_json(raw)
    if not result:
        # Fallback
        value_games = [g for g in games.values() if g.get("is_value")]
        result = {
            "headline": f"今日 {len(games)} 場比賽，{len(value_games)} 場鑽石",
            "best_bets": [],
            "lean_picks": [],
            "avoid_games": [],
            "injury_alerts": ["無法取得傷兵資訊（離線模式）"],
            "bankroll_plan": f"建議保守操作，等待更好的場次。",
            "daily_summary": f"共 {len(games)} 場比賽，{len(value_games)} 場鑽石訊號。建議集中在鑽石場次下注。",
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
