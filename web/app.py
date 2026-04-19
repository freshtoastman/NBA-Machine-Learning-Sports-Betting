"""Lightweight Flask app for Vercel deployment.

Reads pre-computed JSON from data/ directory — no model inference,
no tensorflow/xgboost/pandas dependencies. Only external API calls
are AI analysis (Gemini) and player data (RapidAPI).
"""
import hashlib
import json
import os
import secrets
import sqlite3
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
# Writable cache dir. /tmp is writable on Vercel lambdas and local dev.
CACHE_DIR = Path("/tmp/nba_ai_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
TAIPEI_TZ = ZoneInfo("Asia/Taipei")


def _cache_get(key: str) -> dict | None:
    p = CACHE_DIR / f"{key}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _cache_put(key: str, value: dict) -> None:
    try:
        (CACHE_DIR / f"{key}.json").write_text(
            json.dumps(value, ensure_ascii=False), encoding="utf-8",
        )
    except OSError:
        pass


def _safe_key(*parts: str) -> str:
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


# ---------------------------------------------------------------------------
# AI Prediction Log — persistent SQLite database
# ---------------------------------------------------------------------------
# Use /tmp on Vercel (web/data/ is read-only in serverless); local dev uses web/data/.
_local_log_path = Path(__file__).parent / "data" / "ai_prediction_log.sqlite"
AI_LOG_DB = _local_log_path if _local_log_path.parent.exists() and os.access(str(_local_log_path.parent), os.W_OK) else CACHE_DIR / "ai_prediction_log.sqlite"


def _init_ai_log_db():
    try:
        con = sqlite3.connect(str(AI_LOG_DB))
        con.execute("""CREATE TABLE IF NOT EXISTS ai_prediction_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_date TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            home_team_zh TEXT,
            away_team_zh TEXT,
            ats_pick TEXT,
            ats_units INTEGER,
            ats_reason TEXT,
            ou_pick TEXT,
            ou_reason TEXT,
            summary TEXT,
            source TEXT,
            is_golden INTEGER DEFAULT 0,
            is_value INTEGER DEFAULT 0,
            spread REAL,
            created_at TEXT NOT NULL,
            ats_correct INTEGER,
            ou_correct INTEGER,
            graded_at TEXT,
            UNIQUE(game_date, home_team, away_team)
        )""")
        con.commit()
        con.close()
    except Exception:
        pass


_init_ai_log_db()


def _log_ai_prediction(game_date: str, home: str, away: str, analysis: dict,
                        game: dict):
    """Save an AI analysis prediction to the log database."""
    now = datetime.now(TAIPEI_TZ).isoformat()
    try:
        con = sqlite3.connect(str(AI_LOG_DB))
        con.execute(
            """INSERT OR REPLACE INTO ai_prediction_log
               (game_date, home_team, away_team, home_team_zh, away_team_zh,
                ats_pick, ats_units, ats_reason, ou_pick, ou_reason, summary,
                source, is_golden, is_value, spread, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (game_date, home, away,
             game.get("home_team_zh", ""),
             game.get("away_team_zh", ""),
             analysis.get("ats_pick", ""),
             analysis.get("ats_units"),
             analysis.get("ats_reason", ""),
             analysis.get("ou_pick", ""),
             analysis.get("ou_reason", ""),
             analysis.get("summary", ""),
             analysis.get("source", ""),
             1 if analysis.get("is_golden") else 0,
             1 if analysis.get("is_value") else 0,
             game.get("spread"),
             now),
        )
        con.commit()
        con.close()
    except Exception:
        pass


def _grade_ai_predictions(game_date: str, games: dict):
    """Grade AI predictions for a given date using actual game outcomes."""
    try:
        con = sqlite3.connect(str(AI_LOG_DB))
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT * FROM ai_prediction_log WHERE game_date = ? AND graded_at IS NULL",
            (game_date,),
        ).fetchall()
        if not rows:
            con.close()
            return
        now = datetime.now(TAIPEI_TZ).isoformat()
        for row in rows:
            key = f"{row['away_team']}:{row['home_team']}"
            g = games.get(key)
            if not g:
                continue
            # Grade ATS pick
            ats_winner = g.get("ats_winner")  # 'home', 'away', 'push', or None
            ats_correct = None
            if ats_winner and ats_winner != "push" and row["ats_pick"]:
                home_zh = g.get("home_team_zh", "")
                away_zh = g.get("away_team_zh", "")
                pick_text = row["ats_pick"]
                if home_zh and home_zh in pick_text:
                    ats_correct = 1 if ats_winner == "home" else 0
                elif away_zh and away_zh in pick_text:
                    ats_correct = 1 if ats_winner == "away" else 0
            elif ats_winner == "push":
                ats_correct = None  # push = no decision
            if ats_correct is not None:
                con.execute(
                    "UPDATE ai_prediction_log SET ats_correct=?, graded_at=? WHERE id=?",
                    (ats_correct, now, row["id"]),
                )
        con.commit()
        con.close()
    except Exception:
        pass


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


def load_weekly_golden(today_: date, days: int = 7) -> dict:
    """Collect every golden-tier pick from the last N days.

    Returns a dict: {
        "days": [{date, label, picks: [...], hits, decided, hit_rate}, ...],
        "total_picks": int, "total_hits": int, "total_decided": int,
        "total_hit_rate": float | None,
    }
    Each pick: {date, game_key, away_zh, home_zh, side_zh, value_edge,
                value_ev, status: "win"|"loss"|"pending"}
    """
    out_days = []
    total_picks = total_hits = total_decided = 0
    for offset in range(days):
        d = today_ - timedelta(days=offset)
        iso = d.isoformat()
        data = load_date_data(iso)
        if data is None:
            continue
        picks = []
        hits = decided = 0
        for key, g in (data.get("games") or {}).items():
            if not g.get("is_golden"):
                continue
            side = g.get("value_side")
            if side == "home":
                side_zh = g.get("home_team_zh") or g.get("home_team")
            else:
                side_zh = g.get("away_team_zh") or g.get("away_team")
            # Result: use ml_correct since value side picks home/away ML.
            ml_correct = g.get("ml_correct")
            if ml_correct is None:
                status = "pending"
            elif ml_correct:
                status = "win"
                hits += 1
                decided += 1
            else:
                status = "loss"
                decided += 1
            picks.append({
                "date": iso,
                "game_key": key,
                "away_zh": g.get("away_team_zh") or g.get("away_team"),
                "home_zh": g.get("home_team_zh") or g.get("home_team"),
                "side": side,
                "side_zh": side_zh,
                "spread": g.get("spread"),
                "value_edge": g.get("value_edge"),
                "value_ev": g.get("value_ev"),
                "status": status,
            })
        if not picks:
            continue
        total_picks += len(picks)
        total_hits += hits
        total_decided += decided
        out_days.append({
            "date": iso,
            "label": f"{d.month}/{d.day}",
            "weekday": WEEKDAY_ZH[d.weekday()],
            "picks": picks,
            "hits": hits,
            "decided": decided,
            "pending": len(picks) - decided,
            "hit_rate": round(hits / decided * 100, 1) if decided else None,
        })
    return {
        "days": out_days,
        "total_picks": total_picks,
        "total_hits": total_hits,
        "total_decided": total_decided,
        "total_pending": total_picks - total_decided,
        "total_hit_rate": round(total_hits / total_decided * 100, 1) if total_decided else None,
    }


def load_bracket() -> dict | None:
    p = DATA_DIR / "bracket.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def bracket_in_window(bracket: dict | None, today_: date) -> bool:
    """True if today falls inside bracket's show_from..show_until window."""
    if not bracket:
        return False
    try:
        sf = date.fromisoformat(bracket.get("show_from") or "")
        su = date.fromisoformat(bracket.get("show_until") or "")
    except ValueError:
        return False
    return sf <= today_ <= su


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
    landed_on_today = is_today or no_games_today
    bracket_data = load_bracket()
    show_bracket_banner = landed_on_today and bracket_in_window(bracket_data, today)
    # Auto-redirect to bracket when: user lands on today, no games today, and
    # the playoff bracket window is active — treat it as "playoffs launching".
    if show_bracket_banner and no_games_today and not request.args.get("nobracket"):
        return redirect(url_for("bracket"))
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

    # Serialize per-game team profiles for the H2H side panel.
    # Done in Python (not Jinja loop) to avoid JS syntax errors from inline tojson.
    game_profiles = {
        key: {"home": g.get("home_profile"), "away": g.get("away_profile")}
        for key, g in games.items()
        if g.get("home_profile") or g.get("away_profile")
    }
    import json as _json
    game_profiles_json = _json.dumps(game_profiles, ensure_ascii=False)

    from datetime import timedelta as _td
    yesterday = today - _td(days=1)
    return render_template(
        "index.html",
        today=today,
        yesterday=yesterday,
        selected_date=selected_date,
        is_today=is_today,
        no_games_today=no_games_today,
        date_chips=build_date_chips(selected_date),
        summary=summary,
        season_key=idx.get("season", "2025-26"),
        season_stats=season_stats,
        active_series=active_series,
        is_playoff_view=is_playoff_view,
        show_bracket_banner=show_bracket_banner,
        bracket_playoff_start=(bracket_data or {}).get("playoff_start"),
        bracket_stage=(bracket_data or {}).get("stage"),
        weekly_golden=load_weekly_golden(today),
        data={"fanduel": games, "draftkings": {}, "betmgm": {}},
        game_profiles_json=game_profiles_json,
    )


@app.route("/bracket")
def bracket():
    bracket_data = load_bracket()
    if not bracket_data:
        return render_template(
            "bracket.html",
            bracket=None,
            today=today_taipei(),
        )
    # Resolve team_name_zh / team_logo_url globally from the bracket itself so
    # the template's existing helpers work (logos are already embedded per team).
    name_to_card = {}
    for side in ("east", "west"):
        for s in bracket_data[side]["seeds"] + bracket_data[side]["lottery"]:
            name_to_card[s["team"]] = s
    app.jinja_env.globals["team_name_zh"] = (
        lambda n: (name_to_card.get(n) or {}).get("team_zh") or n
    )
    app.jinja_env.globals["team_logo_url"] = (
        lambda n: (name_to_card.get(n) or {}).get("logo")
    )
    return render_template(
        "bracket.html",
        bracket=bracket_data,
        today=today_taipei(),
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

    cache_key = f"analysis_{game_date}_{_safe_key(away, home)}"
    if not request.args.get("refresh"):
        cached = _cache_get(cache_key)
        if cached:
            cached["cached"] = True
            # Ensure cached results are also logged (first call may have
            # been before the log feature existed, or the DB was wiped).
            _log_ai_prediction(game_date, home, away, cached, game)
            return jsonify(cached)

    ctx = _build_game_context(game)
    prompt = f"""你是職業 NBA 讓分盤分析師。你的讀者是認真的體育投注者。

比賽日期: {game_date}
{ctx}

=== 任務 ===
1. **評估傷兵影響**：傷兵名單已由 ESPN 即時提供（見上方「傷兵名單」）。不需自行搜尋，直接根據提供的名單評估對讓分盤的影響。
2. **仔細閱讀「回測策略訊號」**：這些是根據歷史比賽回測驗證的高勝率情境。🔥強 = 歷史勝率 ≥60%，⚡中 = 勝率 55-60%，🚫警告 = 應避免。你的推薦應該與回測訊號一致，除非有明確傷兵或特殊消息推翻。
3. **分析「歷史輪廓」**：注意兩隊的近期趨勢（近1月/近3月）、本日週幾的勝率、對強/弱隊的表現差異、ATS cover 率。找出有利或不利的歷史模式。
4. 綜合模型 + 回測策略 + 歷史輪廓 + 傷兵，給出明確讓分推薦（從上方『選項A/選項B』中擇一）

=== 回答格式（JSON，不要 code block）===
{{
  "injury_impact": "傷兵對讓分盤的具體影響（根據已提供的 ESPN 傷兵名單）",
  "key_factors": ["因素1", "因素2", "因素3"],
  "golden_verdict": "你如何看待『金鑽面向』這個因素（是/否、以及它對你的推薦信心的影響方向，不必直接照搬回測數字）",
  "backtest_alignment": "回測訊號支持/反對哪一方（引用至少一條回測訊號，說明推薦與訊號的關係）",
  "profile_insight": "歷史輪廓觀察（引用具體數據：近期趨勢、週幾勝率、對強弱隊表現、ATS cover率）",
  "ats_pick": "從上方『選項A』或『選項B』中挑一個整段原封不動複製",
  "ats_reason": "為什麼選這一邊（2-3句，必須提到回測策略訊號和歷史輪廓的佐證）",
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
- **golden_verdict 為必填**：這是讓你確認你有把「金鑽面向」納入決策的欄位。請在整合傷兵、關鍵因素、回測訊號、歷史輪廓、金鑽面向、市場盤口後**自主判斷**，簡述金鑽因素如何（或為何無法）影響你對這場的信心。不要機械式地依金鑽狀態決定 units — units 應該是你綜合所有面向後的結論
- **backtest_alignment 為必填**：必須引用「回測策略訊號」中至少一條，說明推薦與訊號的關係。若無訊號則填「無相關回測訊號」
- **profile_insight 為必填**：必須引用「歷史輪廓」中的具體數據（勝率%、W-L、ATS cover率），分析近期走勢和本場相關模式
- **ats_reason 必須提到回測和歷史輪廓**：明確引用回測訊號和歷史輪廓數據作為佐證
- 用繁體中文回答"""

    raw = _call_gemini(prompt, use_search=False) or _call_gemini(prompt, use_search=False)
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
        # 金鑽面向只寫觀察，不介入 units 決定（離線模式無法做全面判斷）
        if game.get("is_golden"):
            golden_verdict_str = "金鑽面向：是（鑽石 + |讓分|≤6，屬歷史甜蜜區）"
        elif game.get("is_value"):
            golden_verdict_str = "金鑽面向：否（鑽石但 |讓分|>6，銀鑽區）"
        else:
            golden_verdict_str = "金鑽面向：否（非鑽石場次）"
        result = {
            "injury_impact": "傷兵資料已提供於上方，AI 分析暫時不可用",
            "key_factors": ["請參考模型預測"],
            "golden_verdict": golden_verdict_str,
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
    # Always attach ESPN injury data from prediction JSON (not from AI output)
    result["injuries_home"] = game.get("injuries_home") or []
    result["injuries_away"] = game.get("injuries_away") or []
    # Expose the authoritative golden/value flags so the frontend can style
    # the response without parsing the AI's free-form verdict text.
    result["is_golden"] = bool(game.get("is_golden"))
    result["is_value"] = bool(game.get("is_value"))
    # Log AI prediction for tracking accuracy
    _log_ai_prediction(game_date, home, away, result, game)
    # Only cache successful (non-fallback) results so a transient Gemini outage
    # doesn't pin a degraded answer forever.
    if result.get("source") == "gemini":
        _cache_put(cache_key, result)
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

    cache_key = f"pinned_{game_date}_{_safe_key(*sorted(pinned_keys))}"
    if not body.get("refresh"):
        cached = _cache_get(cache_key)
        if cached:
            cached["cached"] = True
            return jsonify(cached)

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
**每場分析的深度必須和單場 AI 分析完全一樣**：傷兵、關鍵因素、回測訊號引用、歷史輪廓數據、讓分推薦、大小分、勝負盤、風險、總結都要寫完整。

日期: {game_date}

{chr(10).join(contexts)}

=== 任務 ===
1. 搜尋這 {len(pinned_games)} 場的傷兵
2. 每場給出完整的單場分析（和單場 AI 分析一樣的面向）
3. 最後額外分析這幾場之間的關聯（同隊連戰、跨場串關機會）並給出整體下注策略

=== 回答格式（JSON，不要 code block）===
{{
  "picks": [
    {{
      "game": "客隊 @ 主隊（繁體中文隊名）",
      "injuries_home": ["球員名 - OUT/GTD/Available（傷勢）", ...],
      "injuries_away": ["球員名 - OUT/GTD/Available（傷勢）", ...],
      "injury_impact": "傷兵對讓分盤的具體影響",
      "key_factors": ["因素1", "因素2", "因素3"],
      "golden_verdict": "你如何看待『金鑽面向』這個因素（是/否、它對你的信心影響方向），不必機械式決定單位",
      "backtest_alignment": "回測訊號支持/反對哪一方（引用至少一條回測訊號）",
      "profile_insight": "歷史輪廓觀察（引用具體數據：近期勝率%、ATS cover率等）",
      "ats_pick": "從該場『選項A』或『選項B』中挑一個整段原封不動複製",
      "ats_reason": "為什麼選這一邊（2-3句，必須提到回測訊號和歷史輪廓佐證）",
      "ats_units": 數字1到5,
      "ou_pick": "大分 N / 小分 N / 不推薦",
      "ou_reason": "一句話理由",
      "ml_note": "勝負盤備註（通常'賠率過低不建議'）",
      "risk_warning": "這場的風險",
      "summary": "一句話結論：押什麼、幾個單位"
    }}
  ],
  "parlay_suggestion": "串關建議（2-3 腿合適組合，或'不建議串關'）",
  "total_units": 數字（這幾場合計建議單位數）,
  "strategy": "整體策略（100字，說明單位分配和風險控制）"
}}

嚴格規則：
- **picks 陣列的每一個元素必須包含上方所有欄位，少寫任何一個都算失敗**
- 每個 ats_pick 必須從該場提供的『選項A』或『選項B』原封不動複製，不能改任何字、不能用 +/- 符號、不能用英文隊名
- ats_units 必須是數字 1-5
- **golden_verdict 為必填**：確認你有把「金鑽面向」納入整體判斷，說明它對該場推薦信心的影響方向（加成/警告/中立），但 units 必須是你綜合所有面向後的結論，不要因為是金鑽就機械式給 3-5U
- backtest_alignment 和 profile_insight 為必填，必須引用具體數據
- 隊名一律用繁體中文
- 用繁體中文回答"""

    raw = _call_gemini(prompt, use_search=True) or _call_gemini(prompt, use_search=False)
    result = _parse_json(raw)
    if not result:
        # Fallback: build per-pick stubs that still match the per-game schema
        # so the frontend can render them uniformly.
        fb_picks = []
        for g in pinned_games:
            home_zh = g.get("home_team_zh") or g.get("home_team", "?")
            away_zh = g.get("away_team_zh") or g.get("away_team", "?")
            ats_side = g.get("ats_model_pick")
            ats_team = home_zh if ats_side == "home" else away_zh if ats_side == "away" else "N/A"
            sp = g.get("spread")
            if ats_team != "N/A" and sp is not None:
                abs_sp = f"{abs(sp):.1f}"
                is_fav = (ats_side == "home" and sp > 0) or (ats_side == "away" and sp < 0)
                ats_pick_str = f"押 {ats_team} {'讓' if is_fav else '受讓'} {abs_sp} 分"
            else:
                ats_pick_str = "不推薦"
            if g.get("is_golden"):
                gv = "金鑽面向：是（甜蜜區）"
            elif g.get("is_value"):
                gv = "金鑽面向：否（鑽石但 |讓分|>6）"
            else:
                gv = "金鑽面向：否（非鑽石）"
            fb_picks.append({
                "game": f"{away_zh} @ {home_zh}",
                "injuries_home": [], "injuries_away": [],
                "injury_impact": "無法查詢傷兵（離線模式）",
                "key_factors": ["請參考模型預測"],
                "golden_verdict": gv,
                "backtest_alignment": "無相關回測訊號",
                "profile_insight": "離線模式無法取用歷史輪廓",
                "ats_pick": ats_pick_str,
                "ats_reason": f"ATS 模型 edge {g.get('ats_value_edge', 0)}pp",
                "ats_units": 2 if g.get("ats_is_value") else 1,
                "ou_pick": f"{'大分' if g.get('ou_pick') == 'OVER' else '小分'} {g.get('ou_value', '?')}",
                "ou_reason": f"模型信心 {g.get('ou_confidence', '?')}%",
                "ml_note": "賠率過低不建議",
                "risk_warning": "AI 分析暫時不可用，僅供參考",
                "summary": f"參考 ATS 模型方向下注，{2 if g.get('ats_is_value') else 1} 個單位。",
            })
        result = {
            "picks": fb_picks,
            "parlay_suggestion": "不建議串關",
            "total_units": sum(p["ats_units"] for p in fb_picks),
            "strategy": "AI 分析暫時不可用，請參考各場模型預測。",
            "source": "fallback",
        }
    else:
        result.setdefault("source", "gemini")
    # Annotate each pick with the authoritative golden/value flags so the
    # frontend can style without parsing the AI's free-form verdict text.
    for p, g in zip(result.get("picks") or [], pinned_games):
        p["is_golden"] = bool(g.get("is_golden"))
        p["is_value"] = bool(g.get("is_value"))
    if result.get("source") == "gemini":
        _cache_put(cache_key, result)
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

    # Profile summary — full historical context
    def _prof(p, name, side_label):
        if not p:
            return ""
        lines = [f"【{name}（{side_label}）歷史輪廓】"]

        def _fmt(stat, label):
            if not stat or not stat.get("games"):
                return None
            ats = ""
            if stat.get("ats_rate") is not None:
                ats = f" ATS {stat['ats_wins']}-{stat['ats_losses']}({stat['ats_rate']}%)"
            return f"  {label}: {stat['wins']}-{stat['losses']}({stat['win_rate']}%){ats}"

        for period_key, period_label in [("season", "本賽季"), ("last_3m", "近3月"), ("last_1m", "近1月")]:
            period = p.get(period_key, {})
            overall = _fmt(period.get("overall"), "整體")
            weekday = _fmt(period.get("weekday"), p.get("weekday_name", "週日"))
            vs_strong = _fmt(period.get("vs_strong"), "對強隊")
            vs_weak = _fmt(period.get("vs_weak"), "對弱隊")
            row_parts = [x for x in [overall, weekday, vs_strong, vs_weak] if x]
            if row_parts:
                lines.append(f" {period_label}:")
                lines.extend(row_parts)
        return "\n".join(lines) if len(lines) > 1 else ""

    hp = _prof(g.get("home_profile"), home_zh, "主場")
    ap = _prof(g.get("away_profile"), away_zh, "客場")

    pick_block = ""
    if pick_option_a and pick_option_b:
        pick_block = (
            f"\n讓分推薦只能從以下兩個字串擇一（複製貼上，不要改任何字）：\n"
            f"  選項A: {pick_option_a}\n"
            f"  選項B: {pick_option_b}"
        )

    # Backtest signals (self-contained module — works on Vercel).
    from backtest_signals import build_backtest_signals
    signals = build_backtest_signals(g)
    if signals:
        tier_label = {"hot": "🔥強", "warm": "⚡中", "base": "📊參考", "danger": "🚫警告"}
        sig_lines = "\n".join(f"  [{tier_label.get(s['tier'], '📊')}] {s['text']}" for s in signals)
        bt_block = f"\n回測策略訊號（根據歷史數據回測結果，勝率已驗證）:\n{sig_lines}"
    else:
        bt_block = ""

    # Golden tier factor: value pick AND |spread| ≤ 6. Just surface the
    # factual status + historical reference — let the AI weigh it against
    # all the other signals before reaching a verdict.
    if g.get("is_golden"):
        abs_sp = f"{abs(float(spread)):.1f}" if spread is not None else "?"
        golden_line = (
            f"🥇 金鑽面向: 是 (押 {value_team}, +{value_edge}pp edge, |讓分| {abs_sp}) "
            f"— 歷史 4 季回測此類組合平均 +12.8% ROI / 71-79% 命中率"
        )
    elif g.get("is_value"):
        sp_abs = abs(float(spread)) if spread is not None else None
        if sp_abs is not None and sp_abs > 6:
            golden_line = (
                f"🥇 金鑽面向: 否 (鑽石 {value_team}, 但 |讓分| {sp_abs:.1f} > 6, "
                f"歷史此區段平均 +5.7% ROI)"
            )
        else:
            golden_line = f"🥇 金鑽面向: 否 (鑽石 {value_team}, 盤口資料不完整)"
    else:
        golden_line = "🥇 金鑽面向: 否 (不是鑽石場次)"

    # Injury context from ESPN (pre-fetched; empty for historical games)
    h_inj = g.get("injuries_home") or []
    a_inj = g.get("injuries_away") or []
    if h_inj or a_inj:
        inj_lines = []
        if h_inj:
            inj_lines.append(f"  {home_zh}（主隊）: " + "、".join(h_inj))
        if a_inj:
            inj_lines.append(f"  {away_zh}（客隊）: " + "、".join(a_inj))
        inj_block = "\n傷兵名單（ESPN 即時數據）:\n" + "\n".join(inj_lines)
    else:
        inj_block = "\n傷兵名單: 無重要傷兵記錄"

    # Playoff ATS signals with full backtest detail (GOLD/SILVER/BRONZE)
    po_picks = g.get("playoff_ats_picks") or []
    if po_picks:
        tier_label = {"GOLD": "🥇金", "SILVER": "🥈銀", "BRONZE": "🥉銅"}
        po_lines = []
        for p in po_picks:
            tier = tier_label.get(p.get("tier", ""), p.get("tier", ""))
            wr = p.get("backtest_wr", 0)
            roi = p.get("backtest_roi", 0)
            n = p.get("backtest_n", "?")
            reason = p.get("reason_zh", "")
            side = p.get("ats_side", "")
            po_lines.append(f"  [{tier}] {side}: {reason}（回測 {n} 場，勝率 {wr*100:.0f}%，ROI +{roi:.1f}%）")
        po_block = "\n季後賽 ATS 策略訊號（回測驗證）:\n" + "\n".join(po_lines)
    else:
        po_block = ""

    # Moneyline odds and model fair probability
    ml_odds_h = g.get("home_team_odds")
    ml_odds_a = g.get("away_team_odds")
    fair_h = g.get("fair_home")
    fair_a = g.get("fair_away")
    odds_line = ""
    if ml_odds_h is not None and ml_odds_a is not None:
        odds_line = f"\n賠率: {home_zh} {ml_odds_h:+d} / {away_zh} {ml_odds_a:+d}"
        if fair_h is not None:
            odds_line += f" | 模型公允機率: {home_zh} {fair_h:.1f}% / {away_zh} {fair_a:.1f}%"

    return f"""{away_zh} @ {home_zh} | 讓分盤: {spread_str}
ML: {ml_pick_zh}({ml_conf}%) | ATS: {ats_team}({ats_conf}%,edge{ats_edge}pp) | OU: {ou_pick} {ou_val}
鑽石ML:{'是 '+value_team+f'(+{value_edge}pp)' if g.get('is_value') else '否'} | ATS鑽石:{'是' if g.get('ats_is_value') else '否'} | 共識:{'🔥是' if g.get('is_consensus') else '否'}{odds_line}
{golden_line}
{hp}
{ap}{inj_block}{po_block}{pick_block}{bt_block}"""


@app.route("/api/daily-report")
def api_daily_report():
    game_date = request.args.get("date")
    if not game_date:
        return jsonify({"error": "Missing date"}), 400
    data = load_date_data(game_date)
    if not data:
        return jsonify({"error": "No data"}), 404

    games = data["games"]
    # Cache key is bound to the set of games that day — if games change
    # (odds update, new game added), the cache invalidates automatically.
    games_sig = _safe_key(*sorted(games.keys()))
    cache_key = f"daily_{game_date}_{games_sig}"
    if not request.args.get("refresh"):
        cached = _cache_get(cache_key)
        if cached:
            cached["cached"] = True
            return jsonify(cached)

    contexts = []
    for i, (k, g) in enumerate(games.items(), 1):
        contexts.append(f"=== 第{i}場 ===\n{_build_game_context(g)}")

    # Collect all injury alerts from games for the summary
    all_injuries = []
    for g in games.values():
        home_zh_g = g.get("home_team_zh") or g.get("home_team", "?")
        away_zh_g = g.get("away_team_zh") or g.get("away_team", "?")
        for p in g.get("injuries_home") or []:
            all_injuries.append(f"{home_zh_g}（主）: {p}")
        for p in g.get("injuries_away") or []:
            all_injuries.append(f"{away_zh_g}（客）: {p}")
    inj_summary = ("\n傷兵彙整（ESPN 即時數據，已含於各場比賽 context 中）:\n" +
                   "\n".join(f"  {x}" for x in all_injuries)) if all_injuries else "\n傷兵: 無重大傷兵記錄"

    prompt = f"""你是職業 NBA 讓分盤分析師團隊主管。產出一份讓投注者能直接照著操作的當日報告。

日期: {game_date}
比賽數量: {len(games)}
{inj_summary}

{chr(10).join(contexts)}

=== 任務 ===
1. **評估傷兵影響**：各場比賽的傷兵名單已由 ESPN 即時提供（見上方傷兵彙整及各場 context 中「傷兵名單」），不需自行搜尋。直接根據已提供名單評估對讓分盤的影響。
2. **仔細閱讀每場比賽的「回測策略訊號」和「季後賽 ATS 策略訊號」**：🔥強 = 歷史勝率 ≥60%，⚡中 = 55-60%，🚫警告 = 應避免。優先挑選有多個 🔥/⚡ 訊號支持的場次作為 best_bets。
3. **分析每場比賽的「歷史輪廓」**：注意近期趨勢（近1月走勢）、本日週幾的勝率、對強/弱隊的表現、ATS cover 率。走勢好或 ATS cover 率高的隊伍更值得信賴。
4. 從每場比賽提供的『選項A/選項B』中挑出值得下注的讓分推薦
5. 標出應該避開的比賽（特別留意有 🚫警告 訊號的場次、或近期走勢極差的隊伍）
6. 寫出整體策略

=== 回答格式（JSON，不要 code block）===
{{
  "headline": "今日 NBA 讓分盤精選（一句吸引眼球的標題）",
  "best_bets": [
    {{
      "game": "客隊 @ 主隊（繁體中文隊名）",
      "spread": "從該場『讓分盤:』那行複製",
      "pick": "從該場『選項A』或『選項B』中挑一個整段原封不動複製",
      "units": 1到5的數字,
      "reason": "2-3句具體分析（含傷兵影響 + 必須引用該場回測策略訊號和歷史輪廓佐證）",
      "model_support": "模型資料佐證（ML信心、ATS edge等）+ 回測訊號摘要 + 歷史輪廓重點"
    }}
  ],
  "lean_picks": [
    {{
      "game": "客隊 @ 主隊",
      "pick": "從該場『選項A』或『選項B』中挑一個整段原封不動複製",
      "units": 1到2,
      "reason": "一句話理由（引用回測訊號或歷史輪廓）"
    }}
  ],
  "avoid_games": [
    {{"game": "客隊 @ 主隊", "reason": "為什麼避開（傷兵不確定/盤口合理/五五波）"}}
  ],
  "injury_alerts": ["整理 ESPN 傷兵名單中影響讓分盤最重要的球員（例如：湖人隊 Luka Doncic 缺陣，影響大）"],
  "bankroll_plan": "今日資金分配建議（總共幾個單位、怎麼分配）",
  "daily_summary": "150字策略總結：今天哪些盤口有機會、整體市場觀察、風險提醒"
}}

=== 嚴格規則 ===
- best_bets: 最多 3 場，units >= 3
- lean_picks: 次級推薦，units 1-2
- **每個 pick 必須從該場提供的『選項A』或『選項B』原封不動複製，不能改任何字、不能用 +/- 符號、不能用英文隊名**
- units 必須是數字 1-5
- 勝負盤（moneyline）只有在大冷門有價值時才提及
- injury_alerts 必須來自已提供的 ESPN 傷兵名單，彙整影響讓分盤最大的球員缺陣情況
- **best_bets 的 reason 和 model_support 必須引用該場「回測策略訊號」和「歷史輪廓」**：例如「回測顯示主場弱隊大讓分歷史 70% 冷門 cover」或「近1月 ATS cover 率 65%，走勢火熱」。若該場無回測訊號則註明「無相關回測訊號」
- **lean_picks 的 reason 也須提到回測訊號或歷史輪廓**
- **avoid_games 的 reason 可引用歷史輪廓走勢不佳的數據**
- 隊名一律用繁體中文
- 用繁體中文回答"""

    raw = _call_gemini(prompt, use_search=False) or _call_gemini(prompt, use_search=False)
    result = _parse_json(raw)
    if not result:
        # Fallback
        value_games = [g for g in games.values() if g.get("is_value")]
        result = {
            "headline": f"今日 {len(games)} 場比賽，{len(value_games)} 場鑽石",
            "best_bets": [],
            "lean_picks": [],
            "avoid_games": [],
            "injury_alerts": all_injuries[:5] if all_injuries else ["無重大傷兵"],
            "bankroll_plan": f"建議保守操作，等待更好的場次。",
            "daily_summary": f"共 {len(games)} 場比賽，{len(value_games)} 場鑽石訊號。建議集中在鑽石場次下注。",
            "source": "fallback",
        }
    else:
        result.setdefault("source", "gemini")
    if result.get("source") == "gemini":
        _cache_put(cache_key, result)
    return jsonify(result)


# ---------------------------------------------------------------------------
# API: AI Prediction Log — history + stats
# ---------------------------------------------------------------------------

@app.route("/api/ai-log")
def api_ai_log():
    """Return AI prediction log with grading and stats.

    Query params:
      days: number of days to look back (default 30)
      date: specific date to filter (optional)
    """
    if not _is_authenticated():
        return jsonify({"error": "Unauthorized"}), 401

    days = int(request.args.get("days", 30))
    specific_date = request.args.get("date")

    # Try to grade ungraded predictions for recent dates
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    d = start_date
    while d <= end_date:
        data = load_date_data(d.isoformat())
        if data and data.get("games"):
            _grade_ai_predictions(d.isoformat(), data["games"])
        d += timedelta(days=1)

    try:
        con = sqlite3.connect(str(AI_LOG_DB))
        con.row_factory = sqlite3.Row
        if specific_date:
            rows = con.execute(
                "SELECT * FROM ai_prediction_log WHERE game_date = ? ORDER BY created_at DESC",
                (specific_date,),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT * FROM ai_prediction_log WHERE game_date >= ? ORDER BY game_date DESC, created_at DESC",
                (start_date.isoformat(),),
            ).fetchall()
        con.close()
    except Exception:
        return jsonify({"records": [], "stats": {}})

    records = []
    for r in rows:
        records.append({
            "game_date": r["game_date"],
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "home_team_zh": r["home_team_zh"],
            "away_team_zh": r["away_team_zh"],
            "ats_pick": r["ats_pick"],
            "ats_units": r["ats_units"],
            "ats_reason": r["ats_reason"],
            "summary": r["summary"],
            "source": r["source"],
            "is_golden": bool(r["is_golden"]),
            "is_value": bool(r["is_value"]),
            "spread": r["spread"],
            "created_at": r["created_at"],
            "ats_correct": r["ats_correct"],
            "graded_at": r["graded_at"],
        })

    # Compute aggregate stats
    ats_graded = [r for r in records if r["ats_correct"] is not None]
    ats_wins = sum(1 for r in ats_graded if r["ats_correct"] == 1)
    golden_graded = [r for r in ats_graded if r["is_golden"]]
    golden_wins = sum(1 for r in golden_graded if r["ats_correct"] == 1)

    # ROI calculation: assume flat $110 to win $100 per bet (standard -110 juice)
    total_wagered = len(ats_graded) * 110
    total_returned = ats_wins * 210  # win back $110 + $100 profit
    roi = round((total_returned - total_wagered) / total_wagered * 100, 1) if total_wagered else None
    pl_units = ats_wins - (len(ats_graded) - ats_wins) * 1.1  # simplified P/L

    stats = {
        "total_predictions": len(records),
        "ats_graded": len(ats_graded),
        "ats_wins": ats_wins,
        "ats_losses": len(ats_graded) - ats_wins,
        "ats_hit_rate": round(ats_wins / len(ats_graded) * 100, 1) if ats_graded else None,
        "golden_graded": len(golden_graded),
        "golden_wins": golden_wins,
        "golden_hit_rate": round(golden_wins / len(golden_graded) * 100, 1) if golden_graded else None,
        "roi": roi,
        "pl_units": round(pl_units, 1) if ats_graded else None,
        "pending": len(records) - len(ats_graded),
    }

    return jsonify({"records": records, "stats": stats})


# ---------------------------------------------------------------------------
# API: Player data (RapidAPI, lightweight)
# ---------------------------------------------------------------------------
# (kept below; pinned flag annotation is inserted at the other return site.)

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


# ---------------------------------------------------------------------------
# API: Head-to-Head Season Records
# ---------------------------------------------------------------------------

_ODDS_DB_PATH = Path(__file__).parent.parent / "Data" / "OddsData.sqlite"


def _freshest_table(con, season: str) -> str:
    """Return the table name with the most-recent Date for a given season."""
    candidates = [
        row[0] for row in
        con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        if season in row[0]
    ]
    if not candidates:
        return season
    if len(candidates) == 1:
        return candidates[0]
    best, best_date = candidates[0], ""
    for c in candidates:
        try:
            row = con.execute(f'SELECT MAX(Date) FROM "{c}"').fetchone()
            d = row[0] or ""
            if d > best_date:
                best, best_date = c, d
        except Exception:
            pass
    return best


@app.route("/api/h2h")
def api_h2h():
    """Return head-to-head records for two teams in the current season.

    Reads from pre-computed web/data/season_h2h.json (available on Vercel).
    Falls back to querying OddsData.sqlite directly if the JSON is absent.

    Query params:
      home: home team name (English)
      away: away team name (English)
      season: season string, default "2025-26"
    """
    home = request.args.get("home", "").strip()
    away = request.args.get("away", "").strip()
    season = request.args.get("season", "2025-26")

    if not home or not away:
        return jsonify({"error": "Missing home or away parameter"}), 400

    # ── Primary path: pre-computed JSON (works on Vercel) ──
    h2h_json = DATA_DIR / "season_h2h.json"
    if h2h_json.exists():
        try:
            h2h_data = json.loads(h2h_json.read_text(encoding="utf-8"))
            canonical = ":".join(sorted([home, away]))
            games = h2h_data.get("pairs", {}).get(canonical, [])
            return jsonify({
                "games": games,
                "home_team": home,
                "away_team": away,
                "season": h2h_data.get("season", season),
                "total": len(games),
            })
        except Exception:
            pass  # fall through to SQLite path

    # ── Fallback path: query SQLite directly (local dev only) ──
    if not _ODDS_DB_PATH.exists():
        return jsonify({"error": "Database not available", "games": [], "total": 0})

    try:
        con = sqlite3.connect(str(_ODDS_DB_PATH))
        con.row_factory = sqlite3.Row
        table = _freshest_table(con, season)
        rows = con.execute(
            f"""SELECT Date, Home, Away, Spread, Win_Margin
                FROM "{table}"
                WHERE (Home=? AND Away=?) OR (Home=? AND Away=?)
                ORDER BY Date""",
            (home, away, away, home),
        ).fetchall()
        con.close()
    except Exception as exc:
        return jsonify({"error": str(exc), "games": [], "total": 0})

    games = []
    for r in rows:
        spread = r["Spread"]
        wm = r["Win_Margin"]
        if spread is not None and wm is not None:
            diff = wm - spread
            if abs(diff) < 0.001:
                ats_result = "push"
            elif diff > 0:
                ats_result = "home"
            else:
                ats_result = "away"
        else:
            ats_result = None
        games.append({"date": r["Date"], "home": r["Home"], "away": r["Away"],
                      "spread": spread, "win_margin": wm, "ats_result": ats_result})

    return jsonify({"games": games, "home_team": home, "away_team": away,
                    "season": season, "total": len(games)})


# ---------------------------------------------------------------------------
# API: ATS Model Daily Pick Log
# ---------------------------------------------------------------------------

@app.route("/api/ats-daily-log")
def api_ats_daily_log():
    """Return all ATS model picks with outcomes for recent days.

    Query params:
      days: look-back window (default 30)
    """
    days = int(request.args.get("days", 30))
    today_ = today_taipei()

    day_records = []
    for offset in range(days):
        d = today_ - timedelta(days=offset)
        iso = d.isoformat()
        data = load_date_data(iso)
        if not data:
            continue

        day_picks = []
        for key, g in (data.get("games") or {}).items():
            ats_winner = g.get("ats_winner")           # 'home', 'away', 'push', or None
            spread = g.get("spread")
            home_zh = g.get("home_team_zh") or g.get("home_team", "?")
            away_zh = g.get("away_team_zh") or g.get("away_team", "?")

            def _ats_correct(pick_side, winner):
                if winner and winner != "push":
                    return 1 if pick_side == winner else 0
                return None

            def _pick_str(pick_side, sp):
                picked_zh = home_zh if pick_side == "home" else away_zh
                if sp is not None:
                    abs_sp = abs(float(sp))
                    is_fav = (pick_side == "home" and sp > 0) or (pick_side == "away" and sp < 0)
                    if abs_sp == 0:
                        return f"{picked_zh}（PK）"
                    return f"{picked_zh} {'讓' if is_fav else '受讓'} {abs_sp:.1f}"
                return picked_zh

            # ── ML model pick ────────────────────────────────────────
            ats_pick_side = g.get("ats_model_pick")
            if ats_pick_side:
                day_picks.append({
                    "game_date": iso,
                    "game_key": key,
                    "home_team": g.get("home_team"),
                    "away_team": g.get("away_team"),
                    "home_team_zh": home_zh,
                    "away_team_zh": away_zh,
                    "spread": spread,
                    "ats_pick": ats_pick_side,
                    "pick_str": _pick_str(ats_pick_side, spread),
                    "ats_edge": g.get("ats_value_edge"),
                    "ats_winner": ats_winner,
                    "ats_correct": _ats_correct(ats_pick_side, ats_winner),
                    "is_golden": bool(g.get("is_golden")),
                    "is_value": bool(g.get("is_value")),
                    "ats_is_value": bool(g.get("ats_is_value")),
                    "pick_type": "ml",
                })

            # ── Playoff strategy pick (best signal per game) ──────────
            po_side = g.get("playoff_ats_best_side")
            po_tier = g.get("playoff_ats_best_tier")
            po_signal = g.get("playoff_ats_best_signal")
            if po_side and po_tier and g.get("is_playoff"):
                day_picks.append({
                    "game_date": iso,
                    "game_key": key,
                    "home_team": g.get("home_team"),
                    "away_team": g.get("away_team"),
                    "home_team_zh": home_zh,
                    "away_team_zh": away_zh,
                    "spread": spread,
                    "ats_pick": po_side,
                    "pick_str": _pick_str(po_side, spread),
                    "ats_edge": None,
                    "ats_winner": ats_winner,
                    "ats_correct": _ats_correct(po_side, ats_winner),
                    "is_golden": False,
                    "is_value": False,
                    "ats_is_value": False,
                    "pick_type": "playoff",
                    "playoff_tier": po_tier,
                    "playoff_signal": po_signal,
                })

        if day_picks:
            day_records.append({"date": iso, "picks": day_picks})

    # Aggregate stats — ML picks only (playoff picks tracked separately)
    all_picks = [p for day in day_records for p in day["picks"]]
    ml_picks = [p for p in all_picks if p.get("pick_type") != "playoff"]
    po_picks = [p for p in all_picks if p.get("pick_type") == "playoff"]

    graded = [p for p in ml_picks if p["ats_correct"] is not None]
    wins = sum(1 for p in graded if p["ats_correct"] == 1)

    # Recent 5-pick streak from graded ML picks
    last5 = graded[-5:] if len(graded) >= 5 else graded
    streak_wins = sum(1 for p in last5 if p["ats_correct"] == 1)
    streak_str = f"{streak_wins}-{len(last5) - streak_wins}" if last5 else None

    # Playoff strategy stats
    po_graded = [p for p in po_picks if p["ats_correct"] is not None]
    po_wins = sum(1 for p in po_graded if p["ats_correct"] == 1)

    stats = {
        "total": len(ml_picks),
        "graded": len(graded),
        "wins": wins,
        "losses": len(graded) - wins,
        "hit_rate": round(wins / len(graded) * 100, 1) if graded else None,
        "pending": len(ml_picks) - len(graded),
        "last5": streak_str,
        "playoff_total": len(po_picks),
        "playoff_graded": len(po_graded),
        "playoff_wins": po_wins,
        "playoff_losses": len(po_graded) - po_wins,
        "playoff_hit_rate": round(po_wins / len(po_graded) * 100, 1) if po_graded else None,
    }

    return jsonify({"days": day_records, "stats": stats})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
