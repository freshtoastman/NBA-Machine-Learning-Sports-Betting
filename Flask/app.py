from datetime import date, datetime, timedelta
import os
import sys
from pathlib import Path
from flask import Flask, render_template, jsonify, request
from functools import lru_cache
import requests
import time

# Load .env from project root.
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from main import predict_today_xgb, predict_historical_xgb
from src.Utils.AIAnalysis import analyze_game, generate_daily_report
from src.Utils.SeasonStats import compute_season_stats
from src.Utils.Teams import team_logo_url, team_name_zh
from src.Utils.tools import current_nba_season, current_nba_season_start_year, today_taipei

RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY")
RAPIDAPI_HEADERS = {
    "x-rapidapi-key": RAPIDAPI_KEY,
    "x-rapidapi-host": "tank01-fantasy-stats.p.rapidapi.com",
}


@lru_cache()
def fetch_game_data(sportsbook, ttl_hash=None):
    del ttl_hash
    games = predict_today_xgb(sportsbook=sportsbook)
    return {f"{g['away_team']}:{g['home_team']}": g for g in games}


@lru_cache(maxsize=64)
def fetch_historical_data(date_iso):
    """Historical predictions for a past date. Returns a dict keyed by 'away:home'."""
    target = date.fromisoformat(date_iso)
    games = predict_historical_xgb(target)
    return {f"{g['away_team']}:{g['home_team']}": g for g in games}


def get_ttl_hash(seconds=600):
    """Return the same value withing `seconds` time period"""
    return round(time.time() / seconds)


def compute_summary(games_dict, is_historical):
    """Compute aggregate stats for the header bar."""
    games = list(games_dict.values())
    n = len(games)
    if n == 0:
        return {"games": 0}

    home_picks = sum(1 for g in games if g.get("winner") == "home")
    away_picks = n - home_picks
    over_picks = sum(1 for g in games if g.get("ou_pick") == "OVER")
    under_picks = n - over_picks
    avg_conf = sum(max(g.get("home_confidence", 0), g.get("away_confidence", 0)) for g in games) / n

    summary = {
        "games": n,
        "home_picks": home_picks,
        "away_picks": away_picks,
        "over_picks": over_picks,
        "under_picks": under_picks,
        "avg_confidence": round(avg_conf, 1),
    }

    value_picks = [g for g in games if g.get("is_value")]
    summary["value_picks"] = len(value_picks)

    golden_picks = [g for g in games if g.get("is_golden")]
    summary["golden_picks"] = len(golden_picks)

    # Collect GOLD/SILVER playoff ATS signals across all games.
    po_ats_alerts = []
    for g in games:
        if g.get("is_playoff") and g.get("playoff_ats_picks"):
            for pick in g["playoff_ats_picks"]:
                if pick.get("tier") in ("GOLD", "SILVER"):
                    po_ats_alerts.append({
                        "game_key": "%s:%s" % (g.get("away_team", ""), g.get("home_team", "")),
                        "home_team": g.get("home_team", ""),
                        "away_team": g.get("away_team", ""),
                        "home_team_zh": g.get("home_team_zh", g.get("home_team", "")),
                        "away_team_zh": g.get("away_team_zh", g.get("away_team", "")),
                        "signal": pick.get("signal"),
                        "side": pick.get("side"),
                        "ats_side": pick.get("ats_side"),
                        "tier": pick.get("tier"),
                        "backtest_wr": pick.get("backtest_wr"),
                        "backtest_roi": pick.get("backtest_roi"),
                        "reason_zh": pick.get("reason_zh"),
                        "ats_winner": g.get("ats_winner"),
                    })
    summary["playoff_ats_alerts"] = po_ats_alerts
    summary["playoff_ats_count"] = len(po_ats_alerts)

    if is_historical:
        ml_correct = sum(1 for g in games if g.get("ml_correct") is True)
        ml_graded = sum(1 for g in games if g.get("ml_correct") is not None)
        ou_correct = sum(1 for g in games if g.get("ou_correct") is True)
        ou_graded = sum(1 for g in games if g.get("ou_correct") is not None)
        summary["ml_correct"] = ml_correct
        summary["ml_graded"] = ml_graded
        summary["ml_hit_rate"] = round(ml_correct / ml_graded * 100, 1) if ml_graded else None
        summary["ou_correct"] = ou_correct
        summary["ou_graded"] = ou_graded
        summary["ou_hit_rate"] = round(ou_correct / ou_graded * 100, 1) if ou_graded else None

        # Value pick hit rate
        value_decided = [g for g in value_picks if g.get("ml_correct") is not None]
        value_wins = sum(
            1 for g in value_decided
            if g["value_side"] == g.get("actual_winner")
        )
        summary["value_decided"] = len(value_decided)
        summary["value_wins"] = value_wins
        summary["value_hit_rate"] = (
            round(value_wins / len(value_decided) * 100, 1) if value_decided else None
        )

    return summary


def build_date_chips(selected_date, days=7):
    """Build a list of {iso, label, is_today, is_selected} chips for header."""
    today = today_taipei()
    chips = []
    for offset in range(0, days + 1):
        d = today - timedelta(days=offset)
        chips.append({
            "iso": d.isoformat(),
            "label": "今天" if offset == 0 else f"{d.month}/{d.day}",
            "weekday": ["週一", "週二", "週三", "週四", "週五", "週六", "週日"][d.weekday()],
            "is_today": offset == 0,
            "is_selected": d == selected_date,
        })
    return chips


app = Flask(__name__)
app.jinja_env.add_extension('jinja2.ext.loopcontrols')
app.jinja_env.globals.update(
    team_logo_url=team_logo_url,
    team_name_zh=team_name_zh,
)


def _has_upcoming_games_today(today):
    """Check the static schedule CSV for any game starting AFTER `now` in TPE."""
    try:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        import pandas as pd
        df = pd.read_csv(
            os.path.join(os.path.dirname(__file__), "..", "Data", f"nba-{today.year if today.month >= 10 else today.year - 1}-UTC.csv"),
            parse_dates=["Date"],
            date_format="%d/%m/%Y %H:%M",
        )
        df["Date"] = df["Date"].dt.tz_localize("UTC").dt.tz_convert("Asia/Taipei")
        now_tpe = datetime.now(ZoneInfo("Asia/Taipei"))
        upcoming = df[df["Date"] > now_tpe]
        return len(upcoming) > 0
    except Exception:
        return True  # fail open: assume there are games


def _last_played_date_with_games(today):
    """Walk back day-by-day until we find a date that has historical predictions."""
    for offset in range(1, 15):
        d = today - timedelta(days=offset)
        try:
            games = fetch_historical_data(d.isoformat())
            if games:
                return d
        except Exception:
            continue
    return None


@app.route("/")
def index():
    today = today_taipei()
    selected_date_str = request.args.get("date")
    if selected_date_str:
        try:
            selected_date = date.fromisoformat(selected_date_str)
        except ValueError:
            selected_date = today
    else:
        selected_date = today

    # If the user is on "today" but there are no upcoming games (NBA off day),
    # auto-redirect view to the most recent date that DID have games.
    no_games_today = False
    if selected_date == today and not _has_upcoming_games_today(today):
        last = _last_played_date_with_games(today)
        if last is not None:
            no_games_today = True
            selected_date = last  # show the recent recap instead

    is_today = selected_date == today
    if is_today:
        ttl = get_ttl_hash()
        fanduel = fetch_game_data("fanduel", ttl_hash=ttl)
        draftkings = fetch_game_data("draftkings", ttl_hash=ttl)
        betmgm = fetch_game_data("betmgm", ttl_hash=ttl)
    else:
        # For past days, we read from local SQLite (single source, not per-book).
        historical = fetch_historical_data(selected_date.isoformat())
        fanduel = historical
        draftkings = {}
        betmgm = {}

    summary = compute_summary(fanduel, is_historical=not is_today)

    # Active playoff series for the date (powers the series tracker UI).
    try:
        from src.Utils.PlayoffContext import get_active_series_for_date, is_playoff_date
        active_series = get_active_series_for_date(selected_date)
        is_playoff_view = is_playoff_date(selected_date)
    except Exception:
        active_series = []
        is_playoff_view = False
    season_key = current_nba_season(selected_date)
    season_stats = compute_season_stats(season_key)

    return render_template(
        'index.html',
        today=today,
        selected_date=selected_date,
        is_today=is_today,
        no_games_today=no_games_today,
        date_chips=build_date_chips(selected_date),
        summary=summary,
        season_key=season_key,
        season_stats=season_stats,
        active_series=active_series,
        is_playoff_view=is_playoff_view,
        data={"fanduel": fanduel, "draftkings": draftkings, "betmgm": betmgm},
    )




def _get_games_for_date(game_date_str):
    """Retrieve prediction dict for a date (today or historical)."""
    today = today_taipei()
    target = date.fromisoformat(game_date_str)
    if target == today:
        return fetch_game_data("fanduel", ttl_hash=get_ttl_hash())
    return fetch_historical_data(game_date_str)


@app.route("/api/analysis")
def api_analysis():
    """AJAX: structured AI analysis for a single game."""
    game_date = request.args.get("date")
    home = request.args.get("home")
    away = request.args.get("away")
    if not game_date or not home or not away:
        return jsonify({"error": "Missing date/home/away params"}), 400

    games = _get_games_for_date(game_date)
    game = games.get(f"{away}:{home}")
    if not game:
        return jsonify({"error": f"Game not found"}), 404

    result = analyze_game(game, game_date)
    return jsonify(result)


def _annotate_picks_with_game_key(picks, games_dict):
    """Add `game_key` (English away:home) to each AI pick by matching team names.

    Tries multiple match strategies because the AI may use Chinese names, raw
    English, or partial matches.
    """
    # Build lookup tables: zh_full -> game_key, en_substring -> game_key.
    zh_to_key = {}
    en_to_key = {}
    for key, g in games_dict.items():
        away_en = g.get("away_team", "")
        home_en = g.get("home_team", "")
        away_zh = team_name_zh(away_en)
        home_zh = team_name_zh(home_en)
        # Multiple lookup keys per game.
        zh_to_key[f"{away_zh} @ {home_zh}"] = key
        zh_to_key[f"{away_zh}@{home_zh}"] = key
        en_to_key[f"{away_en} @ {home_en}"] = key
        # Also store the raw zh names as fallback.
        zh_to_key[away_zh + home_zh] = key

    def find_key(game_str):
        if not game_str:
            return None
        s = game_str.strip().replace(" ", "")
        # Direct match
        for variant, k in zh_to_key.items():
            if variant.replace(" ", "") == s:
                return k
        # Substring match: look for any team name in the game string
        for key, g in games_dict.items():
            away_zh = team_name_zh(g.get("away_team", ""))
            home_zh = team_name_zh(g.get("home_team", ""))
            if away_zh and home_zh and away_zh in game_str and home_zh in game_str:
                return key
        return None

    for p in picks or []:
        gk = find_key(p.get("game"))
        if gk:
            p["game_key"] = gk
    return picks


@app.route("/api/analyze-pinned", methods=["POST"])
def api_analyze_pinned():
    """AJAX: analyze a user-pinned subset of games for a date.

    Body: {date: 'YYYY-MM-DD', pinned: ['Boston Celtics:Miami Heat', ...]}
    Returns: {picks, parlay_suggestion, total_units, strategy}
    """
    payload = request.get_json(silent=True) or {}
    game_date = payload.get("date")
    pinned_keys = payload.get("pinned") or []
    if not game_date or not pinned_keys:
        return jsonify({"error": "Missing date or pinned list"}), 400

    all_games = _get_games_for_date(game_date)
    if not all_games:
        return jsonify({"error": "No games found for date"}), 404

    # Filter to pinned subset (game_keys are 'away:home' English names).
    subset = {k: g for k, g in all_games.items() if k in pinned_keys}
    if not subset:
        return jsonify({"error": "No pinned games match today's slate"}), 404

    report = generate_daily_report(subset, game_date)
    if not isinstance(report, dict):
        return jsonify({"error": "AI report failed"}), 500

    # Annotate AI picks with English game_key for the front-end (consistent with daily-report).
    _annotate_picks_with_game_key(report.get("best_bets"), subset)
    _annotate_picks_with_game_key(report.get("lean_picks"), subset)

    # Reshape to what the front-end expects: a flat `picks` list.
    picks = []
    for b in (report.get("best_bets") or []):
        picks.append({
            "game": b.get("game"),
            "game_key": b.get("game_key"),
            "pick": b.get("pick"),
            "units": b.get("units", 0),
            "reason": b.get("reason"),
            "injuries": [],
        })
    for l in (report.get("lean_picks") or []):
        picks.append({
            "game": l.get("game"),
            "game_key": l.get("game_key"),
            "pick": l.get("pick"),
            "units": l.get("units", 0),
            "reason": l.get("reason"),
            "injuries": [],
        })

    total_units = sum(int(p.get("units") or 0) for p in picks)
    return jsonify({
        "picks": picks,
        "total_units": total_units,
        "strategy": report.get("daily_summary", ""),
        "parlay_suggestion": None,
    })


@app.route("/api/daily-report")
def api_daily_report():
    """AJAX: generate AI daily report for all games on a date."""
    game_date = request.args.get("date")
    if not game_date:
        return jsonify({"error": "Missing date param"}), 400

    games = _get_games_for_date(game_date)
    if not games:
        return jsonify({"error": "No games found"}), 404

    result = generate_daily_report(games, game_date)
    # Annotate each pick with the english game_key so the client can auto-pin.
    if isinstance(result, dict):
        _annotate_picks_with_game_key(result.get("best_bets"), games)
        _annotate_picks_with_game_key(result.get("lean_picks"), games)
    return jsonify(result)


def get_player_data(team_abv):
    """Fetch player data for a given team abbreviation"""
    url = "https://tank01-fantasy-stats.p.rapidapi.com/getNBATeamRoster"
    querystring = {"teamAbv": team_abv}

    try:
        response = requests.get(url, headers=RAPIDAPI_HEADERS, params=querystring)
        data = response.json()
        
        if data.get('statusCode') == 200:
            formatted_players = []
            roster = data.get('body', {}).get('roster', [])
            
            for player in roster:
                # Format injury status
                injury_status = "Healthy"
                if player.get('injury'):
                    injury_info = player['injury']
                    if injury_info.get('designation'):
                        injury_status = injury_info['designation']
                        if injury_info.get('description'):
                            injury_status += f" - {injury_info['description']}"
                
                formatted_player = {
                    'name': player.get('longName'),
                    'shortName': player.get('shortName'),
                    'headshot': player.get('nbaComHeadshot'),
                    'injury': injury_status,
                    'position': player.get('pos'),
                    'height': player.get('height'),
                    'weight': player.get('weight'),
                    'college': player.get('college'),
                    'experience': player.get('exp'),
                    'jerseyNum': player.get('jerseyNum'),
                    'playerId': player.get('playerID'),
                    'birthDate': player.get('bDay')
                }
                formatted_players.append(formatted_player)
            
            return {
                'success': True,
                'players': formatted_players
            }
        
        return {
            'success': False,
            'error': 'Failed to fetch team data'
        }
        
    except Exception as e:
        print(f"Error in get_player_data: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }


@app.route("/team-data/<team_name>")
def team_data(team_name):
    # Convert full team name to abbreviation using the existing dictionary
    team_abv = team_abbreviations.get(team_name)
    
    if not team_abv:
        return jsonify({
            'success': False,
            'error': f'Team abbreviation not found for {team_name}'
        })
    
    # Fetch and return the player data
    result = get_player_data(team_abv)
    return jsonify(result)


    
@app.route("/player-stats/<player_id>")
def player_stats(player_id):
    # First get player info
    info_url = "https://tank01-fantasy-stats.p.rapidapi.com/getNBAPlayerInfo"
    info_querystring = {"playerID": player_id}

    # Then get game stats
    games_url = "https://tank01-fantasy-stats.p.rapidapi.com/getNBAGamesForPlayer"
    games_querystring = {
        "playerID": player_id,
        "season": str(current_nba_season_start_year()),
    }

    try:
        # Get both responses
        info_response = requests.get(info_url, headers=RAPIDAPI_HEADERS, params=info_querystring)
        games_response = requests.get(games_url, headers=RAPIDAPI_HEADERS, params=games_querystring)
        
        info_data = info_response.json()
        games_data = games_response.json()
        
        if info_data.get('statusCode') == 200 and games_data.get('statusCode') == 200:
            # Process games data
            games = list(games_data['body'].values())
            games.sort(key=lambda x: x['gameID'], reverse=True)
            recent_games = games[:10]
            
            # Get player info
            player_info = info_data['body']
            
            # Format injury info
            injury_status = "Healthy"
            if player_info.get('injury'):
                injury_info = player_info['injury']
                injury_status = injury_info

            # Combine and return all data
            return jsonify({
                'success': True,
                'games': recent_games,
                'player': {
                    'name': player_info.get('longName'),
                    'position': player_info.get('pos'),
                    'number': player_info.get('jerseyNum'),
                    'height': player_info.get('height'),
                    'weight': player_info.get('weight'),
                    'team': player_info.get('team'),
                    'college': player_info.get('college'),
                    'experience': player_info.get('exp'),
                    'headshot': player_info.get('nbaComHeadshot'),
                    'injury': injury_status
                }
            })
            
        return jsonify({
            'success': False,
            'error': 'Failed to fetch player data'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

        
team_abbreviations = {
    'Orlando Magic': 'ORL',
    'Minnesota Timberwolves': 'MIN',
    'Miami Heat': 'MIA',
    'Boston Celtics': 'BOS',
    'LA Clippers': 'LAC',
    'Denver Nuggets': 'DEN',
    'Detroit Pistons': 'DET',
    'Atlanta Hawks': 'ATL',
    'Cleveland Cavaliers': 'CLE',
    'Toronto Raptors': 'TOR',
    'Washington Wizards': 'WAS',
    'Phoenix Suns': 'PHO',
    'San Antonio Spurs': 'SA',
    'Chicago Bulls': 'CHI',
    'Charlotte Hornets': 'CHA',
    'Philadelphia 76ers': 'PHI',
    'New Orleans Pelicans': 'NO',
    'Sacramento Kings': 'SAC',
    'Dallas Mavericks': 'DAL',
    'Houston Rockets': 'HOU',
    'Brooklyn Nets': 'BKN',
    'New York Knicks': 'NY',
    'Utah Jazz': 'UTA',
    'Oklahoma City Thunder': 'OKC',
    'Portland Trail Blazers': 'POR',
    'Indiana Pacers': 'IND',
    'Milwaukee Bucks': 'MIL',
    'Golden State Warriors': 'GS',
    'Memphis Grizzlies': 'MEM',
    'Los Angeles Lakers': 'LAL'
}