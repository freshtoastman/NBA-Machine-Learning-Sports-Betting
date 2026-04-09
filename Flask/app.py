from datetime import date, datetime, timedelta
import os
import sys
from flask import Flask, render_template, jsonify, request
from functools import lru_cache
import requests
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from main import predict_today_xgb, predict_historical_xgb
from src.Utils.Teams import team_logo_url, team_name_zh
from src.Utils.tools import current_nba_season_start_year, today_taipei

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

    return render_template(
        'index.html',
        today=today,
        selected_date=selected_date,
        is_today=is_today,
        date_chips=build_date_chips(selected_date),
        data={"fanduel": fanduel, "draftkings": draftkings, "betmgm": betmgm},
    )




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