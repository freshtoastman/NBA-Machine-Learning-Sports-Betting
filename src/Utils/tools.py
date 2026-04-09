import re
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from .Dictionaries import team_index_current

TAIPEI_TZ = ZoneInfo("Asia/Taipei")


def today_taipei():
    """Return today's date in Asia/Taipei timezone."""
    return datetime.now(TAIPEI_TZ).date()

games_header = {
    'user-agent': 'Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/57.0.2987.133 Safari/537.36',
    'Dnt': '1',
    'Accept-Encoding': 'gzip, deflate, sdch',
    'Accept-Language': 'en',
    'origin': 'http://stats.nba.com',
    'Referer': 'https://github.com'
}

data_headers = {
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Origin": "https://www.nba.com",
    "Priority": "u=3, i",
    "Referer": "https://www.nba.com/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0.1 Safari/605.1.15"
}


def get_json_data(url):
    raw_data = requests.get(url, headers=data_headers)
    try:
        json = raw_data.json()
    except Exception as e:
        print(e)
        return {}
    return json.get('resultSets')


def get_todays_games_json(url):
    raw_data = requests.get(url, headers=games_header)
    json = raw_data.json()
    return json.get('gs').get('g')


def to_data_frame(data):
    try:
        data_list = data[0]
    except Exception as e:
        print(e)
        return pd.DataFrame(data={})
    return pd.DataFrame(data=data_list.get('rowSet'), columns=data_list.get('headers'))


def create_todays_games(input_list):
    games = []
    for game in input_list:
        home = game.get('h')
        away = game.get('v')
        home_team = home.get('tc') + ' ' + home.get('tn')
        away_team = away.get('tc') + ' ' + away.get('tn')
        games.append([home_team, away_team])
    return games


def create_todays_games_from_odds(input_dict):
    games = []
    for game in input_dict.keys():
        home_team, away_team = game.split(":")
        if home_team not in team_index_current or away_team not in team_index_current:
            continue
        games.append([home_team, away_team])
    return games


def get_date(date_string):
    year1, month, day = re.search(r'(\d+)-\d+-(\d\d)(\d\d)', date_string).groups()
    year = year1 if int(month) > 8 else int(year1) + 1
    return datetime.strptime(f"{year}-{month}-{day}", '%Y-%m-%d')


def current_nba_season_start_year(today=None):
    """Return the calendar year that the current NBA season started in.

    NBA seasons run October–June. October–December belongs to season starting that year;
    January–September belongs to season starting the previous year.
    """
    today = today or datetime.today()
    return today.year if today.month >= 10 else today.year - 1


def current_nba_season(today=None):
    """Return the current NBA season label, e.g. '2025-26'."""
    start = current_nba_season_start_year(today)
    return f"{start}-{(start + 1) % 100:02d}"
