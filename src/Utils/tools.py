import re
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from . import net as _net  # noqa: F401  (side effect: IPv4-only outbound HTTP)
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


# stats.nba.com occasionally accepts the TCP connection and then never answers.
# Without a timeout that one request hangs the caller forever (the 2017-18 ADV
# backfill sat 20+ minutes on a single ESTABLISHED socket on 2026-10-17, and the
# 06:00 pipeline would do the same). A timeout turns it into an empty result,
# which the callers' retry loops already handle.
REQUEST_TIMEOUT_SECONDS = 30


def get_json_data(url):
    try:
        raw_data = requests.get(url, headers=data_headers, timeout=REQUEST_TIMEOUT_SECONDS)
        json = raw_data.json()
    except Exception as e:
        print(e)
        return {}
    return json.get('resultSets')


def get_todays_games_json(url):
    raw_data = requests.get(url, headers=games_header, timeout=REQUEST_TIMEOUT_SECONDS)
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


def pregame_refresh_allowed(table_date, now_utc=None):
    """True while a stats table named `table_date` can be re-fetched without
    picking up that day's own games.

    stats.nba.com fetches use DateTo=table_date (inclusive) and add a game once
    it is final. The earliest NBA tip-offs are 12:00 ET (Christmas, MLK Day), so
    re-fetching before 12:00 ET on `table_date` only adds the previous day's late
    games; after that the table may already count games played on `table_date`.
    """
    from datetime import date as _date, time as _time, timezone as _tz
    if isinstance(table_date, str):
        table_date = _date.fromisoformat(table_date)
    if now_utc is None:
        now_utc = datetime.now(_tz.utc)
    eastern = ZoneInfo("America/New_York")
    cutoff = datetime.combine(table_date, _time(12, 0), tzinfo=eastern)
    return now_utc.astimezone(eastern) < cutoff


def record_fetch(con, table_name):
    """Log when a stats snapshot was fetched (UTC).

    A table named D fetched before D's games holds pre-game numbers; fetched
    later it already includes them. The log lets audits tell the two apart.
    """
    from datetime import datetime as _dt, timezone as _tz
    con.execute(
        'CREATE TABLE IF NOT EXISTS "_fetch_log" (table_name TEXT PRIMARY KEY, fetched_at_utc TEXT)'
    )
    con.execute(
        'INSERT OR REPLACE INTO "_fetch_log" VALUES (?, ?)',
        (table_name, _dt.now(_tz.utc).isoformat(timespec="seconds")),
    )
