"""Drop-in replacement for ``sbrscrape.Scoreboard`` (same ``.games`` shape).

Why this exists (2026-10-19, the day before opening night): sportsbookreview.com
started answering HTTP 403 to the default ``python-requests/x.y`` user agent. The
``sbrscrape`` package sends no headers and no timeout, swallows the 403 (its
``except IndexError: return []`` path never sets ``.games``), and every caller then
saw either ``AttributeError: 'Scoreboard' object has no attribute 'games'`` or an
empty list — so Get_Odds_Data, update_odds, patch_scores and the export fallback
would all have reported "no games" on 2026-10-20 while the site itself listed the
three openers. A browser user agent gets the page and the ``_next/data`` JSON.

Differences from sbrscrape: browser headers, a timeout on every request, the
``buildId`` cached per process (one page fetch instead of one per date), a stale
``buildId`` refreshed once on a non-200, and ``.games`` always set (``.error`` holds
the last failure text so callers can log it).
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Any

import requests

SBR_BASE = "https://www.sportsbookreview.com"
SPORT_PATHS = {
    "NBA": "nba-basketball",
    "NFL": "nfl-football",
    "NHL": "nhl-hockey",
    "MLB": "mlb-baseball",
    "NCAAB": "ncaa-basketball",
}
REQUEST_TIMEOUT_SECONDS = 30
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": SBR_BASE + "/betting-odds/nba-basketball/",
}
_NEXT_DATA_RE = re.compile(r'__NEXT_DATA__" type="application/json">(.*?)</script>', re.S)

_build_id_cache: dict[str, str] = {}


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def _date_str(value: Any) -> str:
    if isinstance(value, (datetime, date)):
        return value.strftime("%Y-%m-%d")
    if not value:
        return datetime.today().strftime("%Y-%m-%d")
    return str(value)


def _get(session: requests.Session, url: str) -> requests.Response:
    r = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    r.raise_for_status()
    return r


def _fetch_build_id(session: requests.Session, league: str, date_str: str, refresh: bool = False) -> str:
    if not refresh and league in _build_id_cache:
        return _build_id_cache[league]
    page = _get(session, f"{SBR_BASE}/betting-odds/{league}/?date={date_str}")
    found = _NEXT_DATA_RE.findall(page.text)
    if not found:
        raise RuntimeError("__NEXT_DATA__ block not found on SBR page (status %s)" % page.status_code)
    build_id = json.loads(found[0])["buildId"]
    _build_id_cache[league] = build_id
    return build_id


def _rows(session: requests.Session, league: str, build_id: str, path: str, query: str) -> list[dict]:
    url = f"{SBR_BASE}/_next/data/{build_id}/betting-odds/{league}{path}.json?league={league}{query}"
    payload = _get(session, url).json()
    tables = payload["pageProps"]["oddsTables"]
    if not tables:
        return []
    return tables[0]["oddsTableModel"]["gameRows"] or []


def _lines(rows: dict, game_id: Any, line_key: str) -> list[dict]:
    row = rows.get(game_id) or {}
    return [v[line_key] | {"sportsbook": v["sportsbook"]} for v in (row.get("oddsViews") or []) if v and v.get(line_key)]


def fetch_games(date_value: Any = "", sport: str = "NBA", current_line: bool = True,
                session: requests.Session | None = None) -> list[dict]:
    """Return sbrscrape-shaped game dicts for one date. Raises on network/parse errors."""
    league = SPORT_PATHS[sport]
    date_str = _date_str(date_value)
    line_key = "currentLine" if current_line else "openingLine"
    session = session or make_session()

    build_id = _fetch_build_id(session, league, date_str)
    try:
        spread_rows = _rows(session, league, build_id, "", f"&date={date_str}")
    except requests.HTTPError:
        # A deploy changed the buildId since it was cached: refresh once and retry.
        build_id = _fetch_build_id(session, league, date_str, refresh=True)
        spread_rows = _rows(session, league, build_id, "", f"&date={date_str}")
    if not spread_rows:
        return []
    ml_rows = _rows(session, league, build_id, "/money-line/full-game",
                    f"&oddsType=money-line&oddsScope=full-game&date={date_str}")
    total_rows = _rows(session, league, build_id, "/totals/full-game",
                       f"&oddsType=totals&oddsScope=full-game&date={date_str}")

    spreads = {g["gameView"]["gameId"]: g for g in spread_rows}
    moneylines = {g["gameView"]["gameId"]: g for g in ml_rows}
    totals = {g["gameView"]["gameId"]: g for g in total_rows}

    games = []
    for game_id, row in spreads.items():
        gv = row["gameView"]
        game = {
            "date": gv.get("startDate"),
            "status": gv.get("gameStatusText"),
            "home_team": gv["homeTeam"]["fullName"],
            "home_team_loc": gv["homeTeam"].get("displayName"),
            "home_team_abbr": gv["homeTeam"].get("shortName"),
            "home_team_rank": gv["homeTeam"].get("rank"),
            "away_team": gv["awayTeam"]["fullName"],
            "away_team_loc": gv["awayTeam"].get("displayName"),
            "away_team_abbr": gv["awayTeam"].get("shortName"),
            "away_team_rank": gv["awayTeam"].get("rank"),
            "home_score": gv.get("homeTeamScore"),
            "away_score": gv.get("awayTeamScore"),
            "home_spread": {}, "home_spread_odds": {}, "away_spread": {}, "away_spread_odds": {},
            "under_odds": {}, "over_odds": {}, "total": {}, "home_ml": {}, "away_ml": {},
        }
        for line in _lines(spreads, game_id, line_key):
            book = line["sportsbook"]
            game["home_spread"][book] = line.get("homeSpread")
            game["home_spread_odds"][book] = line.get("homeOdds")
            game["away_spread"][book] = line.get("awaySpread")
            game["away_spread_odds"][book] = line.get("awayOdds")
        for line in _lines(totals, game_id, line_key):
            book = line["sportsbook"]
            game["under_odds"][book] = line.get("underOdds")
            game["over_odds"][book] = line.get("overOdds")
            game["total"][book] = line.get("total")
        for line in _lines(moneylines, game_id, line_key):
            book = line["sportsbook"]
            game["home_ml"][book] = line.get("homeOdds")
            game["away_ml"][book] = line.get("awayOdds")
        games.append(game)
    return games


class Scoreboard:
    """API-compatible with ``sbrscrape.Scoreboard``: ``Scoreboard(sport, date).games``.

    ``games`` is always a list (empty on any failure) and ``error`` carries the
    failure text, so ``hasattr(sb, "games")`` checks in old callers keep working.
    """

    def __init__(self, sport: str = "NBA", date: Any = "", current_line: bool = True,
                 session: requests.Session | None = None):
        self.games: list[dict] = []
        self.error: str | None = None
        try:
            self.games = fetch_games(date, sport=sport, current_line=current_line, session=session)
        except Exception as exc:  # noqa: BLE001 - callers only need "no games" + a log line
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"SBR scoreboard fetch failed for {_date_str(date)}: {self.error}")
