"""cdn.nba.com static JSON: season schedule (game ids) and per-game boxscores.

Why this exists (2026-10-21): stats.nba.com `scoreboardv2` answered 500 for every
date of the 2026-27 opening week (and `boxscoresummaryv2` had been empty or 403 for
most of 2025-26: 1,259 games in referee_games with no officials), while
cdn.nba.com kept serving

  * https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json
    the whole current season (preseason + regular + play-in + playoffs) with
    gameId, teams, gameStatus (1 scheduled / 2 live / 3 final) and scores; ~7 MB,
    so it is fetched once per process and cached;
  * https://cdn.nba.com/static/json/liveData/boxscore/boxscore_<gameId>.json
    final score + the three officials for any game that has tipped off (403
    before tip-off), including last season's game ids.

Both need the full stats.nba.com header set plus Referer/Origin (a bare UA = 403).
"""
from __future__ import annotations

from datetime import date, datetime

import requests

from src.Utils.tools import data_headers

NBA_CDN_HEADERS = {**data_headers, "Referer": "https://www.nba.com/", "Origin": "https://www.nba.com"}
SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
BOXSCORE_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json"

# gameId prefix: 001 preseason, 002 regular season, 003 all-star, 004 playoffs,
# 005 play-in, 006 NBA Cup final (counted as regular season by the league).
SEASON_GAME_PREFIXES = ("002", "004", "005", "006")

_schedule_cache: dict[str, list[dict]] | None = None


def normalize_team_name(name: str) -> str:
    return name.replace("Los Angeles Clippers", "LA Clippers")


def team_label(team: dict) -> str:
    city = team.get("teamCity") or ""
    name = team.get("teamName") or ""
    return normalize_team_name(f"{city} {name}".strip())


def is_season_game(game_id: str) -> bool:
    return str(game_id)[:3] in SEASON_GAME_PREFIXES


def _parse_schedule(payload: dict) -> dict[str, list[dict]]:
    """{YYYY-MM-DD: [game, ...]} keyed on the US (Eastern) game date."""
    out: dict[str, list[dict]] = {}
    for gd in (payload.get("leagueSchedule") or {}).get("gameDates") or []:
        raw = gd.get("gameDate") or ""
        try:
            key = datetime.strptime(raw[:10], "%m/%d/%Y").date().isoformat()
        except ValueError:
            continue
        out.setdefault(key, []).extend(gd.get("games") or [])
    return out


def fetch_schedule(timeout: int = 30, force: bool = False) -> dict[str, list[dict]]:
    """Season schedule keyed by ISO date; cached for the life of the process."""
    global _schedule_cache
    if _schedule_cache is not None and not force:
        return _schedule_cache
    r = requests.get(SCHEDULE_URL, headers=NBA_CDN_HEADERS, timeout=timeout)
    r.raise_for_status()
    _schedule_cache = _parse_schedule(r.json())
    return _schedule_cache


def games_for_date(d: date, season_only: bool = True) -> list[dict]:
    """Schedule entries for a US date, each summarised as

    {game_id, status (1/2/3), home, away (full names), home_id, away_id,
     home_score, away_score}. Preseason/all-star ids are dropped unless
    `season_only=False`.
    """
    games = []
    for g in fetch_schedule().get(d.isoformat(), []):
        gid = str(g.get("gameId") or "")
        if season_only and not is_season_game(gid):
            continue
        home = g.get("homeTeam") or {}
        away = g.get("awayTeam") or {}
        games.append({
            "game_id": gid,
            "status": g.get("gameStatus"),
            "home": team_label(home),
            "away": team_label(away),
            "home_id": home.get("teamId"),
            "away_id": away.get("teamId"),
            "home_score": home.get("score"),
            "away_score": away.get("score"),
            # Scheduled tip-off, e.g. "2026-10-24T22:00:00Z"; lets the pre-game
            # freeze lock a game by the clock instead of waiting to see a score.
            "tipoff_utc": g.get("gameDateTimeUTC"),
        })
    return games


def fetch_boxscore(game_id: str, timeout: int = 30) -> dict | None:
    """Final-or-live boxscore summary, or None when the game has not tipped off
    (cdn answers 403 until then) or the request fails.

    {game_id, status, status_text, home, away, home_score, away_score,
     officials: [(person_id, name), ...]}
    """
    r = requests.get(BOXSCORE_URL.format(game_id=game_id), headers=NBA_CDN_HEADERS, timeout=timeout)
    if r.status_code == 403:
        return None
    r.raise_for_status()
    game = (r.json() or {}).get("game") or {}
    if not game:
        return None
    home = game.get("homeTeam") or {}
    away = game.get("awayTeam") or {}
    officials = [
        (o.get("personId"), o.get("name"))
        for o in game.get("officials") or []
        if o.get("name")
    ]
    return {
        "game_id": str(game.get("gameId") or game_id),
        "status": game.get("gameStatus"),
        "status_text": game.get("gameStatusText") or "",
        "home": team_label(home),
        "away": team_label(away),
        "home_score": home.get("score"),
        "away_score": away.get("score"),
        "officials": officials,
    }
