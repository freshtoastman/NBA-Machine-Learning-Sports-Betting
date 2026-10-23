"""Pick a game's line with fallbacks: preferred SBR book → other SBR books → ESPN.

Why this exists (2026-10-23, the first model slate of the season): at 10:05 ET on a
12-game day sportsbookreview.com listed FanDuel for none of the games and *any*
book for only two of them (every ``oddsViews`` entry was ``None``). Get_Odds_Data
only knew ``game["total"]["fanduel"]`` and printed "No fanduel odds data found" twelve
times, so the season table got no 2026-10-23 rows, Create_Games wrote no 2026-27
dataset rows, the export skipped the date and the frontend would have shown
nothing for the night the model was supposed to come online. ESPN's public
scoreboard carried DraftKings spread + total for all 12 and its odds endpoint the
moneylines, with team names spelled exactly as SBR spells them.

Every line dict returned here has the same keys, whatever the source::

    home_team, away_team, ou, spread, ml_home, ml_away, source, home_score, away_score

``spread`` keeps the season-table convention (``away_spread``: positive = home
favoured). ``source`` is ``sbr_<book>`` or ``espn_<provider>`` so odds_history and
the export can say which book a line came from; the frozen record is still
FanDuel-referenced whenever FanDuel is on the board.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any, Callable

import requests

from src.Utils.OddsTables import canonical_team

PREFERRED_BOOK = "fanduel"
# Order tried after the preferred book. Same-market US books first; the half-point
# differences between them are smaller than the gap between a line and no line.
SBR_BOOK_FALLBACKS = ("draftkings", "betmgm", "caesars", "bet365", "fanatics", "pointsbet", "unibet")

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_ODDS = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{eid}/competitions/{eid}/odds"
ESPN_TIMEOUT_SECONDS = 20
# Only reach for ESPN around "now": backfilled past dates want SBR closing lines, and
# a unit test with a 2025 date must never open a socket.
ESPN_LIVE_WINDOW_DAYS = 3

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(name: str) -> str:
    return _SLUG_RE.sub("", (name or "").lower()) or "unknown"


def _num(value: Any):
    if value is None or value == "":
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return int(f) if f.is_integer() and abs(f) >= 100 else f


def _int(value: Any):
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _book_line(game: dict, book: str) -> dict | None:
    ou = (game.get("total") or {}).get(book)
    spread = (game.get("away_spread") or {}).get(book)
    if ou is None and spread is None:
        return None
    return {
        "home_team": game["home_team"],
        "away_team": game["away_team"],
        "ou": ou,
        "spread": spread,
        "ml_home": (game.get("home_ml") or {}).get(book),
        "ml_away": (game.get("away_ml") or {}).get(book),
        "source": f"sbr_{book}",
        "home_score": game.get("home_score") or 0,
        "away_score": game.get("away_score") or 0,
    }


def sbr_line(game: dict, preferred: str = PREFERRED_BOOK) -> dict | None:
    """The preferred book's line for one sbrscrape-shaped game, else the first fallback book's."""
    for book in (preferred, *[b for b in SBR_BOOK_FALLBACKS if b != preferred]):
        line = _book_line(game, book)
        if line is not None:
            return line
    return None


def _espn_json(session: requests.Session, url: str, **params) -> dict:
    r = session.get(url, params=params or None, timeout=ESPN_TIMEOUT_SECONDS)
    r.raise_for_status()
    return r.json()


def _espn_odds_item(odds_list: list, prefer_provider_id: str = "100") -> dict | None:
    """DraftKings (provider 100) when ESPN lists several books, else the first entry."""
    if not odds_list:
        return None
    for item in odds_list:
        if str((item.get("provider") or {}).get("id")) == prefer_provider_id:
            return item
    return odds_list[0]


def fetch_espn_lines(target_date: date, session: requests.Session | None = None) -> list[dict]:
    """DraftKings spread/total/moneyline for every NBA game on target_date, from ESPN.

    The scoreboard carries spread + total (home-referenced ``spread``); moneylines come
    from the per-event odds endpoint and are optional (None when ESPN has none yet).
    Raises on a network/parse failure of the scoreboard itself.
    """
    session = session or requests.Session()
    board = _espn_json(session, ESPN_SCOREBOARD, dates=target_date.strftime("%Y%m%d"), limit=50)
    lines = []
    for event in board.get("events") or []:
        comp = (event.get("competitions") or [{}])[0]
        sides = {c.get("homeAway"): c for c in comp.get("competitors") or []}
        if "home" not in sides or "away" not in sides:
            continue
        home, away = sides["home"], sides["away"]
        item = _espn_odds_item(comp.get("odds") or [])
        home_spread = _num(item.get("spread")) if item else None
        ou = _num(item.get("overUnder")) if item else None
        ml_home = _int(((item or {}).get("homeTeamOdds") or {}).get("moneyLine"))
        ml_away = _int(((item or {}).get("awayTeamOdds") or {}).get("moneyLine"))
        provider = (item or {}).get("provider") or {}
        if item is None or ml_home is None or ml_away is None:
            try:
                detail = _espn_json(session, ESPN_ODDS.format(eid=event["id"]))
                d_item = _espn_odds_item(detail.get("items") or [])
            except Exception:
                d_item = None
            if d_item:
                provider = d_item.get("provider") or provider
                if home_spread is None:
                    home_spread = _num(d_item.get("spread"))
                if ou is None:
                    ou = _num(d_item.get("overUnder"))
                if ml_home is None:
                    ml_home = _int((d_item.get("homeTeamOdds") or {}).get("moneyLine"))
                if ml_away is None:
                    ml_away = _int((d_item.get("awayTeamOdds") or {}).get("moneyLine"))
        if home_spread is None and ou is None:
            continue  # no book has posted this game anywhere ESPN looks
        lines.append({
            "home_team": home["team"]["displayName"],
            "away_team": away["team"]["displayName"],
            "ou": ou,
            # Season tables store the AWAY spread (positive = home favoured); ESPN gives the home one.
            "spread": -home_spread if home_spread is not None else None,
            "ml_home": ml_home,
            "ml_away": ml_away,
            "source": f"espn_{_slug(provider.get('name', 'unknown'))}",
            "home_score": _int(home.get("score")) or 0,
            "away_score": _int(away.get("score")) or 0,
            "espn_id": event.get("id"),
        })
    return lines


def _in_live_window(target_date: date, today: date | None = None) -> bool:
    today = today or date.today()
    return today - timedelta(days=ESPN_LIVE_WINDOW_DAYS) <= target_date <= today + timedelta(days=1)


def lines_for_games(sbr_games: list[dict], target_date: date, preferred: str = PREFERRED_BOOK,
                    espn_fetch: Callable[[date], list[dict]] | None = fetch_espn_lines,
                    today: date | None = None, log=print) -> list[dict]:
    """One line dict per game: SBR preferred book, then another SBR book, then ESPN.

    With no SBR games at all (site down, 403, empty board) and a date inside the
    live window, the ESPN slate itself becomes the game list. ESPN is fetched at
    most once and only when something is missing; a failed ESPN fetch just leaves
    the SBR-less games out (the caller logs the count).
    """
    lines: list[dict] = []
    missing: list[dict] = []
    for game in sbr_games or []:
        line = sbr_line(game, preferred)
        if line is None:
            missing.append(game)
        else:
            lines.append(line)
    if not missing and sbr_games:
        return lines
    if espn_fetch is None or not _in_live_window(target_date, today):
        return lines

    try:
        espn = espn_fetch(target_date)
    except Exception as exc:  # network / schema change: fall through with what SBR gave
        log(f"  ESPN odds fallback failed for {target_date}: {exc}")
        return lines

    if not sbr_games:
        if espn:
            log(f"  SBR listed no games for {target_date}; using {len(espn)} ESPN line(s)")
        return [{k: v for k, v in e.items() if k != "espn_id"} for e in espn]

    by_key = {(canonical_team(e["home_team"]), canonical_team(e["away_team"])): e for e in espn}
    filled = 0
    for game in missing:
        e = by_key.get((canonical_team(game["home_team"]), canonical_team(game["away_team"])))
        if e is None:
            continue
        lines.append({
            **{k: v for k, v in e.items() if k != "espn_id"},
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "home_score": game.get("home_score") or e["home_score"],
            "away_score": game.get("away_score") or e["away_score"],
        })
        filled += 1
    if missing:
        log(f"  {len(missing)} game(s) without a {preferred} line on SBR for {target_date}; "
            f"{filled} filled from ESPN")
    return lines


def source_label(source: str | None) -> str | None:
    """Short human label for a source tag ('sbr_fanduel_open' → 'FanDuel')."""
    if not source:
        return None
    tag = source[:-5] if source.endswith("_open") else source
    names = {
        "sbr_fanduel": "FanDuel", "sbr_draftkings": "DraftKings", "sbr_betmgm": "BetMGM",
        "sbr_caesars": "Caesars", "sbr_bet365": "bet365", "sbr_fanatics": "Fanatics",
        "espn_draftkings": "DraftKings (ESPN)",
    }
    return names.get(tag, tag)
