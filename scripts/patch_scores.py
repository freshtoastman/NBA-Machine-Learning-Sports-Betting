"""Patch final scores for recent dates where Points=0.

1) SBR scoreboard (src/Utils/sbr_scoreboard) when status is Final (legacy).
2) NBA CDN boxscore (stats.nba.com GameHeader game ids + cdn.nba.com boxscore JSON)
   when games are Final but SBR still shows pregame / 0-0.

Usage: PYTHONPATH=. python scripts/patch_scores.py
"""
import sqlite3
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import requests

from src.Utils import nba_cdn
from src.Utils.sbr_scoreboard import Scoreboard  # browser UA + timeout; sbrscrape gets 403
from src.Utils.tools import current_nba_season, data_headers

ODDS_DB = "Data/OddsData.sqlite"
DAYS_BACK = 3
# Resolved in main(): the table Get_Odds_Data writes for the current season.
SEASON_TABLE = "2025-26"
# cdn.nba.com answers 403 to a bare "Mozilla/5.0" UA (seen 2026-10-20 06:00) but 200
# with the stats.nba.com header set plus Referer/Origin.
NBA_CDN_HEADERS = {**data_headers, "Referer": "https://www.nba.com/", "Origin": "https://www.nba.com"}


def _season_table(con, today: date) -> str:
    """Odds table for the season `today` falls in.

    The season label rolls over on Oct 1 but its table only appears on opening
    night, so fall back to the previous season's table until then.
    """
    season = current_nba_season(today)
    exists = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (season,)
    ).fetchone()
    if exists:
        return season
    start = int(season[:4])
    return f"{start - 1}-{start % 100:02d}"


def _normalize_team_name(name: str) -> str:
    return name.replace("Los Angeles Clippers", "LA Clippers")


def _patch_from_sbr(con, d_str: str) -> int:
    try:
        sb = Scoreboard(sport="NBA", date=d_str)
        games = sb.games if hasattr(sb, "games") else []
    except Exception as exc:
        print(f"{d_str}: sbrscrape error: {exc}")
        return 0

    updated = 0
    for g in games:
        status = (g.get("status") or "").lower()
        if status not in ("final", "complete"):
            continue
        home = _normalize_team_name(g.get("home_team", ""))
        away = _normalize_team_name(g.get("away_team", ""))
        hs = g.get("home_score")
        aws = g.get("away_score")
        if hs is None or aws is None:
            continue
        points = hs + aws
        win_margin = hs - aws
        cur = con.execute(
            f'SELECT COUNT(*) FROM "{SEASON_TABLE}" WHERE Date = ? AND Home = ? AND Away = ? '
            "AND (Points IS NULL OR Points = 0)",
            (d_str, home, away),
        ).fetchone()
        if not cur or cur[0] == 0:
            continue
        c = con.execute(
            f'UPDATE "{SEASON_TABLE}" SET Points = ?, Win_Margin = ? '
            "WHERE Date = ? AND Home = ? AND Away = ? AND (Points IS NULL OR Points = 0)",
            (points, win_margin, d_str, home, away),
        )
        if c.rowcount:
            updated += c.rowcount
    return updated


def _game_ids_for_date(d: date) -> list[str]:
    """Game ids for a US date: cdn.nba.com season schedule first (finished games
    only), stats.nba.com scoreboardv2 as fallback (500 all opening week 2026-27)."""
    try:
        cdn_games = nba_cdn.games_for_date(d)
    except Exception as exc:  # noqa: BLE001 - fall through to stats.nba.com
        print(f"{d.isoformat()}: cdn schedule error: {exc}")
        cdn_games = []
    if cdn_games:
        return [g["game_id"] for g in cdn_games if g["status"] == 3]
    mmddyyyy = d.strftime("%m/%d/%Y")
    r = requests.get(
        "https://stats.nba.com/stats/scoreboardv2",
        headers=data_headers,
        params={"DayOffset": "0", "LeagueID": "00", "gameDate": mmddyyyy},
        timeout=60,
    )
    r.raise_for_status()
    payload = r.json()
    for rs in payload.get("resultSets", []):
        if rs.get("name") != "GameHeader":
            continue
        headers = rs.get("headers") or []
        rows = rs.get("rowSet") or []
        try:
            idx = headers.index("GAME_ID")
        except ValueError:
            return []
        return [str(row[idx]) for row in rows]
    return []


def _team_label(team: dict) -> str:
    city = team.get("teamCity") or ""
    name = team.get("teamName") or ""
    return _normalize_team_name(f"{city} {name}".strip())


def _patch_from_nba_cdn(con, d_str: str) -> int:
    d = date.fromisoformat(d_str)
    game_ids = _game_ids_for_date(d)
    if not game_ids:
        return 0

    updated = 0
    for gid in game_ids:
        try:
            url = f"https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gid}.json"
            r = requests.get(url, headers=NBA_CDN_HEADERS, timeout=30)
            r.raise_for_status()
            game = r.json().get("game") or {}
        except Exception as exc:
            print(f"{d_str}: boxscore {gid}: {exc}")
            time.sleep(0.2)
            continue

        if game.get("gameStatus") != 3:
            time.sleep(0.15)
            continue

        home_t = game.get("homeTeam") or {}
        away_t = game.get("awayTeam") or {}
        hs = home_t.get("score")
        aws = away_t.get("score")
        if hs is None or aws is None:
            time.sleep(0.15)
            continue

        home = _team_label(home_t)
        away = _team_label(away_t)
        points = int(hs) + int(aws)
        win_margin = int(hs) - int(aws)

        cur = con.execute(
            f'SELECT COUNT(*) FROM "{SEASON_TABLE}" WHERE Date = ? AND Home = ? AND Away = ? '
            "AND (Points IS NULL OR Points = 0)",
            (d_str, home, away),
        ).fetchone()
        if not cur or cur[0] == 0:
            time.sleep(0.15)
            continue

        con.execute(
            f'UPDATE "{SEASON_TABLE}" SET Points = ?, Win_Margin = ? '
            "WHERE Date = ? AND Home = ? AND Away = ? AND (Points IS NULL OR Points = 0)",
            (points, win_margin, d_str, home, away),
        )
        updated += 1
        time.sleep(0.15)

    return updated


def _patch_from_nba_live_scoreboard(con, d_str: str) -> int:
    """Use NBA CDN todaysScoreboard_00.json (real-time, no auth) to patch finals.

    Works when boxscore API returns 403 (playoff games). Only patches today's games.
    """
    today_str = date.today().isoformat()
    yesterday_str = (date.today() - timedelta(days=1)).isoformat()
    if d_str not in (today_str, yesterday_str):
        return 0
    try:
        r = requests.get(
            "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json",
            headers=NBA_CDN_HEADERS, timeout=10,
        )
        r.raise_for_status()
        games = r.json().get("scoreboard", {}).get("games", [])
    except Exception as exc:
        print(f"{d_str}: nba live scoreboard error: {exc}")
        return 0

    updated = 0
    for g in games:
        status_text = g.get("gameStatusText", "")
        if "Final" not in status_text:
            continue
        home_t = g.get("homeTeam", {})
        away_t = g.get("awayTeam", {})
        hs = home_t.get("score")
        aws = away_t.get("score")
        if hs is None or aws is None:
            continue
        home = _normalize_team_name(f"{home_t.get('teamCity', '')} {home_t.get('teamName', '')}".strip())
        away = _normalize_team_name(f"{away_t.get('teamCity', '')} {away_t.get('teamName', '')}".strip())
        points = int(hs) + int(aws)
        win_margin = int(hs) - int(aws)
        cur = con.execute(
            f'SELECT COUNT(*) FROM "{SEASON_TABLE}" WHERE Date = ? AND Home = ? AND Away = ? '
            "AND (Points IS NULL OR Points = 0)",
            (d_str, home, away),
        ).fetchone()
        if not cur or cur[0] == 0:
            continue
        c = con.execute(
            f'UPDATE "{SEASON_TABLE}" SET Points = ?, Win_Margin = ? '
            "WHERE Date = ? AND Home = ? AND Away = ? AND (Points IS NULL OR Points = 0)",
            (points, win_margin, d_str, home, away),
        )
        if c.rowcount:
            print(f"{d_str}: nba live patched {home} vs {away}: {hs}-{aws}")
            updated += c.rowcount
    return updated


def _patch_from_espn(con, d_str: str) -> int:
    """Fallback: fetch scores from ESPN public API when SBR/NBA-CDN fail."""
    import datetime
    d = date.fromisoformat(d_str)
    date_param = d.strftime("%Y%m%d")
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_param}&limit=20"
    ua = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    try:
        r = requests.get(url, headers=ua, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        print(f"{d_str}: espn error: {exc}")
        return 0

    updated = 0
    for event in data.get("events", []):
        comps = event.get("competitions", [{}])[0]
        completed = comps.get("status", {}).get("type", {}).get("completed", False)
        if not completed:
            continue
        teams = comps.get("competitors", [])
        if len(teams) < 2:
            continue
        # ESPN: teams[0] is home, teams[1] is away
        home_team_data = teams[0]
        away_team_data = teams[1]
        home_name = _normalize_team_name(home_team_data["team"]["displayName"])
        away_name = _normalize_team_name(away_team_data["team"]["displayName"])
        try:
            hs = int(home_team_data["score"])
            aws = int(away_team_data["score"])
        except (KeyError, ValueError, TypeError):
            continue
        points = hs + aws
        win_margin = hs - aws

        cur = con.execute(
            f'SELECT COUNT(*) FROM "{SEASON_TABLE}" WHERE Date = ? AND Home = ? AND Away = ? '
            "AND (Points IS NULL OR Points = 0)",
            (d_str, home_name, away_name),
        ).fetchone()
        if not cur or cur[0] == 0:
            continue
        c = con.execute(
            f'UPDATE "{SEASON_TABLE}" SET Points = ?, Win_Margin = ? '
            "WHERE Date = ? AND Home = ? AND Away = ? AND (Points IS NULL OR Points = 0)",
            (points, win_margin, d_str, home_name, away_name),
        )
        if c.rowcount:
            print(f"{d_str}: espn patched {home_name} vs {away_name}: {hs}-{aws}")
            updated += c.rowcount
    return updated


def _safe_patch(name: str, fn, con, d_str: str) -> int:
    """Run one score source; a failing source is logged and skipped.

    2026-10-20 (opening day, 06:00 Taipei = 18:00 ET the day before): the
    stats.nba.com scoreboard answered 500 for a date whose games had not tipped
    off yet, `_game_ids_for_date` raised, and the whole daily pipeline stopped
    before Export_JSON. No score source may abort this step.
    """
    try:
        updated = fn(con, d_str)
    except Exception as exc:  # noqa: BLE001 - any source error is non-fatal
        print(f"{d_str}: {name} error: {exc}")
        updated = 0
    con.commit()
    if updated:
        print(f"{d_str}: {name} patched {updated} games")
    return updated


def _unscored_count(con, d_str: str) -> int:
    row = con.execute(
        f'SELECT COUNT(*) FROM "{SEASON_TABLE}" WHERE Date = ? AND (Points IS NULL OR Points = 0)',
        (d_str,),
    ).fetchone()
    return int(row[0]) if row else 0


def _us_today() -> date:
    """Game dates are US (Eastern) dates; the pipeline runs on Taipei time."""
    return datetime.now(ZoneInfo("America/New_York")).date()


def main(con=None, today=None, us_today=None):
    global SEASON_TABLE
    close = con is None
    if con is None:
        con = sqlite3.connect(ODDS_DB)
    today = today or date.today()
    us_today = us_today or _us_today()
    SEASON_TABLE = _season_table(con, today)
    total_updated = 0

    sources = (
        # NBA live scoreboard (real-time, works for playoffs unlike boxscore 403)
        ("nba live scoreboard", _patch_from_nba_live_scoreboard),
        ("sbr", _patch_from_sbr),
        ("nba cdn", _patch_from_nba_cdn),
        ("espn", _patch_from_espn),
    )

    for offset in range(DAYS_BACK + 1):
        d = today - timedelta(days=offset)
        d_str = d.isoformat()
        if d > us_today:
            # Taipei is a day ahead of the US: this slate has not tipped off yet.
            continue
        if _unscored_count(con, d_str) == 0:
            continue
        for name, fn in sources:
            if name != "nba live scoreboard" and _unscored_count(con, d_str) == 0:
                break
            total_updated += _safe_patch(name, fn, con, d_str)

    if close:
        con.close()
    print(f"Done. Total patched (sbr + cdn + espn): {total_updated}")
    return total_updated


if __name__ == "__main__":
    main()
