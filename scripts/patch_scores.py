"""Patch final scores for recent dates where Points=0.

1) sbrscrape Scoreboard when status is Final (legacy).
2) NBA CDN boxscore (stats.nba.com GameHeader game ids + cdn.nba.com boxscore JSON)
   when games are Final but SBR still shows pregame / 0-0.

Usage: PYTHONPATH=. python scripts/patch_scores.py
"""
import sqlite3
import time
from datetime import date, timedelta

import requests
from sbrscrape import Scoreboard

from src.Utils.tools import data_headers

ODDS_DB = "Data/OddsData.sqlite"
DAYS_BACK = 3
SEASON_TABLE = "2025-26"


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
        if g.get("status") != "Final":
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
    ua = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15"}
    for gid in game_ids:
        try:
            url = f"https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gid}.json"
            r = requests.get(url, headers=ua, timeout=30)
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


def main():
    con = sqlite3.connect(ODDS_DB)
    today = date.today()
    total_updated = 0

    for offset in range(DAYS_BACK + 1):
        d = today - timedelta(days=offset)
        d_str = d.isoformat()

        rows = con.execute(
            f'SELECT COUNT(*) FROM "{SEASON_TABLE}" WHERE Date = ? AND (Points IS NULL OR Points = 0)',
            (d_str,),
        ).fetchone()
        if not rows or rows[0] == 0:
            continue

        u1 = _patch_from_sbr(con, d_str)
        con.commit()
        if u1:
            print(f"{d_str}: sbr patched {u1} games")

        rows_left = con.execute(
            f'SELECT COUNT(*) FROM "{SEASON_TABLE}" WHERE Date = ? AND (Points IS NULL OR Points = 0)',
            (d_str,),
        ).fetchone()
        u2 = 0
        if rows_left and rows_left[0] > 0:
            u2 = _patch_from_nba_cdn(con, d_str)
            con.commit()
            if u2:
                print(f"{d_str}: nba cdn patched {u2} games")
        total_updated += u1 + u2

    con.close()
    print(f"Done. Total patched (sbr + cdn): {total_updated}")


if __name__ == "__main__":
    main()
