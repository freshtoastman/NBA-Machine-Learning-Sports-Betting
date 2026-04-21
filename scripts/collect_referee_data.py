"""Collect referee assignments from NBA boxscore API and store in SQLite.

Builds a historical referee database for future ATS model improvement.
Run after each game day (e.g., as a post-pipeline step).

Usage: PYTHONPATH=. python scripts/collect_referee_data.py [--days-back N]

Schema:
  referee_games: one row per game, storing official IDs and names
  referee_ats_stats: aggregated ATS cover rates per referee (view)
"""
import argparse
import sqlite3
import time
from datetime import date, timedelta

import requests

from src.Utils.tools import data_headers

REF_DB = "Data/referee_data.sqlite"
ODDS_DB = "Data/OddsData.sqlite"
DEFAULT_DAYS_BACK = 3


def _init_db(con):
    con.executescript("""
    CREATE TABLE IF NOT EXISTS referee_games (
        game_id     TEXT NOT NULL,
        game_date   TEXT NOT NULL,
        home_team   TEXT NOT NULL,
        away_team   TEXT NOT NULL,
        spread      REAL,
        win_margin  REAL,
        home_covered INTEGER,  -- 1=home covered, 0=away, NULL=unknown
        official1_id   INTEGER,
        official1_name TEXT,
        official2_id   INTEGER,
        official2_name TEXT,
        official3_id   INTEGER,
        official3_name TEXT,
        PRIMARY KEY (game_id)
    );
    CREATE TABLE IF NOT EXISTS referee_ats (
        official_id   INTEGER NOT NULL,
        official_name TEXT NOT NULL,
        games         INTEGER NOT NULL DEFAULT 0,
        home_covers   INTEGER NOT NULL DEFAULT 0,
        home_cover_pct REAL,
        last_updated  TEXT,
        PRIMARY KEY (official_id)
    );
    """)
    con.commit()


def _get_game_ids_for_date(d: date) -> list[tuple[str, str, str]]:
    """Returns list of (game_id, home_team_id, away_team_id).

    For today: only Final games (status_id=3).
    For past dates: all games (scoreboardv2 shows historical games as status=1).
    """
    mmddyyyy = d.strftime("%m/%d/%Y")
    is_today = d == date.today()
    try:
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
            gid_idx = headers.index("GAME_ID")
            status_idx = headers.index("GAME_STATUS_ID")
            home_idx = headers.index("HOME_TEAM_ID")
            away_idx = headers.index("VISITOR_TEAM_ID")
            return [
                (row[gid_idx], row[home_idx], row[away_idx])
                for row in rows
                if not is_today or row[status_idx] == 3
            ]
    except Exception as e:
        print(f"  scoreboardv2 error: {e}")
    return []


_TEAM_ID_MAP = {
    1610612737: "Atlanta Hawks", 1610612738: "Boston Celtics",
    1610612751: "Brooklyn Nets", 1610612766: "Charlotte Hornets",
    1610612741: "Chicago Bulls", 1610612739: "Cleveland Cavaliers",
    1610612742: "Dallas Mavericks", 1610612743: "Denver Nuggets",
    1610612765: "Detroit Pistons", 1610612744: "Golden State Warriors",
    1610612745: "Houston Rockets", 1610612754: "Indiana Pacers",
    1610612746: "LA Clippers", 1610612747: "Los Angeles Lakers",
    1610612763: "Memphis Grizzlies", 1610612748: "Miami Heat",
    1610612749: "Milwaukee Bucks", 1610612750: "Minnesota Timberwolves",
    1610612740: "New Orleans Pelicans", 1610612752: "New York Knicks",
    1610612760: "Oklahoma City Thunder", 1610612753: "Orlando Magic",
    1610612755: "Philadelphia 76ers", 1610612756: "Phoenix Suns",
    1610612757: "Portland Trail Blazers", 1610612758: "Sacramento Kings",
    1610612759: "San Antonio Spurs", 1610612761: "Toronto Raptors",
    1610612762: "Utah Jazz", 1610612764: "Washington Wizards",
}


def _fetch_game_officials(game_id: str) -> list[tuple[int, str]]:
    """Returns list of (official_id, full_name) up to 3 officials."""
    try:
        r = requests.get(
            "https://stats.nba.com/stats/boxscoresummaryv2",
            headers=data_headers,
            params={"GameID": game_id},
            timeout=60,
        )
        r.raise_for_status()
        payload = r.json()
        for rs in payload.get("resultSets", []):
            if rs.get("name") != "Officials":
                continue
            headers = rs.get("headers") or []
            rows = rs.get("rowSet") or []
            if not rows:
                return []
            id_idx = headers.index("OFFICIAL_ID")
            fn_idx = headers.index("FIRST_NAME")
            ln_idx = headers.index("LAST_NAME")
            return [
                (row[id_idx], f"{row[fn_idx]} {row[ln_idx]}")
                for row in rows
            ]
    except Exception as e:
        print(f"    officials error {game_id}: {e}")
    return []


def _get_spread_and_margin(con_odds, d_str, home, away):
    """Look up spread and win_margin from OddsData for this game."""
    season_year = int(d_str[:4])
    season_month = int(d_str[5:7])
    season_key = f"{season_year - 1}-{str(season_year)[2:]}" if season_month < 9 else f"{season_year}-{str(season_year + 1)[2:]}"
    try:
        row = con_odds.execute(
            f'SELECT Spread, Win_Margin FROM [{season_key}] WHERE Date=? AND Home=? AND Away=?',
            (d_str, home, away)
        ).fetchone()
        if row and row[1] != 0:
            spread, wm = row
            home_covered = 1 if (wm - spread) > 0 else 0
            return spread, wm, home_covered
    except Exception:
        pass
    return None, None, None


def _update_ats_stats(con):
    """Rebuild referee ATS aggregate stats from referee_games."""
    con.execute("DELETE FROM referee_ats")
    for col_name, col_id in [
        ("official1_name", "official1_id"),
        ("official2_name", "official2_id"),
        ("official3_name", "official3_id"),
    ]:
        con.execute(f"""
            INSERT INTO referee_ats (official_id, official_name, games, home_covers, home_cover_pct, last_updated)
            SELECT {col_id}, {col_name},
                   COUNT(*) as games,
                   SUM(home_covered) as home_covers,
                   ROUND(AVG(home_covered)*100, 1),
                   date('now')
            FROM referee_games
            WHERE {col_id} IS NOT NULL AND home_covered IS NOT NULL
            GROUP BY {col_id}, {col_name}
            ON CONFLICT(official_id) DO UPDATE SET
                games = games + excluded.games,
                home_covers = home_covers + excluded.home_covers,
                home_cover_pct = ROUND(CAST(home_covers + excluded.home_covers AS REAL) / (games + excluded.games) * 100, 1),
                last_updated = excluded.last_updated
        """)
    con.commit()


def collect_for_date(d: date, con_ref, con_odds) -> int:
    d_str = d.strftime("%Y-%m-%d")
    print(f"Collecting referees for {d_str}...")
    game_ids = _get_game_ids_for_date(d)
    if not game_ids:
        print(f"  No final games found")
        return 0

    collected = 0
    for game_id, home_id, away_id in game_ids:
        # Skip if already stored WITH officials. Re-fetch when officials are null
        # (e.g., playoff games that returned 403 before but are now retrievable).
        existing = con_ref.execute(
            "SELECT official1_name FROM referee_games WHERE game_id=?", (game_id,)
        ).fetchone()
        if existing and existing[0]:
            continue

        home = _TEAM_ID_MAP.get(home_id, str(home_id))
        away = _TEAM_ID_MAP.get(away_id, str(away_id))
        officials = _fetch_game_officials(game_id)
        time.sleep(0.6)  # rate limit

        spread, wm, home_covered = _get_spread_and_margin(con_odds, d_str, home, away)

        # Pad to 3 officials
        while len(officials) < 3:
            officials.append((None, None))

        con_ref.execute("""
            INSERT OR REPLACE INTO referee_games
            (game_id, game_date, home_team, away_team, spread, win_margin, home_covered,
             official1_id, official1_name, official2_id, official2_name, official3_id, official3_name)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            game_id, d_str, home, away, spread, wm, home_covered,
            officials[0][0], officials[0][1],
            officials[1][0], officials[1][1],
            officials[2][0], officials[2][1],
        ))
        print(f"  {home} vs {away}: officials={[o[1] for o in officials if o[1]]} covered={home_covered}")
        collected += 1

    con_ref.commit()
    return collected


def print_top_refs(con_ref, min_games=20):
    print("\n=== Top Referee ATS Tendencies (min {min_games} games) ===")
    rows = con_ref.execute("""
        SELECT official_name, games, home_covers, home_cover_pct
        FROM referee_ats
        WHERE games >= ?
        ORDER BY ABS(home_cover_pct - 50) DESC
        LIMIT 20
    """, (min_games,)).fetchall()
    if not rows:
        print("  Not enough data yet.")
        return
    print(f"{'Referee':30} {'Games':>6} {'HomeCvr':>7} {'HmCvr%':>7}")
    for r in rows:
        trend = "HOME+" if r[3] > 50 else "AWAY+"
        print(f"  {r[0]:28} {r[1]:6d} {r[2]:7d} {r[3]:6.1f}% ({trend})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days-back", type=int, default=DEFAULT_DAYS_BACK)
    parser.add_argument("--show-stats", action="store_true")
    args = parser.parse_args()

    con_ref = sqlite3.connect(REF_DB)
    con_odds = sqlite3.connect(ODDS_DB)
    _init_db(con_ref)

    today = date.today()
    total = 0
    for i in range(args.days_back):
        d = today - timedelta(days=i)
        total += collect_for_date(d, con_ref, con_odds)
        time.sleep(1)

    if total > 0:
        print(f"\nUpdating ATS stats for all referees...")
        _update_ats_stats(con_ref)

    if args.show_stats:
        print_top_refs(con_ref)

    print(f"\nDone. Collected {total} new game entries.")
    con_ref.close()
    con_odds.close()


if __name__ == "__main__":
    main()
