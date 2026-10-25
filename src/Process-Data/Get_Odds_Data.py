import argparse
import os
import random
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import toml

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(1, os.fspath(BASE_DIR))

# Not sbrscrape: SBR answers 403 to its python user agent since 2026-10 (see module doc).
from src.Utils.sbr_scoreboard import Scoreboard  # noqa: E402
# 2026-10-23: SBR had FanDuel for 0 of 12 games at 10:05 ET; fall back to other
# SBR books, then ESPN (DraftKings), instead of inserting nothing for the date.
from src.Utils.odds_sources import lines_for_games, sbr_line  # noqa: E402

CONFIG_PATH = BASE_DIR / "config.toml"
DB_PATH = BASE_DIR / "Data" / "OddsData.sqlite"
MIN_DELAY_SECONDS = 1
MAX_DELAY_SECONDS = 3


def load_config():
    return toml.load(CONFIG_PATH)


def iter_dates(start_date, end_date):
    date_pointer = start_date
    while date_pointer <= end_date:
        yield date_pointer
        date_pointer += timedelta(days=1)


def fetch_scoreboard(date_pointer):
    try:
        return Scoreboard(date=date_pointer)
    except Exception as exc:
        print(f"Failed to fetch odds data for {date_pointer}: {exc}")
        return None


def parse_date_value(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(value).date()
        except ValueError:
            return None
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value)).date()
    except ValueError:
        return None


def table_exists(con, table_name):
    cursor = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    return cursor.fetchone() is not None


def select_current_season(config, today):
    for season_key, value in config["get-odds-data"].items():
        start_date = datetime.strptime(value["start_date"], "%Y-%m-%d").date()
        end_date = datetime.strptime(value["end_date"], "%Y-%m-%d").date()
        if start_date <= today <= end_date:
            return season_key, value, start_date, end_date
    return None, None, None, None


def get_existing_dates(con, season_key):
    if not table_exists(con, season_key):
        return set()
    cursor = con.execute(f'SELECT DISTINCT Date FROM "{season_key}"')
    dates = set()
    for (value,) in cursor.fetchall():
        date_value = parse_date_value(value)
        if date_value:
            dates.add(date_value)
    return dates


def get_teams_last_played(con, season_key, before_date):
    if not table_exists(con, season_key):
        return {}
    df = pd.read_sql_query(
        f'SELECT Date, Home, Away FROM "{season_key}" WHERE Date < ?',
        con,
        params=[before_date],
    )
    teams_last_played = {}
    for row in df.itertuples(index=False):
        date_value = parse_date_value(row.Date)
        if not date_value:
            continue
        teams_last_played[row.Home] = max(date_value, teams_last_played.get(row.Home, date_value))
        teams_last_played[row.Away] = max(date_value, teams_last_played.get(row.Away, date_value))
    return teams_last_played


def get_existing_games_by_date(con, season_key, start_date, end_date):
    if not table_exists(con, season_key):
        return {}
    df = pd.read_sql_query(
        f'SELECT Date, Home, Away FROM "{season_key}" WHERE Date BETWEEN ? AND ?',
        con,
        params=[start_date, end_date],
    )
    games_by_date = {}
    for row in df.itertuples(index=False):
        date_value = parse_date_value(row.Date)
        if not date_value:
            continue
        games_by_date.setdefault(date_value, []).append((row.Home, row.Away))
    return games_by_date


def append_game_rows(game_rows, date_pointer, game, sportsbook, teams_last_played, line=None):
    """Append one season-table row. `line` (see odds_sources) overrides the per-book
    lookup; without it the preferred book, then any other SBR book, is used, and a
    game no book prices raises KeyError like the original single-book code did."""
    if line is None:
        line = sbr_line(game, sportsbook)
    if line is None:
        raise KeyError(sportsbook)
    def days_rest(team):
        last_played = teams_last_played.get(team)
        if last_played is None:
            return 7
        return (date_pointer - last_played).days

    home_team = game["home_team"]
    away_team = game["away_team"]
    home_days = days_rest(home_team)
    away_days = days_rest(away_team)

    teams_last_played[home_team] = date_pointer
    teams_last_played[away_team] = date_pointer

    game_rows.append({
        "Date": date_pointer,
        "Home": home_team,
        "Away": away_team,
        "OU": line["ou"],
        "Spread": line["spread"],
        "ML_Home": line["ml_home"],
        "ML_Away": line["ml_away"],
        "Points": (line["away_score"] or 0) + (line["home_score"] or 0),
        "Win_Margin": (line["home_score"] or 0) - (line["away_score"] or 0),
        "Days_Rest_Home": home_days,
        "Days_Rest_Away": away_days,
        # Not a table column (dropped by rows_frame); feeds the odds_history source tag.
        "_source": line["source"],
    })


def rows_frame(rows):
    """DataFrame for to_sql without the private bookkeeping keys."""
    return pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])


def collect_odds_for_dates(dates, sportsbook, teams_last_played, skip_pairs=None):
    """Fetch and build season-table rows for `dates`. `skip_pairs` maps a date to
    the (Home, Away) pairs already stored for it, so a partially inserted slate
    (2026-10-25: 1 of 7 games priced at 06:00) can be completed without duplicating
    the rows that exist."""
    df_data = []
    skip_pairs = skip_pairs or {}
    for date_pointer in dates:
        print("Getting odds data:", date_pointer)
        sb = fetch_scoreboard(date_pointer)
        sbr_games = list(sb.games) if sb and getattr(sb, "games", None) else []
        # Preferred book → other SBR books → ESPN; with an empty SBR board inside the
        # live window the ESPN slate itself is the game list.
        lines = lines_for_games(sbr_games, date_pointer, sportsbook)
        if not lines:
            if sbr_games:
                print(f"No priced games for {date_pointer} on SBR or ESPN ({len(sbr_games)} listed)")
            time.sleep(random.randint(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS))
            continue

        stored = set(skip_pairs.get(date_pointer, ()))
        if stored:
            before = len(lines)
            lines = [l for l in lines if (l["home_team"], l["away_team"]) not in stored]
            print(f"  {date_pointer}: {len(stored)} game(s) already stored, {len(lines)} of {before} priced game(s) new")
            if not lines:
                time.sleep(random.randint(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS))
                continue

        for line in lines:
            game = {"home_team": line["home_team"], "away_team": line["away_team"]}
            append_game_rows(df_data, date_pointer, game, sportsbook, teams_last_played, line=line)
        n_fallback = sum(1 for l in lines if l["source"] != f"sbr_{sportsbook}")
        if n_fallback:
            print(f"  {date_pointer}: {n_fallback} of {len(lines)} line(s) from a fallback source: "
                  + ", ".join(sorted({l['source'] for l in lines if l['source'] != f'sbr_{sportsbook}'})))

        time.sleep(random.randint(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS))

    return df_data


def backfill_season(con, season_key, value, sportsbook, today):
    start_date = datetime.strptime(value["start_date"], "%Y-%m-%d").date()
    end_date = datetime.strptime(value["end_date"], "%Y-%m-%d").date()
    fetch_end = min(today - timedelta(days=1), end_date)

    existing_by_date = get_existing_games_by_date(con, season_key, start_date, fetch_end)
    teams_last_played = {}
    new_rows = []

    for date_pointer in iter_dates(start_date, fetch_end):
        if date_pointer in existing_by_date:
            for home_team, away_team in existing_by_date[date_pointer]:
                teams_last_played[home_team] = date_pointer
                teams_last_played[away_team] = date_pointer
            continue

        sb = fetch_scoreboard(date_pointer)
        if not sb or not hasattr(sb, "games") or not sb.games:
            time.sleep(random.randint(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS))
            continue

        print("Backfilling odds data:", date_pointer)
        for game in sb.games:
            try:
                append_game_rows(new_rows, date_pointer, game, sportsbook, teams_last_played)
            except KeyError:
                print(f"No {sportsbook} odds data found for game: {game}")

        time.sleep(random.randint(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS))

    if new_rows:
        rows_frame(new_rows).to_sql(season_key, con, if_exists="append", index=False)


OPEN_SOURCE_TAG = "sbr_fanduel_open"


def ensure_history_table(con):
    """Same schema as scripts/update_odds.py (CREATE IF NOT EXISTS on both sides)."""
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS odds_history (
            fetched_at TEXT NOT NULL,
            season TEXT NOT NULL,
            Date TEXT NOT NULL,
            Home TEXT NOT NULL,
            Away TEXT NOT NULL,
            OU REAL,
            Spread REAL,
            ML_Home INTEGER,
            ML_Away INTEGER,
            source TEXT,
            PRIMARY KEY (fetched_at, Date, Home, Away)
        )
        """
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_odds_history_game ON odds_history(season, Date, Home, Away)"
    )


def log_opening_snapshots(con, season_key, game_rows, fetched_at=None):
    """Record the line a game is first seen with as its opening odds_history snapshot.

    2026-10-20 (opening night): Get_Odds_Data inserted NYK -5.5 / total 230.5 at
    08:57 Taipei, update_odds --db-only overwrote the season table with -4.5 /
    232.5 at 14:57 and logged only that as the first snapshot, so spread_first
    (the line-movement candidate feature) lost the real open. Only the daily
    (non-backfill) path calls this; backfilled dates hold closing lines.
    Returns the number of snapshots written.
    """
    if not game_rows:
        return 0
    ensure_history_table(con)
    if fetched_at is None:
        fetched_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    written = 0
    for row in game_rows:
        date_str = str(row["Date"])
        exists = con.execute(
            "SELECT 1 FROM odds_history WHERE season=? AND Date=? AND Home=? AND Away=? LIMIT 1",
            (season_key, date_str, row["Home"], row["Away"]),
        ).fetchone()
        if exists:
            continue
        con.execute(
            "INSERT OR REPLACE INTO odds_history "
            "(fetched_at, season, Date, Home, Away, OU, Spread, ML_Home, ML_Away, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (fetched_at, season_key, date_str, row["Home"], row["Away"],
             row["OU"], row["Spread"], row["ML_Home"], row["ML_Away"],
             f"{row['_source']}_open" if row.get("_source") else OPEN_SOURCE_TAG),
        )
        written += 1
    return written


def main(sportsbook="fanduel", backfill=False, season=None, today=None, db_path=DB_PATH):
    config = load_config()
    if today is None:
        today = datetime.today().date()

    with sqlite3.connect(db_path) as con:
        if backfill:
            season_items = config["get-odds-data"].items()
            if season:
                season_items = [
                    (key, value) for key, value in season_items if key == season
                ]
                if not season_items:
                    print("Season not found in config:", season)
                    return
            for season_key, value in season_items:
                backfill_season(con, season_key, value, sportsbook, today)
            return

        season_key, value, start_date, end_date = select_current_season(config, today)
        if not season_key:
            print("No current season found for today:", today)
            return

        fetch_end = min(today, end_date)
        existing_dates = get_existing_dates(con, season_key)
        latest_date = max(existing_dates) if existing_dates else None
        fetch_start = start_date if latest_date is None else latest_date + timedelta(days=1)
        df_data = []
        if fetch_start > fetch_end:
            print("No new odds dates to fetch. Latest date:", latest_date)
        else:
            teams_last_played = get_teams_last_played(con, season_key, fetch_start)
            df_data = collect_odds_for_dates(iter_dates(fetch_start, fetch_end), sportsbook, teams_last_played)

        # A date inserted with a partial board is never revisited by the branch above
        # (fetch_start = latest + 1) and update_odds only UPDATEs rows that exist, so the
        # 2026-10-25 slate sat at 1 of 7 games all day (06:00: SBR priced one game, ESPN
        # none). Re-scan the still-unplayed dates already in the table and append the
        # games whose (Home, Away) pair is missing.
        df_data += complete_partial_dates(con, season_key, sportsbook, today, fetch_end)

        if df_data:
            rows_frame(df_data).to_sql(season_key, con, if_exists="append", index=False)
            n_open = log_opening_snapshots(con, season_key, df_data)
            print(f"Logged {n_open} opening line snapshot(s) to odds_history.")


def unplayed_dates(con, season_key, from_date, to_date):
    """Dates in [from_date, to_date] whose stored games have no final score yet."""
    if not table_exists(con, season_key):
        return []
    rows = con.execute(
        f'SELECT DISTINCT Date FROM "{season_key}" WHERE Date BETWEEN ? AND ? '
        f'AND (Points IS NULL OR Points = 0)',
        (str(from_date), str(to_date)),
    ).fetchall()
    dates = {parse_date_value(v) for (v,) in rows}
    return sorted(d for d in dates if d)


def complete_partial_dates(con, season_key, sportsbook, today, fetch_end):
    """Rows for games missing from already-inserted, still-unplayed dates (see main)."""
    rescan = unplayed_dates(con, season_key, today - timedelta(days=1), fetch_end)
    if not rescan:
        return []
    stored = get_existing_games_by_date(con, season_key, rescan[0], rescan[-1])
    # Days-rest must be computed from games before the slate, not from the slate itself.
    teams_last_played = get_teams_last_played(con, season_key, rescan[0])
    rows = collect_odds_for_dates(rescan, sportsbook, teams_last_played, skip_pairs=stored)
    if rows:
        print(f"Completed {len(rows)} game(s) missing from partially inserted date(s): "
              + ", ".join(str(d) for d in rescan))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch NBA odds data from SBR.")
    parser.add_argument(
        "--sportsbook",
        default="fanduel",
        help="Sportsbook key to pull odds from (default: fanduel).",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Fetch missing odds dates for seasons in config.toml.",
    )
    parser.add_argument(
        "--season",
        help="Limit backfill to a single season key (e.g. 2025-26).",
    )
    args = parser.parse_args()
    main(sportsbook=args.sportsbook, backfill=args.backfill, season=args.season)
