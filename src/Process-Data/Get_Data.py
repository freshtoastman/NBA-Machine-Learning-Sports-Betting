import argparse
import os
import random
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import toml

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(1, os.fspath(BASE_DIR))

from src.Utils.tools import (  # noqa: E402
    get_json_data,
    pregame_refresh_allowed,
    record_fetch,
    to_data_frame,
)

CONFIG_PATH = BASE_DIR / "config.toml"
DB_PATH = BASE_DIR / "Data" / "TeamData.sqlite"
MIN_DELAY_SECONDS = 1
MAX_DELAY_SECONDS = 3
MAX_RETRIES = 3


def load_config():
    return toml.load(CONFIG_PATH)


def iter_dates(start_date, end_date):
    date_pointer = start_date
    while date_pointer <= end_date:
        yield date_pointer
        date_pointer += timedelta(days=1)


def select_current_season(config, today):
    for season_key, value in config["get-data"].items():
        start_date = datetime.strptime(value["start_date"], "%Y-%m-%d").date()
        end_date = datetime.strptime(value["end_date"], "%Y-%m-%d").date()
        if start_date <= today <= end_date:
            return season_key, value, start_date, end_date
    return None, None, None, None


def get_table_dates(con):
    table_dates = []
    cursor = con.execute("SELECT name FROM sqlite_master WHERE type='table'")
    for (name,) in cursor.fetchall():
        try:
            table_dates.append(datetime.strptime(name, "%Y-%m-%d").date())
        except ValueError:
            continue
    return table_dates


def fetch_data(url, date_pointer, start_year, season_key):
    for attempt in range(1, MAX_RETRIES + 1):
        raw_data = get_json_data(
            url.format(date_pointer.month, date_pointer.day, start_year, date_pointer.year, season_key)
        )
        df = to_data_frame(raw_data)
        if not df.empty:
            return df
        if attempt < MAX_RETRIES:
            time.sleep(MIN_DELAY_SECONDS + random.random() * (MAX_DELAY_SECONDS - MIN_DELAY_SECONDS))
    return pd.DataFrame(data={})


def backfill_season(con, url, season_key, value, existing_dates, today):
    start_date = datetime.strptime(value["start_date"], "%Y-%m-%d").date()
    end_date = datetime.strptime(value["end_date"], "%Y-%m-%d").date()
    fetch_end = min(today - timedelta(days=1), end_date)
    missing_dates = [
        date_pointer
        for date_pointer in iter_dates(start_date, fetch_end)
        if date_pointer not in existing_dates
    ]

    if not missing_dates:
        print(f"No missing dates for season {season_key}.")
        return

    print(f"Backfilling {len(missing_dates)} dates for season {season_key}.")
    for date_pointer in missing_dates:
        print("Getting data:", date_pointer)
        df = fetch_data(url, date_pointer, value["start_year"], season_key)
        if df.empty:
            print("No data returned for:", date_pointer)
            continue

        table_name = date_pointer.strftime("%Y-%m-%d")
        df["Date"] = table_name
        df.to_sql(table_name, con, if_exists="replace", index=False)
        record_fetch(con, table_name)
        existing_dates.add(date_pointer)

        time.sleep(random.randint(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS))

        # TODO: Add tests


def main(config=None, db_path=DB_PATH, today=None, backfill=False, season=None,
         refresh_latest=False, now_utc=None):
    if config is None:
        config = load_config()
    url = config["data_url"]
    if today is None:
        today = datetime.today().date()

    with sqlite3.connect(db_path) as con:
        existing_dates = set(get_table_dates(con))
        if backfill:
            season_items = config["get-data"].items()
            if season:
                season_items = [
                    (key, value) for key, value in season_items if key == season
                ]
                if not season_items:
                    print("Season not found in config:", season)
                    return

            for season_key, value in season_items:
                backfill_season(con, url, season_key, value, existing_dates, today)
            return

        season_key, value, start_date, end_date = select_current_season(config, today)
        if not season_key:
            print("No current season found for today:", today)
            return

        fetch_end = min(today, end_date)
        season_dates = [
            date_value for date_value in existing_dates
            if start_date <= date_value <= fetch_end
        ]
        latest_date = max(season_dates) if season_dates else None
        fetch_start = start_date if latest_date is None else latest_date + timedelta(days=1)

        if refresh_latest and latest_date == fetch_end:
            # Afternoon (Taipei) run: the table named today was fetched at 06:00
            # Taipei = 18:00 ET yesterday, before yesterday's games. Re-fetch it so
            # it holds every game through yesterday, but only while today's own
            # games cannot be in it yet.
            if pregame_refresh_allowed(fetch_end, now_utc):
                print(f"Refreshing today's table before tip-off:", fetch_end)
                fetch_start = fetch_end
            else:
                print(f"Refresh window closed (past 12:00 ET); keeping table", fetch_end)

        if fetch_start > fetch_end:
            print("No new dates to fetch. Latest date:", latest_date)
            return

        for date_pointer in iter_dates(fetch_start, fetch_end):
            print("Getting data:", date_pointer)
            df = fetch_data(url, date_pointer, value["start_year"], season_key)
            if df.empty:
                print("No data returned for:", date_pointer)
                continue

            table_name = date_pointer.strftime("%Y-%m-%d")
            df["Date"] = table_name
            df.to_sql(table_name, con, if_exists="replace", index=False)
            record_fetch(con, table_name)

            time.sleep(random.randint(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS))

            # TODO: Add tests


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch NBA team stats data.")
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Fetch all missing dates for seasons in config.toml.",
    )
    parser.add_argument(
        "--season",
        help="Limit backfill to a single season key (e.g. 2025-26).",
    )
    parser.add_argument(
        "--refresh-latest",
        action="store_true",
        help="Re-fetch today's table if it already exists and it is before 12:00 ET "
             "(afternoon Taipei run: pick up all of yesterday's games).",
    )
    args = parser.parse_args()
    main(backfill=args.backfill, season=args.season, refresh_latest=args.refresh_latest)
