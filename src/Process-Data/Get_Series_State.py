"""Derive NBA playoff series state from OddsData.sqlite.

Rather than calling stats.nba.com (which has unstable playoff endpoints), we
group every game inside the configured playoff date window by matchup and
compute current series scores from Win_Margin signs. This lets us reuse the
already-cached sbrscrape data and stay idempotent.

Output table (per season): series_state_<season-key>
    series_id        TEXT PRIMARY KEY  e.g. "2024-25_005"
    round_num        INTEGER           1=R1, 2=conf-semi, 3=conf-final, 4=finals (0=Play-In)
    round_label      TEXT              繁中: 首輪 / 分區半決賽 / 分區決賽 / 總決賽 / 附加賽
    high_seed        TEXT              alphabetically lower team name
    low_seed         TEXT              alphabetically higher team name
    high_wins        INTEGER
    low_wins         INTEGER
    earliest_date    TEXT              YYYY-MM-DD of first game in the series
    last_updated     TEXT              ISO timestamp

Usage:
    python -m Get_Series_State                    # current playoff season
    python -m Get_Series_State --season 2024-25
    python -m Get_Series_State --all-seasons
"""

import argparse
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import toml

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(1, os.fspath(BASE_DIR))

CONFIG_PATH = BASE_DIR / "config.toml"
ODDS_DB = BASE_DIR / "Data" / "OddsData.sqlite"

ROUND_LABELS = {
    0: "附加賽",
    1: "首輪",
    2: "分區半決賽",
    3: "分區決賽",
    4: "總決賽",
}


def load_config():
    return toml.load(CONFIG_PATH)


def _table_exists(con, name):
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _select_odds_table(con, season_key):
    """Return the freshest odds table name for this season."""
    candidates = [
        f"odds_{season_key}_new",
        f"odds_{season_key}",
        f"{season_key}_new",
        season_key,
    ]
    existing = [t for t in candidates if _table_exists(con, t)]
    if not existing:
        return None
    if len(existing) == 1:
        return existing[0]
    return max(
        existing,
        key=lambda t: con.execute(f'SELECT MAX(Date) FROM "{t}"').fetchone()[0] or "",
    )


def ensure_table(con, season_key):
    table = f"series_state_{season_key}"
    con.execute(
        f'''CREATE TABLE IF NOT EXISTS "{table}" (
            series_id TEXT PRIMARY KEY,
            round_num INTEGER,
            round_label TEXT,
            high_seed TEXT,
            low_seed TEXT,
            high_wins INTEGER,
            low_wins INTEGER,
            earliest_date TEXT,
            last_updated TEXT
        )'''
    )
    return table


def assign_rounds(sorted_series):
    """Bucket series into rounds based on chronological order + game count.

    Heuristic:
      - Play-In teams appear in MULTIPLE matchups (e.g. GS plays LAC then PHX).
        Any matchup containing such a team is Play-In (round 0).
      - Eliminated Play-In teams appear in only one matchup, but their opponent
        is a confirmed Play-In team, so the matchup is still caught.
      - Remaining matchups are real bracket series bucketed chronologically:
        first 8 → R1, next 4 → R2, next 2 → R3, last 1 → R4
    """
    from collections import Counter

    # Step 1: count how many distinct matchups each team appears in.
    team_matchup_count: Counter = Counter()
    for _matchup, data in sorted_series:
        team_matchup_count[data["team_a"]] += 1
        team_matchup_count[data["team_b"]] += 1

    # Teams in 2+ matchups are confirmed Play-In participants.
    playin_teams = {t for t, cnt in team_matchup_count.items() if cnt >= 2}

    play_in = []
    real = []
    for matchup, data in sorted_series:
        max_wins = max(data["wins_a"], data["wins_b"])
        # Confirmed Play-In matchup: either team is a Play-In participant.
        if data["team_a"] in playin_teams or data["team_b"] in playin_teams:
            play_in.append((matchup, data))
        elif data["num_games"] <= 1 and max_wins <= 1:
            # Single-game series whose teams don't appear elsewhere — ambiguous.
            # Could be an eliminated Play-In team's only game, or R1 Game 1.
            # We keep it in play_in tentatively; the bucketing below resolves it.
            play_in.append((matchup, data))
        else:
            real.append((matchup, data))

    # Step 2: bucket real bracket series.
    if len(real) >= 8:
        # Enough R1 series exist — our play_in list is reliable.
        play_in_actual = play_in
        all_real = real
    elif len(real) == 0 and len(play_in) <= 8:
        # Only Play-In games have been played so far; no R1 data yet.
        play_in_actual = play_in
        all_real = []
    else:
        # Mixed: some ambiguous single-game series alongside a partial bracket.
        # Sort everything and keep the last ≤15 as the real bracket.
        all_combined = sorted(play_in + real, key=lambda s: s[1]["earliest_date"])
        play_in_actual = all_combined[:max(0, len(all_combined) - 15)]
        all_real = all_combined[len(play_in_actual):]

    out = []
    for matchup, data in play_in_actual:
        out.append((matchup, data, 0))

    # Bucket real series chronologically.
    sorted_real = sorted(all_real, key=lambda s: s[1]["earliest_date"])
    bucket_sizes = [(8, 1), (4, 2), (2, 3), (1, 4)]
    cursor = 0
    for size, round_num in bucket_sizes:
        chunk = sorted_real[cursor:cursor + size]
        for matchup, data in chunk:
            out.append((matchup, data, round_num))
        cursor += size
    # Anything left over (shouldn't happen in normal NBA format) → label as round 1.
    for matchup, data in sorted_real[cursor:]:
        out.append((matchup, data, 1))

    return out


def derive_series_for_season(season_key):
    """Read OddsData and compute per-series stats. Returns list of dicts."""
    config = load_config()
    bounds = config.get("get-playoffs", {}).get(season_key)
    if not bounds:
        print(f"  no [get-playoffs.{season_key}] in config")
        return []
    start = bounds["start_date"]
    end = bounds["end_date"]

    with sqlite3.connect(ODDS_DB) as con:
        table = _select_odds_table(con, season_key)
        if not table:
            print(f"  no odds table for {season_key}")
            return []
        rows = con.execute(
            f'SELECT Date, Home, Away, Win_Margin, Points FROM "{table}" '
            f'WHERE Date >= ? AND Date <= ? ORDER BY Date',
            (start, end),
        ).fetchall()

    if not rows:
        print(f"  no playoff games found for {season_key} in {table}")
        return []

    # Group by matchup (frozenset of two teams).
    grouped = defaultdict(lambda: {
        "games": [],
        "earliest_date": None,
        "wins_a": 0,
        "wins_b": 0,
        "team_a": None,
        "team_b": None,
        "num_games": 0,
    })
    for date_str, home, away, margin, points in rows:
        # Skip games that haven't been played yet (Points==0 placeholder).
        if margin is None or points in (None, 0):
            continue
        key = frozenset({home, away})
        g = grouped[key]
        # Stable team_a / team_b assignment by alphabetical order.
        teams = sorted([home, away])
        g["team_a"] = teams[0]
        g["team_b"] = teams[1]
        g["games"].append((date_str, home, away, margin))
        g["num_games"] += 1
        if g["earliest_date"] is None or date_str < g["earliest_date"]:
            g["earliest_date"] = date_str
        # margin > 0 means home won. Convert to which alphabetic team won.
        winner = home if margin > 0 else away
        if winner == g["team_a"]:
            g["wins_a"] += 1
        else:
            g["wins_b"] += 1

    sorted_series = sorted(grouped.items(), key=lambda kv: kv[1]["earliest_date"])
    bucketed = assign_rounds(sorted_series)

    # Build output rows.
    out = []
    for idx, (_matchup, data, round_num) in enumerate(bucketed):
        out.append({
            "series_id": f"{season_key}_{idx:03d}",
            "round_num": round_num,
            "round_label": ROUND_LABELS.get(round_num, f"R{round_num}"),
            "high_seed": data["team_a"],  # alphabetic, NOT actual seed (we don't have seed info)
            "low_seed": data["team_b"],
            "high_wins": data["wins_a"],
            "low_wins": data["wins_b"],
            "earliest_date": data["earliest_date"],
        })
    return out


def upsert_series(season_key, series_rows):
    if not series_rows:
        return 0
    now = datetime.now().isoformat(timespec="seconds")
    with sqlite3.connect(ODDS_DB) as con:
        table = ensure_table(con, season_key)
        # Wipe and re-insert — chronological bucketing means series_id can shift
        # if we partially update, so a clean replace is safer.
        con.execute(f'DELETE FROM "{table}"')
        for s in series_rows:
            con.execute(
                f'''INSERT INTO "{table}"
                    (series_id, round_num, round_label, high_seed, low_seed,
                     high_wins, low_wins, earliest_date, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    s["series_id"],
                    s["round_num"],
                    s["round_label"],
                    s["high_seed"],
                    s["low_seed"],
                    s["high_wins"],
                    s["low_wins"],
                    s["earliest_date"],
                    now,
                ),
            )
        con.commit()
    return len(series_rows)


def current_playoff_season(config, today):
    for season_key, value in config.get("get-playoffs", {}).items():
        start = datetime.strptime(value["start_date"], "%Y-%m-%d").date()
        end = datetime.strptime(value["end_date"], "%Y-%m-%d").date()
        if start <= today <= end:
            return season_key
    return None


def main():
    parser = argparse.ArgumentParser(description="Derive playoff series state from OddsData.")
    parser.add_argument("--season", help="Single season key (e.g. 2024-25).")
    parser.add_argument("--all-seasons", action="store_true", help="Every configured playoff season.")
    args = parser.parse_args()

    config = load_config()
    today = datetime.today().date()

    if args.all_seasons:
        seasons = list(config.get("get-playoffs", {}).keys())
    elif args.season:
        seasons = [args.season]
    else:
        s = current_playoff_season(config, today)
        if not s:
            print(f"No active playoff season for today ({today}).")
            return
        seasons = [s]

    for season_key in seasons:
        print(f"Deriving series state for {season_key}...")
        rows = derive_series_for_season(season_key)
        n = upsert_series(season_key, rows)
        print(f"  upserted {n} series")
        for r in rows[:5]:
            print(f"    {r['round_label']}: {r['high_seed']} vs {r['low_seed']} ({r['high_wins']}-{r['low_wins']})")


if __name__ == "__main__":
    main()
