"""Fetch latest SBR odds for unplayed games, update OddsData.sqlite, re-export JSON, push.

Usage: PYTHONPATH=. python scripts/update_odds.py
"""
import sqlite3
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.Utils.tools import today_taipei

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "Data" / "OddsData.sqlite"


def fetch_sbr_odds_for_date(target_date: date) -> list[dict]:
    """Fetch SBR scoreboard for a given date, return list of game dicts."""
    try:
        from sbrscrape import Scoreboard
        sb = Scoreboard(date=target_date)
        return sb.games if hasattr(sb, "games") else []
    except Exception as e:
        print(f"  SBR fetch failed for {target_date}: {e}")
        return []


def get_unplayed_dates(con: sqlite3.Connection, season_key: str, from_date: date) -> list[date]:
    """Return dates in the DB where all games have Points=NULL (unplayed)."""
    rows = con.execute(
        f'SELECT DISTINCT Date FROM "{season_key}" WHERE Date >= ? AND (Points IS NULL OR Points = 0)',
        (str(from_date),),
    ).fetchall()
    return [date.fromisoformat(r[0]) for r in rows]


def update_odds_for_date(con: sqlite3.Connection, season_key: str, target_date: date, games: list[dict], sportsbook: str = "fanduel") -> int:
    """Update OU, Spread, ML_Home, ML_Away for unplayed games on target_date. Returns count updated."""
    updated = 0
    for game in games:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        try:
            ou = game["total"][sportsbook]
            spread = game["away_spread"][sportsbook]  # positive = away underdog
            ml_home = game["home_ml"][sportsbook]
            ml_away = game["away_ml"][sportsbook]
        except (KeyError, TypeError):
            continue
        if ou is None and spread is None:
            continue
        n = con.execute(
            f'UPDATE "{season_key}" SET OU=?, Spread=?, ML_Home=?, ML_Away=? '
            f'WHERE Date=? AND Home=? AND Away=? AND (Points IS NULL OR Points = 0)',
            (ou, spread, ml_home, ml_away, str(target_date), home, away),
        ).rowcount
        if n:
            print(f"  {home} vs {away}: spread={spread:+.1f}  OU={ou}")
            updated += n
    return updated


def main():
    today = today_taipei()
    # Check today and yesterday (Taipei +8 → may cover ET "today")
    check_dates = [today - timedelta(days=1), today]

    try:
        from src.Utils.tools import current_nba_season
        season_key = current_nba_season()
    except Exception:
        season_key = "2025-26"

    total_updated = 0
    with sqlite3.connect(str(DB_PATH)) as con:
        unplayed = get_unplayed_dates(con, season_key, today - timedelta(days=2))
        if not unplayed:
            print("No unplayed games found in DB — nothing to update.")
        for d in unplayed:
            print(f"Fetching SBR odds for {d}...")
            games = fetch_sbr_odds_for_date(d)
            if not games:
                print(f"  No games returned from SBR for {d}")
                continue
            n = update_odds_for_date(con, season_key, d, games)
            total_updated += n
            if n == 0:
                print(f"  No rows updated for {d} (odds unchanged or names mismatch)")
        con.commit()

    if total_updated == 0:
        print("Odds unchanged — skipping re-export and push.")
        return

    print(f"\nUpdated {total_updated} game(s). Re-exporting JSON...")
    result = subprocess.run(
        ["python", "scripts/export_predictions.py"],
        capture_output=True, text=True, cwd=str(BASE_DIR),
        env={**__import__("os").environ, "PYTHONPATH": str(BASE_DIR)},
    )
    if result.returncode != 0:
        print("Export failed:", result.stderr[-500:])
        sys.exit(1)
    print(result.stdout.strip())

    print("\nPushing to git...")
    subprocess.run(["git", "add", "web/data/"], cwd=str(BASE_DIR), check=True)
    diff = subprocess.run(["git", "diff", "--cached", "--stat"], capture_output=True, text=True, cwd=str(BASE_DIR))
    if not diff.stdout.strip():
        print("No JSON changes to commit.")
        return
    subprocess.run(
        ["git", "commit", "-m", f"Auto-update odds {today}"],
        cwd=str(BASE_DIR), check=True,
    )
    subprocess.run(["git", "push", "origin", "master"], cwd=str(BASE_DIR), check=True)
    print("Done.")


if __name__ == "__main__":
    main()
