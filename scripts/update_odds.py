"""Fetch latest SBR odds for unplayed games, update OddsData.sqlite, re-export JSON, push.

Usage: PYTHONPATH=. python scripts/update_odds.py
"""
import sqlite3
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.Utils.tools import today_taipei

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "Data" / "OddsData.sqlite"
SOURCE_TAG = "sbr_fanduel"


def _ensure_history_table(con: sqlite3.Connection) -> None:
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


def _log_history_snapshot(
    con: sqlite3.Connection,
    season: str,
    target_date: date,
    home: str,
    away: str,
    ou,
    spread,
    ml_home,
    ml_away,
    fetched_at_iso: str,
) -> None:
    """Insert a row into odds_history only when spread/OU/ML differ from the last snapshot."""
    row = con.execute(
        "SELECT OU, Spread, ML_Home, ML_Away FROM odds_history "
        "WHERE season=? AND Date=? AND Home=? AND Away=? "
        "ORDER BY fetched_at DESC LIMIT 1",
        (season, str(target_date), home, away),
    ).fetchone()
    changed = row is None or any(
        (row[i] is None) != (v is None) or (v is not None and row[i] != v)
        for i, v in enumerate([ou, spread, ml_home, ml_away])
    )
    if not changed:
        return
    con.execute(
        "INSERT OR REPLACE INTO odds_history "
        "(fetched_at, season, Date, Home, Away, OU, Spread, ML_Home, ML_Away, source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (fetched_at_iso, season, str(target_date), home, away, ou, spread, ml_home, ml_away, SOURCE_TAG),
    )


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


def update_odds_for_date(con: sqlite3.Connection, season_key: str, target_date: date, games: list[dict], sportsbook: str = "fanduel", fetched_at_iso: str | None = None) -> int:
    """Update OU, Spread, ML_Home, ML_Away for unplayed games on target_date. Returns count updated.

    Also appends a snapshot to odds_history whenever any value changes.
    """
    updated = 0
    if fetched_at_iso is None:
        fetched_at_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
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
        odds_tbl = f"odds_{season_key}"
        try:
            con.execute(
                f'UPDATE "{odds_tbl}" SET OU=?, Spread=?, ML_Home=?, ML_Away=? '
                f'WHERE Date=? AND Home=? AND Away=? AND (Points IS NULL OR Points = 0)',
                (ou, spread, ml_home, ml_away, str(target_date), home, away),
            )
        except Exception:
            pass
        _log_history_snapshot(
            con, season_key, target_date, home, away, ou, spread, ml_home, ml_away, fetched_at_iso
        )
        if n:
            print(f"  {home} vs {away}: spread={spread:+.1f}  OU={ou}")
            updated += n
    return updated


def _sync_series_states(con: sqlite3.Connection, season_key: str) -> int:
    """Derive series wins from game results and update series_state table.

    Returns the number of series rows updated.
    """
    ss_tbl = f"series_state_{season_key}"
    try:
        series = con.execute(
            f'SELECT series_id, high_seed, low_seed, high_wins, low_wins, earliest_date '
            f'FROM "{ss_tbl}" WHERE round_num >= 1'
        ).fetchall()
    except Exception:
        return 0

    updated = 0
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for sid, hs, ls, old_hw, old_lw, earliest in series:
        rows = con.execute(
            f'SELECT Home, Away, Win_Margin FROM "{season_key}" '
            f'WHERE Date >= ? AND Win_Margin IS NOT NULL AND Win_Margin != 0 '
            f'AND ((Home=? AND Away=?) OR (Home=? AND Away=?))',
            (earliest, hs, ls, ls, hs),
        ).fetchall()
        hw, lw = 0, 0
        for home, away, margin in rows:
            winner = home if margin > 0 else away
            if winner == hs:
                hw += 1
            elif winner == ls:
                lw += 1
        if hw != (old_hw or 0) or lw != (old_lw or 0):
            con.execute(
                f'UPDATE "{ss_tbl}" SET high_wins=?, low_wins=?, last_updated=? WHERE series_id=?',
                (hw, lw, now_iso, sid),
            )
            updated += 1
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
    fetched_at_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with sqlite3.connect(str(DB_PATH)) as con:
        _ensure_history_table(con)
        unplayed = get_unplayed_dates(con, season_key, today - timedelta(days=2))
        if not unplayed:
            print("No unplayed games found in DB — nothing to update.")
        for d in unplayed:
            print(f"Fetching SBR odds for {d}...")
            games = fetch_sbr_odds_for_date(d)
            if not games:
                print(f"  No games returned from SBR for {d}")
                continue
            n = update_odds_for_date(con, season_key, d, games, fetched_at_iso=fetched_at_iso)
            total_updated += n
            if n == 0:
                print(f"  No rows updated for {d} (odds unchanged or names mismatch)")

        ss_fixed = _sync_series_states(con, season_key)
        if ss_fixed:
            print(f"Synced {ss_fixed} series state(s) from game results.")
            total_updated += ss_fixed

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
