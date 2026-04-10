"""Patch final scores from sbrscrape for recent dates where Points=0.

Usage: PYTHONPATH=. python scripts/patch_scores.py
"""
import sqlite3
from datetime import date, timedelta
from sbrscrape import Scoreboard

ODDS_DB = "Data/OddsData.sqlite"
DAYS_BACK = 3


def main():
    con = sqlite3.connect(ODDS_DB)
    today = date.today()
    total_updated = 0

    for offset in range(DAYS_BACK + 1):
        d = today - timedelta(days=offset)
        d_str = d.isoformat()

        # Check if any games on this date still have Points=0.
        rows = con.execute(
            'SELECT COUNT(*) FROM "2025-26" WHERE Date = ? AND (Points IS NULL OR Points = 0)',
            (d_str,),
        ).fetchone()
        if not rows or rows[0] == 0:
            continue

        try:
            sb = Scoreboard(sport="NBA", date=d_str)
            games = sb.games if hasattr(sb, "games") else []
        except Exception as exc:
            print(f"{d_str}: sbrscrape error: {exc}")
            continue

        updated = 0
        for g in games:
            if g.get("status") != "Final":
                continue
            home = g.get("home_team", "").replace("Los Angeles Clippers", "LA Clippers")
            away = g.get("away_team", "").replace("Los Angeles Clippers", "LA Clippers")
            hs = g.get("home_score")
            aws = g.get("away_score")
            if hs is None or aws is None:
                continue
            points = hs + aws
            win_margin = hs - aws
            con.execute(
                'UPDATE "2025-26" SET Points = ?, Win_Margin = ? '
                "WHERE Date = ? AND Home = ? AND Away = ? AND (Points IS NULL OR Points = 0)",
                (points, win_margin, d_str, home, away),
            )
            updated += 1

        con.commit()
        if updated:
            print(f"{d_str}: patched {updated} games")
            total_updated += updated

    con.close()
    print(f"Done. Total patched: {total_updated}")


if __name__ == "__main__":
    main()
