"""Export model predictions + season stats to static JSON for Vercel.

Generates web/data/YYYY-MM-DD.json for today + past 7 days,
plus web/data/season_stats.json and web/data/dates.json (index).

Usage: PYTHONPATH=. python scripts/export_predictions.py
"""
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.Utils.tools import today_taipei, current_nba_season
from src.Utils.SeasonStats import compute_season_stats, reset_cache as reset_season_cache
from src.Utils.Teams import team_name_zh, team_logo_url

OUT_DIR = Path(__file__).resolve().parents[1] / "web" / "data"
DAYS_BACK = 7

EAST_TEAMS = {
    "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets",
    "Chicago Bulls", "Cleveland Cavaliers", "Detroit Pistons", "Indiana Pacers",
    "Miami Heat", "Milwaukee Bucks", "New York Knicks", "Orlando Magic",
    "Philadelphia 76ers", "Toronto Raptors", "Washington Wizards",
}
WEST_TEAMS = {
    "Dallas Mavericks", "Denver Nuggets", "Golden State Warriors", "Houston Rockets",
    "LA Clippers", "Los Angeles Clippers", "Los Angeles Lakers", "Memphis Grizzlies",
    "Minnesota Timberwolves", "New Orleans Pelicans", "Oklahoma City Thunder",
    "Phoenix Suns", "Portland Trail Blazers", "Sacramento Kings",
    "San Antonio Spurs", "Utah Jazz",
}


def _serialize(obj):
    """JSON serializer that handles dates and numpy types."""
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    raise TypeError(f"Not serializable: {type(obj)}")


def export_date(target_date: date) -> dict | None:
    """Generate prediction dict for a single date."""
    from main import predict_historical_xgb

    today = today_taipei()
    is_today = target_date == today

    # Always use historical mode (reads from local SQLite).
    # Never call predict_today_xgb in CI — it may try interactive input()
    # when SBR odds are unavailable, which hangs on stdin EOF.
    games_list = predict_historical_xgb(target_date)

    if not games_list:
        return None

    games_dict = {}
    for g in games_list:
        key = f"{g['away_team']}:{g['home_team']}"
        # Add Chinese names + logo URLs for frontend rendering.
        g["home_team_zh"] = team_name_zh(g["home_team"])
        g["away_team_zh"] = team_name_zh(g["away_team"])
        g["home_logo"] = team_logo_url(g["home_team"])
        g["away_logo"] = team_logo_url(g["away_team"])
        games_dict[key] = g

    # Summary stats.
    n = len(games_list)
    home_picks = sum(1 for g in games_list if g.get("winner") == "home")
    over_picks = sum(1 for g in games_list if g.get("ou_pick") == "OVER")
    avg_conf = sum(max(g.get("home_confidence", 0), g.get("away_confidence", 0)) for g in games_list) / n if n else 0

    summary = {
        "games": n,
        "home_picks": home_picks,
        "away_picks": n - home_picks,
        "over_picks": over_picks,
        "under_picks": n - over_picks,
        "avg_confidence": round(avg_conf, 1),
        "value_picks": sum(1 for g in games_list if g.get("is_value")),
        "golden_picks": sum(1 for g in games_list if g.get("is_golden")),
    }

    if not is_today:
        ml_graded = [g for g in games_list if g.get("ml_correct") is not None]
        ou_graded = [g for g in games_list if g.get("ou_correct") is not None]
        summary["ml_correct"] = sum(1 for g in ml_graded if g["ml_correct"])
        summary["ml_graded"] = len(ml_graded)
        summary["ml_hit_rate"] = round(summary["ml_correct"] / len(ml_graded) * 100, 1) if ml_graded else None
        summary["ou_correct"] = sum(1 for g in ou_graded if g["ou_correct"])
        summary["ou_graded"] = len(ou_graded)
        summary["ou_hit_rate"] = round(summary["ou_correct"] / len(ou_graded) * 100, 1) if ou_graded else None

        # Diamond (value) picks hit rate
        value_games = [g for g in games_list if g.get("is_value")]
        value_decided = [g for g in value_games if g.get("ml_correct") is not None]
        value_wins = sum(1 for g in value_decided if g["ml_correct"])
        summary["value_decided"] = len(value_decided)
        summary["value_wins"] = value_wins
        summary["value_losses"] = len(value_decided) - value_wins
        summary["value_hit_rate"] = round(value_wins / len(value_decided) * 100, 1) if value_decided else None

        # Golden hit rate (subset of value picks)
        golden_games = [g for g in games_list if g.get("is_golden")]
        golden_decided = [g for g in golden_games if g.get("ml_correct") is not None]
        golden_wins = sum(1 for g in golden_decided if g["ml_correct"])
        summary["golden_decided"] = len(golden_decided)
        summary["golden_wins"] = golden_wins
        summary["golden_losses"] = len(golden_decided) - golden_wins
        summary["golden_hit_rate"] = round(golden_wins / len(golden_decided) * 100, 1) if golden_decided else None

    # Active playoff series (for the tracker UI).
    try:
        from src.Utils.PlayoffContext import get_active_series_for_date, is_playoff_date
        active_series = get_active_series_for_date(target_date)
        is_playoff_view = is_playoff_date(target_date)
    except Exception:
        active_series = []
        is_playoff_view = False

    return {
        "date": target_date.isoformat(),
        "is_today": is_today,
        "summary": summary,
        "games": games_dict,
        "active_series": active_series,
        "is_playoff_view": is_playoff_view,
    }


def _team_card(team, wins, losses):
    return {
        "team": team,
        "team_zh": team_name_zh(team),
        "logo": team_logo_url(team),
        "wins": int(wins),
        "losses": int(losses),
        "win_pct": round(wins / (wins + losses), 3) if (wins + losses) else 0.0,
    }


def build_bracket(target_date: date) -> dict | None:
    """Build East/West seedings, play-in matchups and round 1 matchups.

    Reads the latest snapshot in Data/TeamData.sqlite on or before target_date.
    Returns None if no snapshot is available.
    """
    import sqlite3
    db = Path(__file__).resolve().parents[1] / "Data" / "TeamData.sqlite"
    if not db.exists():
        return None
    iso = target_date.isoformat()
    try:
        with sqlite3.connect(db) as con:
            row = con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name <= ? "
                "ORDER BY name DESC LIMIT 1", (iso,),
            ).fetchone()
            if not row:
                return None
            snapshot_date = row[0]
            rows = con.execute(
                f'SELECT TEAM_NAME, W, L FROM "{snapshot_date}"'
            ).fetchall()
    except Exception:
        return None

    east, west = [], []
    for name, w, l in rows:
        card = _team_card(name, w or 0, l or 0)
        if name in EAST_TEAMS:
            east.append(card)
        elif name in WEST_TEAMS:
            west.append(card)

    def _sort(seq):
        seq.sort(key=lambda c: (-c["win_pct"], -c["wins"]))
        for i, c in enumerate(seq, 1):
            c["seed"] = i
        return seq

    east = _sort(east)
    west = _sort(west)

    def _conference(name, seeds):
        # Play-in: 7v8 winner → #7; 9v10 winner vs 7v8 loser → #8.
        play_in = {
            "game_a": {"label": "7 vs 8", "home": seeds[6], "away": seeds[7],
                       "winner_seed": 7, "loser_note": "敗者進 8/9 淘汰賽"},
            "game_b": {"label": "9 vs 10", "home": seeds[8], "away": seeds[9],
                       "winner_note": "晉級 8 號淘汰賽", "loser_note": "淘汰"},
            "game_c": {"label": "8 號種子決定戰", "home": "7/8 敗者",
                       "away": "9/10 勝者", "winner_seed": 8},
        }
        first_round = [
            {"label": "1 vs 8", "high": seeds[0], "low": "待定（8 號種子）"},
            {"label": "4 vs 5", "high": seeds[3], "low": seeds[4]},
            {"label": "3 vs 6", "high": seeds[2], "low": seeds[5]},
            {"label": "2 vs 7", "high": seeds[1], "low": "待定（7 號種子）"},
        ]
        return {
            "name": name,
            "seeds": seeds[:10],
            "lottery": seeds[10:],
            "play_in": play_in,
            "first_round": first_round,
        }

    # Detection window for the UI: 7 days before playoffs start through 10
    # days after (covers pre-playoff anticipation + play-in tournament).
    show_from = show_until = playoff_start = None
    try:
        import toml
        from datetime import datetime as _dt, timedelta as _td
        cfg = toml.load(Path(__file__).resolve().parents[1] / "config.toml")
        for v in cfg.get("get-playoffs", {}).values():
            s = _dt.strptime(v["start_date"], "%Y-%m-%d").date()
            e = _dt.strptime(v["end_date"], "%Y-%m-%d").date()
            if s <= target_date <= e or 0 <= (s - target_date).days <= 14:
                playoff_start = s.isoformat()
                show_from = (s - _td(days=7)).isoformat()
                show_until = (s + _td(days=10)).isoformat()
                break
    except Exception:
        pass

    return {
        "snapshot_date": snapshot_date,
        "generated_for": target_date.isoformat(),
        "playoff_start": playoff_start,
        "show_from": show_from,
        "show_until": show_until,
        "east": _conference("東區", east),
        "west": _conference("西區", west),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    today = today_taipei()
    exported_dates = []

    for offset in range(DAYS_BACK + 1):
        d = today - timedelta(days=offset)
        print(f"Exporting {d}...")
        data = export_date(d)
        if data is None:
            print(f"  skipped (no games)")
            continue
        out_path = OUT_DIR / f"{d.isoformat()}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1, default=_serialize)
        n = data["summary"]["games"]
        print(f"  wrote {n} games → {out_path.name}")
        exported_dates.append(d.isoformat())

    # Season stats.
    season_key = current_nba_season(today)
    reset_season_cache()
    stats = compute_season_stats(season_key)
    if stats:
        stats_path = OUT_DIR / "season_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=1, default=_serialize)
        print(f"Season stats → {stats_path.name}")

    # Playoff bracket (only meaningful near/after playoff start).
    try:
        from src.Utils.PlayoffContext import is_playoff_date
        from datetime import timedelta as _td
        near_playoffs = is_playoff_date(today) or any(
            is_playoff_date(today + _td(days=i)) for i in range(1, 15)
        )
    except Exception:
        near_playoffs = False
    if near_playoffs:
        bracket = build_bracket(today)
        if bracket:
            bracket_path = OUT_DIR / "bracket.json"
            with open(bracket_path, "w", encoding="utf-8") as f:
                json.dump(bracket, f, ensure_ascii=False, indent=1, default=_serialize)
            print(f"Bracket → {bracket_path.name}")

    # Date index.
    index_path = OUT_DIR / "dates.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({"dates": exported_dates, "today": today.isoformat(), "season": season_key},
                  f, ensure_ascii=False, indent=1, default=_serialize)
    print(f"Index → {index_path.name}")
    print("Done.")


if __name__ == "__main__":
    main()
