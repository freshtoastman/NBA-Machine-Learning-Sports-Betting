"""Playoff series context — convert series_state_* tables into per-game features
and human-readable strings for the AI prompt.

Lookups are lru_cached because Flask hits this on every game card render.
"""

from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path
from datetime import date, datetime

import toml

BASE_DIR = Path(__file__).resolve().parents[2]
ODDS_DB = BASE_DIR / "Data" / "OddsData.sqlite"
CONFIG_PATH = BASE_DIR / "config.toml"


def _season_for_date(target_date) -> str:
    """Return the NBA season key (YYYY-YY) covering the given date."""
    if isinstance(target_date, str):
        target_date = datetime.strptime(target_date, "%Y-%m-%d").date()
    year = target_date.year
    if target_date.month >= 10:
        return f"{year}-{(year + 1) % 100:02d}"
    return f"{year - 1}-{year % 100:02d}"


def is_playoff_date(target_date) -> bool:
    """True if this date falls inside any configured [get-playoffs.*] window."""
    if isinstance(target_date, str):
        target_date = datetime.strptime(target_date, "%Y-%m-%d").date()
    try:
        config = toml.load(CONFIG_PATH)
    except Exception:
        return False
    for value in config.get("get-playoffs", {}).values():
        start = datetime.strptime(value["start_date"], "%Y-%m-%d").date()
        end = datetime.strptime(value["end_date"], "%Y-%m-%d").date()
        if start <= target_date <= end:
            return True
    return False


@lru_cache(maxsize=32)
def _load_series_table(season_key: str):
    """Load the entire series_state_<season> table as a dict keyed by frozenset({team_a, team_b})."""
    table = f"series_state_{season_key}"
    out = {}
    try:
        with sqlite3.connect(ODDS_DB) as con:
            rows = con.execute(
                f'SELECT round_num, round_label, high_seed, low_seed, high_wins, low_wins, earliest_date '
                f'FROM "{table}"'
            ).fetchall()
        for round_num, round_label, high_seed, low_seed, high_wins, low_wins, earliest_date in rows:
            if not high_seed or not low_seed:
                continue
            key = frozenset({high_seed, low_seed})
            out[key] = {
                "round_num": round_num,
                "round_label": round_label,
                "high_seed": high_seed,
                "low_seed": low_seed,
                "high_wins": high_wins or 0,
                "low_wins": low_wins or 0,
                "earliest_date": earliest_date,
            }
    except Exception:
        pass
    return out


def _count_series_games_before(home_team: str, away_team: str, target_date, season_key: str, series_start: str | None = None) -> int | None:
    """Count completed games for this matchup in the playoff series before target_date.

    series_start restricts to games on or after the series' earliest_date so that
    regular-season games between the same teams are excluded.
    Returns number of completed games (Points > 0) or None if query fails.
    Used to reconstruct the pre-game series state for historical exports.
    """
    if isinstance(target_date, str):
        target_date = datetime.strptime(target_date, "%Y-%m-%d").date()
    table = f"[{season_key}]"
    try:
        with sqlite3.connect(ODDS_DB) as con:
            if series_start:
                # Count all scheduled games in this series before target_date
                # (no Points check — unscored past games still happened)
                row = con.execute(
                    f'SELECT COUNT(*) FROM {table} '
                    f'WHERE Date >= ? AND Date < ? '
                    f'AND ((Home = ? AND Away = ?) OR (Home = ? AND Away = ?))',
                    (series_start, target_date.isoformat(), home_team, away_team, away_team, home_team),
                ).fetchone()
            else:
                # Legacy fallback: require Points > 0 to avoid counting regular-season games
                row = con.execute(
                    f'SELECT COUNT(*) FROM {table} '
                    f'WHERE Date < ? AND Points > 0 '
                    f'AND ((Home = ? AND Away = ?) OR (Home = ? AND Away = ?))',
                    (target_date.isoformat(), home_team, away_team, away_team, home_team),
                ).fetchone()
            return row[0] if row else 0
    except Exception:
        return None


def get_series_state(home_team: str, away_team: str, game_date, as_of_date=None) -> dict | None:
    """Return the series state for this matchup at game time.

    When as_of_date is provided (a date object or ISO string), series_game_num is
    computed from games completed before that date rather than from the current
    series_state table. Use this for historical exports to avoid showing G2 signals
    on G1 games after the series state has been updated.

    Returned dict keys:
        round_num, round_label, high_seed, low_seed,
        home_wins, away_wins, total_games_played, series_game_num,
        is_must_win_for ('home'|'away'|None), is_elimination,
        status_text (zh, e.g. '湖人領先 3-1')
    """
    if not is_playoff_date(game_date):
        return None
    season_key = _season_for_date(game_date)
    series_map = _load_series_table(season_key)
    raw = series_map.get(frozenset({home_team, away_team}))
    if raw is None:
        return None

    if raw["high_seed"] == home_team:
        home_wins = raw["high_wins"]
        away_wins = raw["low_wins"]
    else:
        home_wins = raw["low_wins"]
        away_wins = raw["high_wins"]

    if as_of_date is not None:
        series_start = raw.get("earliest_date")
        games_before = _count_series_games_before(home_team, away_team, as_of_date, season_key, series_start)
        if games_before is not None:
            total_played = games_before
        else:
            total_played = home_wins + away_wins
    else:
        total_played = home_wins + away_wins
    series_game_num = total_played + 1  # the upcoming game
    is_must_win_for = None
    if home_wins == 3 and away_wins < 3:
        is_must_win_for = "away"
    elif away_wins == 3 and home_wins < 3:
        is_must_win_for = "home"
    is_elimination = is_must_win_for is not None

    if home_wins > away_wins:
        leader_zh = home_team
        status_text = f"主隊領先 {home_wins}-{away_wins}"
    elif away_wins > home_wins:
        leader_zh = away_team
        status_text = f"客隊領先 {away_wins}-{home_wins}"
    else:
        leader_zh = None
        status_text = f"平手 {home_wins}-{away_wins}"

    # Detect if away/home team came through a play-in (round_num=0) series.
    # This fires the "附加賽升組客隊不利" signal for R1G1 where the visitor
    # is a play-in survivor facing a rested, seeded home team.
    away_from_playin = False
    home_from_playin = False
    if raw.get("round_num", 0) >= 1:
        for key_pair, s_data in series_map.items():
            if s_data.get("round_num", 0) == 0:
                if away_team in key_pair:
                    away_from_playin = True
                if home_team in key_pair:
                    home_from_playin = True

    # Compute G1 ATS margin for the current home team.
    # Positive = home covered (home won by more than the spread).
    # Used by the G2 blowout follow-through signal (>7 → 64.3% home covers).
    g1_ats_margin: float | None = None
    series_start = raw.get("earliest_date")
    if series_game_num == 2 and series_start:
        try:
            tbl_od = season_key
            with sqlite3.connect(ODDS_DB) as _con:
                row = _con.execute(
                    f'SELECT Home, Away, Win_Margin, Spread FROM "{tbl_od}"'
                    f' WHERE Date = ? AND ((Home = ? AND Away = ?) OR (Home = ? AND Away = ?))'
                    f' AND Points > 0',
                    (series_start, home_team, away_team, away_team, home_team),
                ).fetchone()
                if row:
                    _g1_home, _g1_away, _wm, _sp = row
                    if _wm is not None and _sp is not None:
                        # Normalise so margin is always relative to the G2 home team
                        if _g1_home == home_team:
                            g1_ats_margin = float(_wm) - float(_sp)
                        else:
                            # G1 home ≠ G2 home (series alternates courts)
                            g1_ats_margin = -(float(_wm) - float(_sp))
        except Exception:
            pass

    return {
        "round_num": raw["round_num"],
        "round_label": raw["round_label"],
        "home_team": home_team,
        "away_team": away_team,
        "home_wins": home_wins,
        "away_wins": away_wins,
        "total_games_played": total_played,
        "series_game_num": series_game_num,
        "is_must_win_for": is_must_win_for,
        "is_elimination": is_elimination,
        "status_text": status_text,
        "leader": leader_zh,
        "away_from_playin": away_from_playin,
        "home_from_playin": home_from_playin,
        "g1_ats_margin": g1_ats_margin,
    }


def build_playoff_features(state: dict | None) -> dict:
    """Convert a series state dict to numeric features for the model.

    Always returns the same keys (zero-filled when state is None) so the trainer
    sees a stable feature set regardless of regular season vs playoff.
    """
    if state is None:
        return {
            "is_playoff": 0,
            "series_game_num": 0,
            "series_lead_for_home": 0,
            "is_elimination_game": 0,
        }
    return {
        "is_playoff": 1,
        "series_game_num": int(state["series_game_num"]),
        "series_lead_for_home": int(state["home_wins"] - state["away_wins"]),
        "is_elimination_game": 1 if state["is_elimination"] else 0,
    }


def get_active_series_for_date(target_date) -> list[dict]:
    """Return series that are PROBABLY in progress on the given date.

    Uses the earliest_date stored in series_state to filter out series that
    have already concluded long ago (e.g., showing Play-In games while the
    Conference Finals are happening).
    """
    if not is_playoff_date(target_date):
        return []
    if isinstance(target_date, str):
        target_date = datetime.strptime(target_date, "%Y-%m-%d").date()
    season_key = _season_for_date(target_date)

    # Re-read the table including earliest_date this time (not in the cached load).
    table = f"series_state_{season_key}"
    raw_rows = []
    try:
        with sqlite3.connect(ODDS_DB) as con:
            for row in con.execute(
                f'SELECT round_num, round_label, high_seed, low_seed, high_wins, low_wins, earliest_date '
                f'FROM "{table}"'
            ):
                raw_rows.append(row)
    except Exception:
        return []

    out = []
    for round_num, round_label, high_seed, low_seed, high_wins, low_wins, earliest in raw_rows:
        if not high_seed or not low_seed:
            continue
        # Skip Play-In after a few days — it's single-elimination and not a "series".
        if round_num == 0 and earliest:
            try:
                e = datetime.strptime(earliest, "%Y-%m-%d").date()
                if (target_date - e).days > 5:
                    continue
            except ValueError:
                pass
        # Mark as concluded once one side reaches 4 wins (best-of-7).
        if (high_wins or 0) >= 4 or (low_wins or 0) >= 4:
            # Include conclude series only if it ended within last 3 days.
            if earliest:
                try:
                    e = datetime.strptime(earliest, "%Y-%m-%d").date()
                    if (target_date - e).days > 21:
                        continue
                except ValueError:
                    continue
            else:
                continue
        out.append({
            "round_num": round_num,
            "round_label": round_label,
            "high_seed": high_seed,
            "low_seed": low_seed,
            "high_wins": high_wins or 0,
            "low_wins": low_wins or 0,
        })
    # Sort by round_num descending so finals appear first.
    out.sort(key=lambda s: -(s.get("round_num") or 0))
    return out
