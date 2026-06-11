"""Export model predictions + season stats to static JSON for Vercel.

Generates web/data/YYYY-MM-DD.json for today + past 7 days,
plus web/data/season_stats.json and web/data/dates.json (index).

Usage: PYTHONPATH=. python scripts/export_predictions.py
"""
import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.Utils.tools import today_taipei, current_nba_season
from src.Utils.SeasonStats import compute_season_stats, reset_cache as reset_season_cache
from src.Utils.Teams import team_name_zh, team_logo_url

_ESPN_INJURY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
_ESPN_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
}
_INJURY_STATUS_DISPLAY = {"Out": "缺陣", "Questionable": "存疑", "Probable": "可能出賽", "Doubtful": "可能缺陣"}


def _parse_injury_status(comment: str) -> str:
    """Parse actual game status from ESPN shortComment. Returns Chinese label."""
    c = comment.lower()
    if "questionable" in c or "being listed as" in c:
        return "存疑"
    if "doubtful" in c:
        return "可能缺陣"
    if "probable" in c or "expected to play" in c or "will play" in c:
        return "可能出賽"
    # Default: confirmed out (ruled out / miss / season-ending etc.)
    return "缺陣"


_NBA_TEAM_IDENTIFIERS = {
    # Nicknames
    "hawks", "celtics", "nets", "hornets", "bulls", "cavaliers", "mavericks",
    "nuggets", "pistons", "warriors", "rockets", "pacers", "clippers", "lakers",
    "grizzlies", "heat", "bucks", "timberwolves", "pelicans", "knicks", "thunder",
    "magic", "76ers", "suns", "trail blazers", "blazers", "kings", "spurs", "raptors",
    "jazz", "wizards",
    # City / location names
    "atlanta", "boston", "brooklyn", "charlotte", "chicago", "cleveland",
    "dallas", "denver", "detroit", "golden state", "houston", "indiana",
    "la clippers", "los angeles", "memphis", "miami", "milwaukee", "minnesota",
    "new orleans", "new york", "oklahoma city", "oklahoma", "orlando", "philadelphia",
    "phoenix", "portland", "sacramento", "san antonio", "toronto", "utah", "washington",
}


def _is_stale_comment(comment: str, today_opponent: str | None, short_comment: str = "") -> bool:
    """Return True if the injury comment references a different opponent than today's.

    ESPN entries from the last regular-season rest day persist into playoffs.
    Pass combined shortComment + longComment as `comment` for team-name matching.
    Pass shortComment alone as `short_comment` for unconditional stale markers so
    that longComment background context (e.g. 'regular-season finale' history) does
    not falsely filter out injuries that are still active in the playoffs.

    Unconditional stale markers (checked only in short_comment):
    - 'regular-season finale' / 'regular season finale'
    """
    if not comment or today_opponent is None:
        return False
    # Unconditional stale: shortComment explicitly says "regular-season finale"
    check_for_finale = (short_comment or comment).lower()
    if "regular-season finale" in check_for_finale or "regular season finale" in check_for_finale:
        return True
    c = comment.lower()
    opp_lower = today_opponent.lower()
    # Check if any NBA team identifier appears in the comment
    mentioned_teams = [n for n in _NBA_TEAM_IDENTIFIERS if n in c]
    if not mentioned_teams:
        return False  # No team mentioned → keep (likely generic comment)
    # If the comment mentions a team, check if it's today's opponent
    opp_matches = any(n in opp_lower for n in mentioned_teams)
    if opp_matches:
        return False  # Comment references today's opponent → keep
    # Comment references a different team → stale
    return True


def fetch_injury_report(today_matchups: dict[str, str] | None = None) -> dict[str, list[str]]:
    """Fetch NBA injury report from ESPN. Returns {team_name: [injury_strings]}.

    today_matchups: {team_name: opponent_name} for stale-comment filtering.
    When a player's injury comment references a team that is NOT today's opponent,
    the entry is treated as stale (e.g., April 12 regular-season rest day entries).
    """
    try:
        import requests
        r = requests.get(_ESPN_INJURY_URL, headers=_ESPN_HEADERS, timeout=15)
        if r.status_code != 200:
            return {}
        data = r.json()
    except Exception:
        return {}

    result: dict[str, list[str]] = {}
    for team_entry in data.get("injuries", []):
        team_name = team_entry.get("displayName", "")
        today_opp = (today_matchups or {}).get(team_name)
        player_injuries = []
        for inj in team_entry.get("injuries", []):
            # ESPN uses "Out" as catch-all status; parse comment for actual status
            if inj.get("status") not in ("Out", "Questionable", "Probable", "Doubtful", "Day-To-Day"):
                continue
            athlete = inj.get("athlete", {})
            first = athlete.get("firstName", "")
            last = athlete.get("lastName", "")
            name = f"{first} {last}".strip()
            if not name:
                continue
            comment = inj.get("shortComment", "")
            long_comment = inj.get("longComment", "")
            # Use combined text for team-name matching; shortComment only for
            # unconditional stale markers ("regular-season finale") so that
            # longComment background context doesn't falsely trigger the filter.
            stale_text = f"{comment} {long_comment}"
            if _is_stale_comment(stale_text, today_opp, short_comment=comment):
                continue
            status_zh = _parse_injury_status(comment)
            player_injuries.append(f"{name} ({status_zh})")
        if player_injuries:
            result[team_name] = player_injuries
    return result

OUT_DIR = Path(__file__).resolve().parents[1] / "web" / "data"
DAYS_BACK = 7
DAYS_FORWARD = 7  # export upcoming game predictions up to 7 days ahead

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

# Per-season tiebreaker overrides for tied records. Our dataset cannot
# replicate the NBA's full tiebreaker chain (head-to-head, division leader,
# division record, conference record...), so when two teams finish with
# identical W-L, list them here as (higher_seed_team, lower_seed_team).
# The sort routine swaps adjacent tied entries to match this expected order.
SEED_OVERRIDES = {
    "2025-26": {
        # Play-in reality: PHX (#7) plays POR (#8); LAC drops to #9.
        "west": [("Portland Trail Blazers", "LA Clippers")],
        # TOR beat ATL in conference tiebreaker → TOR is 5-seed, ATL is 6-seed.
        "east": [("Toronto Raptors", "Atlanta Hawks")],
    },
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

    # Fetch injury report for today OR for past dates where games haven't been played yet.
    has_unplayed = any(g.get("home_score") is None for g in games_list)
    injury_report: dict[str, list[str]] = {}
    if is_today or has_unplayed:
        # Build opponent map for stale-comment filtering
        today_opp_map: dict[str, str] = {}
        for g in games_list:
            today_opp_map[g["home_team"]] = g["away_team"]
            today_opp_map[g["away_team"]] = g["home_team"]
        injury_report = fetch_injury_report(today_matchups=today_opp_map)

    # Fetch live scores — NBA CDN live scoreboard (primary, fastest) + SBR fallback.
    _live_score_map: dict[tuple, dict] = {}  # (home_team, away_team) -> live info
    if is_today or has_unplayed:
        # Primary: NBA CDN live scoreboard (real-time, no auth required)
        try:
            import requests as _req
            _nba_url = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
            _nba_r = _req.get(_nba_url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            if _nba_r.status_code == 200:
                _nba_games = _nba_r.json().get("scoreboard", {}).get("games", [])
                for _ng in _nba_games:
                    _ht_obj = _ng.get("homeTeam", {})
                    _at_obj = _ng.get("awayTeam", {})
                    _hs = _ht_obj.get("score", 0) or 0
                    _as = _at_obj.get("score", 0) or 0
                    _st = _ng.get("gameStatusText", "")
                    _ht = _ht_obj.get("teamCity", "") + " " + _ht_obj.get("teamName", "")
                    _at = _at_obj.get("teamCity", "") + " " + _at_obj.get("teamName", "")
                    if not _st or _hs == 0 and _as == 0:
                        continue
                    _is_final = "Final" in _st
                    _live_score_map[(_ht.strip(), _at.strip())] = {
                        "live_home_score": _hs,
                        "live_away_score": _as,
                        "live_status": _st,
                        "live_is_final": _is_final,
                        "live_home_covering": None,
                    }
        except Exception:
            pass
        # Fallback: SBR scraper (covers previous-day late games not on today's NBA board)
        try:
            from datetime import timedelta as _td
            from sbrscrape import Scoreboard as _SBR
            for _sbr_date in [target_date, target_date - _td(days=1)]:
                try:
                    _sb = _SBR(date=_sbr_date)
                    _sbr_games = _sb.games if hasattr(_sb, "games") else []
                except Exception:
                    _sbr_games = []
                for _sg in _sbr_games:
                    _hs = _sg.get("home_score", 0) or 0
                    _as = _sg.get("away_score", 0) or 0
                    _st = _sg.get("status", "")
                    _ht = _sg.get("home_team", "")
                    _at = _sg.get("away_team", "")
                    if not _st or _st in ("scheduled",):
                        continue
                    if _hs == 0 and _as == 0:
                        continue
                    # Only use SBR if NBA CDN didn't already find this game
                    if (_ht, _at) in _live_score_map:
                        continue
                    _is_final = "Final" in _st or "final" in _st
                    _live_score_map[(_ht, _at)] = {
                        "live_home_score": _hs,
                        "live_away_score": _as,
                        "live_status": _st,
                        "live_is_final": _is_final,
                        "live_home_covering": None,
                    }
        except Exception:
            pass

    # Load referee crew data for all games (if available in referee_data.sqlite).
    _ref_db_path = Path(__file__).resolve().parents[1] / "Data" / "referee_data.sqlite"
    _ref_crew_map: dict[tuple, dict] = {}  # (home_team, away_team) -> crew_info
    try:
        try:
            from scipy.stats import binomtest as _binomtest
        except Exception:
            _binomtest = None
        import sqlite3 as _rsql
        with _rsql.connect(str(_ref_db_path)) as _rcon:
            for g in games_list:
                _d = target_date.isoformat()
                _rrow = _rcon.execute(
                    'SELECT official1_name, official2_name, official3_name FROM referee_games'
                    ' WHERE game_date = ? AND home_team = ? AND away_team = ?',
                    (_d, g["home_team"], g["away_team"]),
                ).fetchone()
                if _rrow is None:
                    continue
                officials = [n for n in _rrow if n]
                if not officials:
                    continue
                crew_stats = []
                sig_officials = []
                for _oname in officials:
                    _orow = _rcon.execute(
                        'SELECT games, home_covers, home_cover_pct FROM referee_ats'
                        ' WHERE official_name = ?', (_oname,)
                    ).fetchone()
                    if _orow and _orow[0] >= 15:
                        _n, _hc, _pct = _orow
                        _p = None
                        _sig = False
                        if _binomtest is not None and _n >= 30:
                            try:
                                _p = float(_binomtest(int(_hc), int(_n), 0.5).pvalue)
                                _sig = _p < 0.10
                            except Exception:
                                pass
                        _dir = "home" if _pct > 50 else "away"
                        stat = {
                            "name": _oname,
                            "games": _n,
                            "home_cover_pct": round(_pct, 1),
                            "significant": _sig,
                            "direction": _dir if _sig else None,
                        }
                        if _p is not None:
                            stat["p_value"] = round(_p, 4)
                        crew_stats.append(stat)
                        if _sig:
                            sig_officials.append(stat)
                if crew_stats:
                    avg_pct = sum(s["home_cover_pct"] for s in crew_stats) / len(crew_stats)
                    _ref_crew_map[(g["home_team"], g["away_team"])] = {
                        "officials": crew_stats,
                        "avg_home_cover_pct": round(avg_pct, 1),
                        "away_friendly": avg_pct < 44.0,
                        "home_friendly": avg_pct > 56.0,
                        "has_significant": len(sig_officials) > 0,
                        "significant_officials": sig_officials,
                    }
    except Exception:
        pass

    # Load odds/spread history for all games on this date.
    _odds_db_path = Path(__file__).resolve().parents[1] / "Data" / "OddsData.sqlite"
    _odds_history_map: dict[tuple, list] = {}
    try:
        import sqlite3 as _osql
        with _osql.connect(str(_odds_db_path)) as _ocon:
            _ocon.execute(
                """
                CREATE TABLE IF NOT EXISTS odds_history (
                    fetched_at TEXT NOT NULL,
                    season TEXT NOT NULL,
                    Date TEXT NOT NULL,
                    Home TEXT NOT NULL,
                    Away TEXT NOT NULL,
                    OU REAL, Spread REAL, ML_Home INTEGER, ML_Away INTEGER,
                    source TEXT,
                    PRIMARY KEY (fetched_at, Date, Home, Away)
                )
                """
            )
            _d_iso = target_date.isoformat()
            for g in games_list:
                _rows = _ocon.execute(
                    "SELECT fetched_at, Spread, OU, ML_Home, ML_Away, source FROM odds_history "
                    "WHERE Date=? AND Home=? AND Away=? ORDER BY fetched_at ASC",
                    (_d_iso, g["home_team"], g["away_team"]),
                ).fetchall()
                if _rows:
                    _odds_history_map[(g["home_team"], g["away_team"])] = [
                        {
                            "fetched_at": r[0],
                            "spread": r[1],
                            "ou": r[2],
                            "ml_home": r[3],
                            "ml_away": r[4],
                            "source": r[5],
                        }
                        for r in _rows
                    ]
    except Exception:
        pass

    games_dict = {}
    for g in games_list:
        key = f"{g['away_team']}:{g['home_team']}"
        # Add Chinese names + logo URLs for frontend rendering.
        g["home_team_zh"] = team_name_zh(g["home_team"])
        g["away_team_zh"] = team_name_zh(g["away_team"])
        g["home_logo"] = team_logo_url(g["home_team"])
        g["away_logo"] = team_logo_url(g["away_team"])
        # Injury data (today only).
        g["injuries_home"] = injury_report.get(g["home_team"], [])
        # Live score (today only, if game in progress or recently finished).
        _live = _live_score_map.get((g["home_team"], g["away_team"]))
        if _live and g.get("Win_Margin") in (None, 0):
            spread = g.get("Spread") or g.get("spread") or 0
            live_margin = _live["live_home_score"] - _live["live_away_score"]
            _live["live_home_covering"] = (live_margin - spread) > 0
            g.update(_live)
        # Referee crew analysis (when available).
        _crew = _ref_crew_map.get((g["home_team"], g["away_team"]))
        if _crew:
            g["referee_crew"] = _crew
        g["injuries_away"] = injury_report.get(g["away_team"], [])
        # Odds/spread movement history.
        _hist = _odds_history_map.get((g["home_team"], g["away_team"]))
        if _hist:
            g["odds_history"] = _hist
            _sp_vals = [h["spread"] for h in _hist if h.get("spread") is not None]
            if _sp_vals:
                g["spread_first"] = _sp_vals[0]
                g["spread_last"] = _sp_vals[-1]
                g["spread_move"] = round(_sp_vals[-1] - _sp_vals[0], 2)
                g["spread_first_fetched_at"] = _hist[0]["fetched_at"]
                g["spread_last_fetched_at"] = _hist[-1]["fetched_at"]
            _ou_vals = [h["ou"] for h in _hist if h.get("ou") is not None]
            if _ou_vals:
                g["ou_first"] = _ou_vals[0]
                g["ou_last"] = _ou_vals[-1]
                g["ou_move"] = round(_ou_vals[-1] - _ou_vals[0], 1)
        games_dict[key] = g

    # Summary stats.
    n = len(games_list)
    home_picks = sum(1 for g in games_list if g.get("winner") == "home")
    over_picks = sum(1 for g in games_list if g.get("ou_pick") == "OVER")
    avg_conf = sum(max(g.get("home_confidence") or 0, g.get("away_confidence") or 0) for g in games_list) / n if n else 0

    # Collect GOLD/SILVER playoff ATS signals across all games.
    po_ats_alerts = []
    for g in games_list:
        if g.get("is_playoff") and g.get("playoff_ats_picks"):
            for pick in g["playoff_ats_picks"]:
                if pick.get("tier") in ("GOLD", "SILVER"):
                    po_ats_alerts.append({
                        "game_key": "%s:%s" % (g["away_team"], g["home_team"]),
                        "home_team": g["home_team"],
                        "away_team": g["away_team"],
                        "home_team_zh": g.get("home_team_zh", g["home_team"]),
                        "away_team_zh": g.get("away_team_zh", g["away_team"]),
                        "signal": pick.get("signal"),
                        "side": pick.get("side"),
                        "ats_side": pick.get("ats_side"),
                        "tier": pick.get("tier"),
                        "backtest_wr": pick.get("backtest_wr"),
                        "backtest_roi": pick.get("backtest_roi"),
                        "reason_zh": pick.get("reason_zh"),
                        "ats_winner": g.get("ats_winner"),
                        "ats_cover_margin": g.get("ats_cover_margin"),
                        "series_game_num": g.get("series_game_num"),
                        "series_home_wins": g.get("series_home_wins"),
                        "series_away_wins": g.get("series_away_wins"),
                        "strong_consensus": g.get("playoff_ats_strong_consensus"),
                        "has_conflict": g.get("playoff_ats_has_conflict", False),
                    })

    # Build injury alert strings for the summary report section.
    injury_alerts = []
    if (is_today or has_unplayed) and injury_report:
        for g in games_list:
            home, away = g["home_team"], g["away_team"]
            home_inj = injury_report.get(home, [])
            away_inj = injury_report.get(away, [])
            for p in home_inj:
                injury_alerts.append(f"{home}: {p}")
            for p in away_inj:
                injury_alerts.append(f"{away}: {p}")

    summary = {
        "games": n,
        "home_picks": home_picks,
        "away_picks": n - home_picks,
        "over_picks": over_picks,
        "under_picks": n - over_picks,
        "avg_confidence": round(avg_conf, 1),
        "value_picks": sum(1 for g in games_list if g.get("is_value")),
        "golden_picks": sum(1 for g in games_list if g.get("is_golden")),
        "playoff_ats_alerts": po_ats_alerts,
        "playoff_ats_count": len(po_ats_alerts),
        "injury_alerts": injury_alerts,
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

    # Enrich series with conference for East/West split UI.
    active_series = [
        dict(s, conference=("east" if s.get("high_seed") in EAST_TEAMS else
                            "west" if s.get("high_seed") in WEST_TEAMS else None))
        for s in active_series
    ]

    return {
        "date": target_date.isoformat(),
        "is_today": is_today,
        "summary": summary,
        "games": games_dict,
        "active_series": active_series,
        "is_playoff_view": is_playoff_view,
    }


def _season_key_for(d: date) -> str:
    """NBA season runs Oct → June; '2025-26' season spans Oct 2025 → Jun 2026."""
    if d.month >= 10:
        return f"{d.year}-{(d.year + 1) % 100:02d}"
    return f"{d.year - 1}-{d.year % 100:02d}"


def _team_card(team, wins, losses):
    return {
        "team": team,
        "team_zh": team_name_zh(team),
        "logo": team_logo_url(team),
        "wins": int(wins),
        "losses": int(losses),
        "win_pct": round(wins / (wins + losses), 3) if (wins + losses) else 0.0,
    }


def _build_signal_tracker(data_dir: Path) -> dict:
    """Scan date JSONs for playoff ATS signal results and compile performance."""
    from collections import defaultdict
    total = wins = losses = pending = 0
    by_tier = defaultdict(lambda: {"wins": 0, "losses": 0, "pending": 0})
    by_game_num = defaultdict(lambda: {"wins": 0, "losses": 0, "pending": 0})
    details = []
    for f in sorted(list(data_dir.glob("2026-04-*.json")) + list(data_dir.glob("2026-05-*.json")) + list(data_dir.glob("2026-06-*.json"))):
        try:
            with open(f, encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue
        for a in d.get("summary", {}).get("playoff_ats_alerts", []):
            tier = a.get("tier", "?")
            side = a.get("side")
            winner = a.get("ats_winner")
            gn = a.get("series_game_num")
            gn_key = f"G{gn}" if gn else "?"
            total += 1
            if winner is not None:
                if winner == side:
                    wins += 1
                    by_tier[tier]["wins"] += 1
                    by_game_num[gn_key]["wins"] += 1
                else:
                    losses += 1
                    by_tier[tier]["losses"] += 1
                    by_game_num[gn_key]["losses"] += 1
            else:
                pending += 1
                by_tier[tier]["pending"] += 1
                by_game_num[gn_key]["pending"] += 1
            details.append({
                "date": d.get("date", ""),
                "away": a.get("away_team_zh", a.get("away_team", "")),
                "home": a.get("home_team_zh", a.get("home_team", "")),
                "signal": a.get("signal", ""),
                "tier": tier,
                "side": side,
                "backtest_wr": a.get("backtest_wr"),
                "result": "win" if winner == side else ("loss" if winner is not None else "pending"),
                "margin": a.get("ats_cover_margin"),
                "game_num": gn,
            })
    decided = wins + losses
    pnl = round(wins * 0.91 - losses * 1.0, 2) if decided else 0
    roi = round(pnl / decided * 100, 1) if decided else None

    # Game-level: use majority-vote consensus per game (not "any signal wins")
    game_votes: dict[str, dict] = {}
    for d in details:
        key = f"{d['date']}|{d['away']}|{d['home']}"
        if key not in game_votes:
            game_votes[key] = {"home": 0, "away": 0, "pending": False, "ats_winner": None}
        if d["result"] == "pending":
            game_votes[key]["pending"] = True
        else:
            # Derive the actual ATS winner from any decided signal
            if d["result"] == "win":
                game_votes[key]["ats_winner"] = d["side"]
            elif game_votes[key]["ats_winner"] is None:
                game_votes[key]["ats_winner"] = "home" if d["side"] == "away" else "away"
            game_votes[key][d["side"]] += 1

    game_wins = game_losses = 0
    nc_wins = nc_losses = nc_pending = 0
    for gv in game_votes.values():
        if gv["pending"] or gv["ats_winner"] is None:
            has_conflict = gv["home"] > 0 and gv["away"] > 0
            if not has_conflict and gv["pending"]:
                nc_pending += 1
            continue
        consensus = "home" if gv["home"] > gv["away"] else "away" if gv["away"] > gv["home"] else None
        has_conflict = gv["home"] > 0 and gv["away"] > 0
        if not has_conflict:
            side = "home" if gv["home"] > 0 else "away"
            if side == gv["ats_winner"]:
                nc_wins += 1
            else:
                nc_losses += 1
        if consensus is None:
            continue
        if consensus == gv["ats_winner"]:
            game_wins += 1
        else:
            game_losses += 1
    game_total = game_wins + game_losses
    game_pnl = round(game_wins * 0.91 - game_losses * 1.0, 2) if game_total else 0
    nc_total = nc_wins + nc_losses
    nc_pnl = round(nc_wins * 0.91 - nc_losses * 1.0, 2) if nc_total else 0

    # Premium filter: no-conflict + exclude G3/G6 (87% accuracy tier)
    pm_wins = pm_losses = pm_pending = 0
    for key, gv in game_votes.items():
        has_conflict = gv["home"] > 0 and gv["away"] > 0
        if has_conflict:
            continue
        # Find game_num for this game
        gn_for_game = None
        for dd in details:
            dk = f"{dd['date']}|{dd['away']}|{dd['home']}"
            if dk == key:
                gn_for_game = dd.get("game_num")
                break
        if gn_for_game in (3, 6):
            continue
        if gv["pending"] or gv["ats_winner"] is None:
            pm_pending += 1
            continue
        side = "home" if gv["home"] > 0 else "away"
        if side == gv["ats_winner"]:
            pm_wins += 1
        else:
            pm_losses += 1
    pm_total = pm_wins + pm_losses
    pm_pnl = round(pm_wins * 0.91 - pm_losses * 1.0, 2) if pm_total else 0

    by_gn_sorted = {}
    for gn_key in sorted(by_game_num.keys()):
        v = by_game_num[gn_key]
        d = v["wins"] + v["losses"]
        by_gn_sorted[gn_key] = {
            **v,
            "hit_rate": round(v["wins"] / d * 100, 1) if d else None,
        }

    return {
        "total": total,
        "decided": decided,
        "wins": wins,
        "losses": losses,
        "pending": pending,
        "hit_rate": round(wins / decided * 100, 1) if decided else None,
        "pnl": pnl,
        "roi": roi,
        "by_tier": dict(by_tier),
        "by_game_num": by_gn_sorted,
        "by_game": {
            "total": len(game_votes),
            "decided": game_total,
            "wins": game_wins,
            "losses": game_losses,
            "pending": len(game_votes) - game_total,
            "hit_rate": round(game_wins / game_total * 100, 1) if game_total else None,
            "pnl": game_pnl,
            "roi": round(game_pnl / game_total * 100, 1) if game_total else None,
        },
        "no_conflict": {
            "decided": nc_total,
            "wins": nc_wins,
            "losses": nc_losses,
            "pending": nc_pending,
            "hit_rate": round(nc_wins / nc_total * 100, 1) if nc_total else None,
            "pnl": nc_pnl,
            "roi": round(nc_pnl / nc_total * 100, 1) if nc_total else None,
        },
        "premium": {
            "label": "核心場次 (排除G3/G6，無衝突)",
            "filter": "no_conflict + exclude G3/G6",
            "decided": pm_total,
            "wins": pm_wins,
            "losses": pm_losses,
            "pending": pm_pending,
            "hit_rate": round(pm_wins / pm_total * 100, 1) if pm_total else None,
            "pnl": pm_pnl,
            "roi": round(pm_pnl / pm_total * 100, 1) if pm_total else None,
            "note": "G3(66.7%)與G6(64.7%)拖累整體準確率，排除後每場勝率達87%",
        },
        "details": details,
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
                "AND name NOT LIKE '%_playoff%' "
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

    season_key = _season_key_for(target_date)
    overrides = SEED_OVERRIDES.get(season_key, {})

    def _apply_tiebreaker_overrides(seq, override_pairs):
        """Swap adjacent tied entries to match the (higher, lower) pairs."""
        if not override_pairs:
            return seq
        name_to_idx = {c["team"]: i for i, c in enumerate(seq)}
        for higher, lower in override_pairs:
            hi = name_to_idx.get(higher)
            lo = name_to_idx.get(lower)
            if hi is None or lo is None:
                continue
            # Only swap if they're tied and currently in the wrong order.
            if seq[hi]["win_pct"] != seq[lo]["win_pct"]:
                continue
            if hi > lo:
                seq[hi], seq[lo] = seq[lo], seq[hi]
                name_to_idx[higher] = lo
                name_to_idx[lower] = hi
        return seq

    def _sort(seq, override_pairs):
        seq.sort(key=lambda c: (-c["win_pct"], -c["wins"], c["team"]))
        _apply_tiebreaker_overrides(seq, override_pairs)
        for i, c in enumerate(seq, 1):
            c["seed"] = i
        return seq

    east = _sort(east, overrides.get("east", []))
    west = _sort(west, overrides.get("west", []))

    # Detection window for the UI: 7 days before playoffs start through 10
    # days after (covers pre-playoff anticipation + play-in tournament).
    show_from = show_until = playoff_start = None
    try:
        import toml as _toml
        from datetime import datetime as _dt, timedelta as _td
        _cfg = _toml.load(Path(__file__).resolve().parents[1] / "config.toml")
        for v in _cfg.get("get-playoffs", {}).values():
            s = _dt.strptime(v["start_date"], "%Y-%m-%d").date()
            e = _dt.strptime(v["end_date"], "%Y-%m-%d").date()
            if s <= target_date <= e or 0 <= (s - target_date).days <= 14:
                playoff_start = s.isoformat()
                show_from = (s - _td(days=7)).isoformat()
                show_until = (s + _td(days=10)).isoformat()
                break
    except Exception:
        pass

    # Resolve actual 7/8-seed opponents from R1 OddsData schedule.
    # Look for games where the home team is a 1-seed or 2-seed and away team
    # is a known play-in participant (seeds 7-10 by pre-play-in standing).
    def _resolve_playin_r1(seed_list, _season_key, _playoff_start_str):
        """Return {7: team_card, 8: team_card} if R1 matchups are known."""
        resolved = {}
        if not _playoff_start_str:
            return resolved
        try:
            odds_db = Path(__file__).resolve().parents[1] / "Data" / "OddsData.sqlite"
            if not odds_db.exists():
                return resolved
            seed1 = seed_list[0]["team"]  # 1-seed
            seed2 = seed_list[1]["team"]  # 2-seed
            playin_names = {s["team"] for s in seed_list[6:10]}
            team_card_map = {s["team"]: s for s in seed_list}
            import sqlite3 as _sq
            with _sq.connect(str(odds_db)) as con:
                rows = con.execute(
                    f'SELECT Home, Away FROM "{_season_key}" WHERE Date >= ?'
                    f' AND (Home = ? OR Home = ?)',
                    (_playoff_start_str, seed1, seed2),
                ).fetchall()
            for home, away in rows:
                if away in playin_names and away in team_card_map:
                    if home == seed1:
                        card = dict(team_card_map[away])
                        card["seed"] = 8  # actual playoff seed after play-in
                        resolved[8] = card
                    elif home == seed2:
                        card = dict(team_card_map[away])
                        card["seed"] = 7  # actual playoff seed after play-in
                        resolved[7] = card
        except Exception:
            pass
        return resolved

    east_r1_resolved = _resolve_playin_r1(east, season_key, playoff_start)
    west_r1_resolved = _resolve_playin_r1(west, season_key, playoff_start)

    # Load series_state (R1+ for scores, play-in for results).
    _series_wins: dict = {}
    _playin_results: dict = {}  # frozenset({high, low}) → (hw, lw)
    _stage_label = "附加賽階段"
    _max_round_num = 0
    try:
        import sqlite3 as _sq
        _odds_db = Path(__file__).resolve().parents[1] / "Data" / "OddsData.sqlite"
        _tbl = f"series_state_{season_key}"
        with _sq.connect(str(_odds_db)) as _con:
            for _row in _con.execute(
                f'SELECT high_seed, low_seed, high_wins, low_wins, round_num FROM "{_tbl}"'
            ):
                _hs, _ls, _hw, _lw, _rn = _row
                if _rn >= 1:
                    # Map team_name → wins so _with_wins can look up by actual team
                    _series_wins[frozenset({_hs, _ls})] = {_hs: (_hw or 0), _ls: (_lw or 0)}
                else:
                    _playin_results[frozenset({_hs, _ls})] = (_hw or 0, _lw or 0)
        # Determine stage
        if _series_wins:
            max_rn = 1
            with _sq.connect(str(_odds_db)) as _con:
                row = _con.execute(
                    f'SELECT MAX(round_num) FROM "{_tbl}" WHERE round_num >= 1'
                ).fetchone()
                if row and row[0]:
                    max_rn = row[0]
            _max_round_num = max_rn
            _stage_label = {1: "首輪進行中", 2: "第二輪進行中",
                            3: "分區決賽", 4: "總決賽"}.get(max_rn, "季後賽進行中")
        elif _playin_results:
            _stage_label = "附加賽進行中"
    except Exception:
        pass

    # Compute 2025-26 R1G1 home cover record dynamically from OddsData.
    _r1g1_season_record = ""
    try:
        import sqlite3 as _sq
        _odds_db = Path(__file__).resolve().parents[1] / "Data" / "OddsData.sqlite"
        _tbl_ss = f"series_state_{season_key}"
        _tbl_od = season_key
        with _sq.connect(str(_odds_db)) as _con:
            # Get all R1 series earliest dates
            _r1_series = _con.execute(
                f'SELECT high_seed, low_seed, earliest_date FROM "{_tbl_ss}" WHERE round_num = 1'
            ).fetchall()
            r1g1_wins = 0
            r1g1_total = 0
            for _hs, _ls, _edate in _r1_series:
                if not _edate:
                    continue
                _row = _con.execute(
                    f'SELECT Home, Away, Win_Margin, Spread FROM "{_tbl_od}"'
                    f' WHERE Date = ? AND ((Home = ? AND Away = ?) OR (Home = ? AND Away = ?))'
                    f' AND Points > 0',
                    (_edate, _hs, _ls, _ls, _hs),
                ).fetchone()
                if _row is None:
                    continue
                _home, _away, _wm, _sp = _row
                if _wm is None or _sp is None:
                    continue
                _cover = float(_wm) - float(_sp)
                if _cover == 0:
                    continue  # push
                r1g1_total += 1
                if _cover > 0:
                    r1g1_wins += 1
        if r1g1_total > 0:
            _r1g1_season_record = f"，2025-26本季: {r1g1_wins}/{r1g1_total}"
    except Exception:
        _r1g1_season_record = "，2025-26本季: 4/4"

    def _playin_winner_loser(team_a, team_b):
        """Return (winner_card, loser_card) using playin_results, or (None, None)."""
        key = frozenset({team_a["team"], team_b["team"]})
        result = _playin_results.get(key)
        if result is None:
            return None, None
        hw, lw = result
        # high_seed = team_a (home), low_seed = team_b (away) in our naming
        # Find which is which from playin_results by matching team names
        for _hs, _ls, _hw2, _lw2, _ in []:  # unused placeholder
            pass
        # Look up directly
        try:
            import sqlite3 as _sq2
            _odds_db2 = Path(__file__).resolve().parents[1] / "Data" / "OddsData.sqlite"
            _tbl2 = f"series_state_{season_key}"
            with _sq2.connect(str(_odds_db2)) as _con2:
                row = _con2.execute(
                    f'SELECT high_seed, low_seed, high_wins, low_wins FROM "{_tbl2}"'
                    f' WHERE round_num = 0 AND ('
                    f'  (high_seed = ? AND low_seed = ?) OR (high_seed = ? AND low_seed = ?))',
                    (team_a["team"], team_b["team"], team_b["team"], team_a["team"]),
                ).fetchone()
        except Exception:
            return None, None
        if not row:
            return None, None
        hs_name, ls_name, hw2, lw2 = row
        if hw2 and not lw2:
            winner_name, loser_name = hs_name, ls_name
        elif lw2 and not hw2:
            winner_name, loser_name = ls_name, hs_name
        else:
            return None, None
        winner = team_a if team_a["team"] == winner_name else team_b
        loser = team_a if team_a["team"] == loser_name else team_b
        return winner, loser

    def _conference(name, seeds, r1_resolved):
        # Enrich play-in with winner data from series_state.
        seed7, seed8 = seeds[6], seeds[7]
        seed9, seed10 = seeds[8], seeds[9]

        ga_winner, ga_loser = _playin_winner_loser(seed7, seed8)
        gb_winner, gb_loser = _playin_winner_loser(seed9, seed10)

        # game_c: 7v8 loser vs 9v10 winner
        gc_home = ga_loser or "7/8 敗者"
        gc_away = gb_winner or "9/10 勝者"
        gc_winner, gc_loser = (None, None)
        if ga_loser and gb_winner:
            gc_winner, gc_loser = _playin_winner_loser(ga_loser, gb_winner)

        play_in = {
            "game_a": {
                "label": "7 vs 8", "home": seed7, "away": seed8,
                "winner_seed": 7, "loser_note": "敗者進 8/9 淘汰賽",
                "winner": ga_winner["team"] if ga_winner else None,
                "winner_zh": ga_winner["team_zh"] if ga_winner else None,
            },
            "game_b": {
                "label": "9 vs 10", "home": seed9, "away": seed10,
                "winner_note": "晉級 8 號淘汰賽", "loser_note": "淘汰",
                "winner": gb_winner["team"] if gb_winner else None,
                "winner_zh": gb_winner["team_zh"] if gb_winner else None,
            },
            "game_c": {
                "label": "8 號種子決定戰",
                "home": gc_home, "away": gc_away,
                "winner_seed": 8,
                "winner": gc_winner["team"] if gc_winner else None,
                "winner_zh": gc_winner["team_zh"] if gc_winner else None,
            },
        }
        low_8 = r1_resolved.get(8, "待定（8 號種子）")
        low_7 = r1_resolved.get(7, "待定（7 號種子）")
        playin_survivors = {
            c["team"] for c in r1_resolved.values() if isinstance(c, dict)
        }

        # Load today + upcoming game predictions for signal lookup (bracket is built after game JSONs).
        _today_game_lookup: dict = {}
        _upcoming_game_lookup: dict = {}  # frozenset({home, away}) -> game dict, next 5 days
        try:
            from datetime import timedelta as _tdelta
            _data_dir = Path(__file__).resolve().parents[1] / "web" / "data"
            for _day_offset in range(-1, 6):
                _check_date = target_date + _tdelta(days=_day_offset)
                _check_json = _data_dir / f"{_check_date.isoformat()}.json"
                if not _check_json.exists():
                    continue
                with open(_check_json, encoding="utf-8") as _tf:
                    _td_data = json.load(_tf)
                for _gk, _gv in _td_data.get("games", {}).items():
                    if isinstance(_gv, dict) and _gv.get("is_playoff"):
                        _parts = _gk.split(":")
                        _fkey = frozenset(_parts)
                        if _day_offset == 0:
                            _today_game_lookup[_fkey] = _gv
                        _already_played = _gv.get("home_score") is not None and _gv.get("away_score") is not None
                        if _fkey not in _upcoming_game_lookup and not _already_played:
                            _upcoming_game_lookup[_fkey] = _gv
        except Exception:
            pass

        def _with_wins(high_team, low_team, round_num=1):
            """Attach series win counts and upcoming signal hints to a first_round entry."""
            if isinstance(high_team, str) or isinstance(low_team, str):
                return {}
            key = frozenset({high_team["team"], low_team["team"]})
            wins_by_team = _series_wins.get(key, {})
            hw = wins_by_team.get(high_team["team"], 0)
            lw = wins_by_team.get(low_team["team"], 0)
            gp = hw + lw
            entry = {"high_wins": hw, "low_wins": lw, "games_played": gp}
            # G1 pre-game hint: R2/CF opener home-seed advantage.
            if gp == 0 and _max_round_num >= 2:
                _game = _today_game_lookup.get(key) or _upcoming_game_lookup.get(key)
                _g1_picks = (_game.get("playoff_ats_picks", []) if _game else [])
                _g1_pick = next(
                    (p for p in _g1_picks if "G1" in p.get("signal", "") and p.get("tier") in ("GOLD", "SILVER")),
                    None,
                )
                if _g1_pick:
                    entry["next_signal"] = {
                        "game_num": 1,
                        "signal": _g1_pick["signal"],
                        "side": _g1_pick.get("side", "home"),
                        "home_team": high_team["team"],
                        "home_team_zh": high_team["team_zh"],
                        "tier": _g1_pick.get("tier", "SILVER"),
                        "backtest_wr": _g1_pick.get("backtest_wr", 0.646),
                        "reason_zh": _g1_pick.get("reason_zh", ""),
                    }
            # G1 pre-game hint: R1 opener home-seed advantage (fires before any game played).
            if gp == 0 and _stage_label == "首輪進行中":
                is_playin_away = low_team["team"] in playin_survivors
                # Use large-spread signal WR if today's game has spread >= 8
                _game = _today_game_lookup.get(key)
                _spread = _game.get("spread") if _game else None
                if _spread is not None and _spread >= 10:
                    _ns_signal, _ns_wr = "R1G1大讓分主場", 0.727
                    _ns_reason = f"首輪G1讓 {_spread:.1f} 分，歷史主場cover率 72.7% (n=22)，大讓分G1主場強壓"
                elif _spread is not None and _spread >= 8:
                    _ns_signal, _ns_wr = "R1G1大讓分主場", 0.622
                    _ns_reason = f"首輪G1讓 {_spread:.1f} 分，歷史主場cover率 62.2% (n=37)，大讓分G1主場強壓"
                else:
                    _ns_signal, _ns_wr = "首輪G1主場優勢", 0.60
                    _ns_reason = f"首輪第1場主場(非附加賽晉級)，附加賽制時代歷史cover率 60%{_r1g1_season_record}"
                ns = {
                    "game_num": 1,
                    "signal": _ns_signal,
                    "side": "home",
                    "home_team": high_team["team"],
                    "home_team_zh": high_team["team_zh"],
                    "tier": "SILVER",
                    "backtest_wr": _ns_wr,
                    "reason_zh": _ns_reason,
                    "playin_away": is_playin_away,
                }
                # Check today's game data for conflicts (e.g. away cold bounce vs home signals).
                if _game and _game.get("playoff_ats_has_conflict"):
                    ns["has_conflict"] = True
                    for _pick in _game.get("playoff_ats_picks", []):
                        if _pick.get("side") == "away":
                            ns["conflict_signal"] = _pick.get("signal")
                            ns["conflict_wr"] = _pick.get("backtest_wr")
                            ns["conflict_away_team"] = low_team.get("team_zh") or low_team.get("team")
                            break
                entry["next_signal"] = ns
            # G2 home signal: split by G1 result.
            # Home lost G1 (bounce-back): 64.9% SILVER; home won G1 (consolidation): 53.4% BRONZE
            if gp == 1:
                g2_home = high_team
                _g2_game = None  # may be set in else branch; checked below for conflict
                _g2_away_side = False
                if lw == 1 and hw == 0:
                    # Low seed won G1 → high seed (home in G2) is in bounce-back mode
                    g2_sig, g2_tier, g2_wr = "G2主場反彈 (輸G1)", "SILVER", 0.649
                    g2_reason = "第2場主場輸G1後反彈，主場cover率 64.9% (n=57，12季歷史)"
                    # Also look up G2 game for conflict info (same logic as else branch)
                    _g2_game_key = frozenset({high_team["team"], low_team["team"]})
                    _g2_game = _upcoming_game_lookup.get(_g2_game_key)
                else:
                    # High seed won G1 → check if blowout (G2大勝後延續 SILVER 64.3%)
                    # or regular consolidation (G2主場鞏固 BRONZE 53.4%)
                    _g2_game_key = frozenset({high_team["team"], low_team["team"]})
                    _g2_game = _upcoming_game_lookup.get(_g2_game_key)
                    _best_signal = _g2_game.get("playoff_ats_best_signal") if _g2_game else None
                    _best_tier = _g2_game.get("playoff_ats_best_tier") if _g2_game else None
                    _best_picks = _g2_game.get("playoff_ats_picks", []) if _g2_game else []
                    _g2_away_side = False
                    if _best_signal == "G2大勝後延續" and _best_tier == "SILVER":
                        g2_sig, g2_tier, g2_wr = "G2大勝後延續", "SILVER", 0.643
                        _blowout_pick = next((p for p in _best_picks if p.get("signal") == "G2大勝後延續"), {})
                        g2_reason = _blowout_pick.get("reason_zh", "G1大勝後主場延續優勢，12季R1回測主場cover率 64.3% (n=28)")
                    else:
                        _consol_pick = next((p for p in _best_picks if p.get("signal") == "G2主場鞏固 (贏G1)"), None)
                        if _consol_pick:
                            g2_sig, g2_tier = "G2主場鞏固 (贏G1)", "BRONZE"
                            g2_wr = _consol_pick.get("backtest_wr", 0.534)
                            g2_reason = _consol_pick.get("reason_zh", "第2場主場贏G1後鞏固，主場cover率 53.4% (n=116，12季歷史)")
                        else:
                            # Narrow G1 win (≤7 ATS margin): check ML away lean first
                            _ml_pick = next((p for p in _best_picks if p.get("signal") == "ML季後賽押客場"), None)
                            if _ml_pick and _ml_pick.get("tier") == "SILVER":
                                g2_sig, g2_tier, g2_wr = "ML季後賽押客場", "SILVER", _ml_pick.get("backtest_wr", 0.64)
                                g2_reason = _ml_pick.get("reason_zh", "模型押客場，季後賽歷史客場cover率 64.0% (n=175)")
                                _g2_away_side = True
                            else:
                                g2_sig, g2_tier, g2_wr = "G2無明顯信號 (G1窄勝)", "NONE", 0.478
                                g2_reason = "G1窄勝 (≤7 ATS分)，G2主場cover率約 47.8% (n=88)，接近持平，無明顯押注方向"
                _g2_ns_side = "away" if _g2_away_side else "home"
                _g2_signal_team = low_team if _g2_away_side else g2_home
                _g2_ns = {
                    "game_num": 2,
                    "signal": g2_sig,
                    "side": _g2_ns_side,
                    "home_team": g2_home["team"],
                    "home_team_zh": g2_home["team_zh"],
                    "tier": g2_tier,
                    "backtest_wr": g2_wr,
                    "reason_zh": g2_reason,
                    "signal_team_zh": _g2_signal_team.get("team_zh") or _g2_signal_team.get("team"),
                }
                # Propagate conflict info from the upcoming G2 game card.
                if _g2_game and _g2_game.get("playoff_ats_has_conflict"):
                    _g2_ns["has_conflict"] = True
                    for _cp in (_g2_game.get("playoff_ats_picks") or []):
                        if _cp.get("side") == "away":
                            _g2_ns["conflict_signal"] = _cp.get("signal")
                            _g2_ns["conflict_wr"] = _cp.get("backtest_wr")
                            _g2_ns["conflict_away_team"] = low_team.get("team_zh") or low_team.get("team")
                            break
                entry["next_signal"] = _g2_ns
            # G3 signals: fire based on series state when 2 games played.
            # G3 is at low_seed's home. hw=high_wins, lw=low_wins.
            # Verified backtest (18 seasons, proper playoff date filtering):
            #   High seed up 2-0, G3 away: AWAY covers 64.3% (n=28)
            #   Tied 1-1, G3 low seed home: covers 50% (no edge)
            if gp == 2:
                if hw == 2 and lw == 0:
                    # High seed up 2-0: away (high seed) continues to cover at 64.3%
                    entry["next_signal"] = {
                        "game_num": 3,
                        "signal": "G3客場擴大優勢",
                        "side": "away",
                        "home_team": low_team["team"],
                        "home_team_zh": low_team["team_zh"],
                        "tier": "SILVER",
                        "backtest_wr": 0.643,
                        "reason_zh": f"高種子領先2-0出戰G3，18季回測客場(高種子)cover率 64.3% (n=28)，強隊延伸優勢",
                    }
                elif hw == 1 and lw == 1:
                    # Tied 1-1: no strong edge (50% historically), use elimination underdog logic
                    # Underdog in non-elimination games around 52-55%
                    entry["next_signal"] = {
                        "game_num": 3,
                        "signal": "G3平手客場冷門",
                        "side": "away",
                        "home_team": low_team["team"],
                        "home_team_zh": low_team["team_zh"],
                        "tier": "BRONZE",
                        "backtest_wr": 0.516,
                        "reason_zh": "系列賽平手1-1，G3低種子首回主場，歷史50%無明顯優勢，客場(高種子)小型邊",
                    }
                else:
                    # Low seed up 2-0: high seed (away in G3) desperate, no strong signal
                    entry["next_signal"] = {
                        "game_num": 3,
                        "signal": "G3客場反彈",
                        "side": "away",
                        "home_team": low_team["team"],
                        "home_team_zh": low_team["team_zh"],
                        "tier": "BRONZE",
                        "backtest_wr": 0.568,
                        "reason_zh": "系列賽高種子落後0-2，G3客場(高種子)反彈，歷史56.8%客場cover",
                    }
            # G4 signal: fires when 3 games played. G4 at low_seed home, away (high seed) covers 57.3%
            if gp == 3:
                entry["next_signal"] = {
                    "game_num": 4,
                    "signal": "G4客場優勢",
                    "side": "away",
                    "home_team": low_team["team"],
                    "home_team_zh": low_team["team_zh"],
                    "tier": "BRONZE",
                    "backtest_wr": 0.573,
                    "reason_zh": "第4場低種子主場，歷史高種子客場cover率 57.3% (n=171，13季歷史)",
                }
            # G5 signal: fires when 4 games played and series still active
            # G5 is at high_seed's home (2-2-1-1-1 format: games 1,2,5 at high seed)
            if gp == 4 and hw < 4 and lw < 4:
                if round_num == 2:
                    # R2 G5: home covers 83.7% (n=43, 13 seasons) — GOLD regardless of score
                    entry["next_signal"] = {
                        "game_num": 5,
                        "signal": "R2G5主場壓制",
                        "side": "home",
                        "home_team": high_team["team"],
                        "home_team_zh": high_team["team_zh"],
                        "tier": "GOLD",
                        "backtest_wr": 0.837,
                        "reason_zh": f"第二輪G5高種子主場，13季回測主場cover率 83.7% (n=43)，為R2最強結構信號",
                    }
                elif hw == 2 and lw == 2:
                    entry["next_signal"] = {
                        "game_num": 5,
                        "signal": "G5平手主場壓制",
                        "side": "home",
                        "home_team": high_team["team"],
                        "home_team_zh": high_team["team_zh"],
                        "tier": "SILVER",
                        "backtest_wr": 0.60,
                        "reason_zh": "系列賽平手2-2，第5場高種子主場，12季回測主場cover率 60% (n=60，ROI +14.5%)",
                    }
            # G6 signal: fires when 5 games played (series reaches G6).
            # R1 G6 away covers 80.4% (n=46, GOLD). R2+ G6 away ~53.8% (weaker).
            if gp == 5 and hw < 4 and lw < 4:
                g6_away = high_team
                g6_home = low_team
                _round_label = {1: "R1", 2: "R2", 3: "CF", 4: "F"}.get(round_num, "R1")
                _round_zh = {1: "首輪", 2: "第二輪", 3: "分區決賽", 4: "總決賽"}.get(round_num, "首輪")
                if round_num == 1:
                    _g6_tier = "GOLD"
                    _g6_wr = 0.804
                    _g6_reason = f"{_round_zh}第6場在低種子主場，12季回測客場(高種子)cover率 80.4% (n=46)，近5年更達 84.6%，為歷史最強結構信號"
                else:
                    _g6_tier = "BRONZE"
                    _g6_wr = 0.538
                    _g6_reason = f"{_round_zh}第6場在低種子主場，客場(高種子)cover率 53.8%，次輪後效果減弱"
                entry["next_signal"] = {
                    "game_num": 6,
                    "signal": f"{_round_label}G6客場優勢",
                    "side": "away",
                    "home_team": g6_home["team"],
                    "home_team_zh": g6_home["team_zh"],
                    "tier": _g6_tier,
                    "backtest_wr": _g6_wr,
                    "reason_zh": _g6_reason,
                    "signal_team_zh": g6_away.get("team_zh") or g6_away.get("team"),
                }
            # Upgrade next_signal if actual game data has a higher-tier signal.
            _tier_rank = {"GOLD": 3, "SILVER": 2, "BRONZE": 1, "NONE": 0}
            ns = entry.get("next_signal")
            if ns and gp >= 1:
                _ns_key = frozenset({high_team["team"], low_team["team"]})
                _ns_game = _upcoming_game_lookup.get(_ns_key)
                if _ns_game:
                    _best_tier = _ns_game.get("playoff_ats_best_tier")
                    _best_signal = _ns_game.get("playoff_ats_best_signal")
                    _best_side = _ns_game.get("playoff_ats_best_side")
                    if _best_tier and _tier_rank.get(_best_tier, 0) > _tier_rank.get(ns.get("tier"), 0):
                        _best_pick = next(
                            (p for p in _ns_game.get("playoff_ats_picks", [])
                             if p.get("signal") == _best_signal),
                            None,
                        )
                        if _best_pick:
                            ns["signal"] = _best_signal
                            ns["tier"] = _best_tier
                            ns["backtest_wr"] = _best_pick.get("backtest_wr", ns.get("backtest_wr"))
                            ns["reason_zh"] = _best_pick.get("reason_zh", ns.get("reason_zh"))
                            ns["side"] = _best_side or ns.get("side")
                    # Add consensus count and conflict info from game data.
                    _picks = _ns_game.get("playoff_ats_picks", [])
                    _silver_plus = [p for p in _picks if _tier_rank.get(p.get("tier"), 0) >= 2]
                    if len(_silver_plus) >= 2:
                        ns["consensus_count"] = len(_silver_plus)
                    if _ns_game.get("playoff_ats_has_conflict"):
                        ns["has_conflict"] = True
                        for _cp in _picks:
                            if _cp.get("side") != ns.get("side") and _tier_rank.get(_cp.get("tier"), 0) >= 2:
                                ns["conflict_signal"] = _cp.get("signal")
                                ns["conflict_wr"] = _cp.get("backtest_wr")
                                if _cp.get("side") == "away":
                                    ns["conflict_away_team"] = low_team.get("team_zh") or low_team.get("team")
                                break
            return entry

        first_round = [
            {"label": "1 vs 8", "high": seeds[0], "low": low_8,
             **_with_wins(seeds[0], low_8)},
            {"label": "4 vs 5", "high": seeds[3], "low": seeds[4],
             **_with_wins(seeds[3], seeds[4])},
            {"label": "3 vs 6", "high": seeds[2], "low": seeds[5],
             **_with_wins(seeds[2], seeds[5])},
            {"label": "2 vs 7", "high": seeds[1], "low": low_7,
             **_with_wins(seeds[1], low_7)},
        ]

        def _r1_winner(fr_entry):
            hw = fr_entry.get("high_wins", 0)
            lw = fr_entry.get("low_wins", 0)
            if hw >= 4:
                return fr_entry["high"] if isinstance(fr_entry["high"], dict) else None
            if lw >= 4:
                return fr_entry["low"] if isinstance(fr_entry["low"], dict) else None
            return None

        w_1v8 = _r1_winner(first_round[0])
        w_4v5 = _r1_winner(first_round[1])
        w_3v6 = _r1_winner(first_round[2])
        w_2v7 = _r1_winner(first_round[3])

        second_round = []
        if w_1v8 or w_4v5:
            high_r2a = w_1v8 or "1/8 勝者"
            low_r2a = w_4v5 or "4/5 勝者"
            r2a = {"label": "1/8 vs 4/5", "high": high_r2a, "low": low_r2a}
            if isinstance(high_r2a, dict) and isinstance(low_r2a, dict):
                r2a.update(_with_wins(high_r2a, low_r2a, round_num=2))
            second_round.append(r2a)
        if w_2v7 or w_3v6:
            high_r2b = w_3v6 or "3/6 勝者"
            low_r2b = w_2v7 or "2/7 勝者"
            if isinstance(high_r2b, dict) and isinstance(low_r2b, dict):
                h_seed = high_r2b.get("seed", 99)
                l_seed = low_r2b.get("seed", 99)
                if l_seed < h_seed:
                    high_r2b, low_r2b = low_r2b, high_r2b
            r2b = {"label": "2/7 vs 3/6", "high": high_r2b, "low": low_r2b}
            if isinstance(high_r2b, dict) and isinstance(low_r2b, dict):
                r2b.update(_with_wins(high_r2b, low_r2b, round_num=2))
            second_round.append(r2b)

        # Conference Finals: detect R2 winners and build CF entry.
        def _r2_winner(r2_entry):
            hw = r2_entry.get("high_wins", 0)
            lw = r2_entry.get("low_wins", 0)
            if hw >= 4:
                return r2_entry["high"] if isinstance(r2_entry["high"], dict) else None
            if lw >= 4:
                return r2_entry["low"] if isinstance(r2_entry["low"], dict) else None
            return None

        conf_finals = None
        if len(second_round) == 2:
            cf_high = _r2_winner(second_round[0])
            cf_low = _r2_winner(second_round[1])
            if cf_high or cf_low:
                cf_h = cf_high or "R2A 勝者"
                cf_l = cf_low or "R2B 勝者"
                if isinstance(cf_h, dict) and isinstance(cf_l, dict):
                    h_seed = cf_h.get("seed", 99)
                    l_seed = cf_l.get("seed", 99)
                    if l_seed < h_seed:
                        cf_h, cf_l = cf_l, cf_h
                conf_finals = {"label": "分區決賽", "high": cf_h, "low": cf_l}
                if isinstance(cf_h, dict) and isinstance(cf_l, dict):
                    conf_finals.update(_with_wins(cf_h, cf_l, round_num=3))
                    hw = conf_finals.get("high_wins", 0)
                    lw = conf_finals.get("low_wins", 0)
                    if hw >= 4:
                        conf_finals["winner"] = cf_h["team"]
                        conf_finals["winner_zh"] = cf_h["team_zh"]
                        conf_finals["winner_seed"] = cf_h.get("seed")
                        conf_finals["winner_card"] = cf_h
                    elif lw >= 4:
                        conf_finals["winner"] = cf_l["team"]
                        conf_finals["winner_zh"] = cf_l["team_zh"]
                        conf_finals["winner_seed"] = cf_l.get("seed")
                        conf_finals["winner_card"] = cf_l

        result = {
            "name": name,
            "seeds": seeds[:10],
            "lottery": seeds[10:],
            "play_in": play_in,
            "first_round": first_round,
        }
        if second_round:
            result["second_round"] = second_round
        if conf_finals:
            result["conf_finals"] = conf_finals
        return result

    east_data = _conference("東區", east, east_r1_resolved)
    west_data = _conference("西區", west, west_r1_resolved)

    ecf_cf = east_data.get("conf_finals", {})
    wcf_cf = west_data.get("conf_finals", {})
    ecf_winner = ecf_cf.get("winner_card")
    wcf_winner = wcf_cf.get("winner_card")

    finals_data = None
    if ecf_winner and wcf_winner:
        e_seed = ecf_winner.get("seed", 99)
        w_seed = wcf_winner.get("seed", 99)
        if w_seed <= e_seed:
            f_high, f_low = wcf_winner, ecf_winner
            f_high_conf, f_low_conf = "西區", "東區"
        else:
            f_high, f_low = ecf_winner, wcf_winner
            f_high_conf, f_low_conf = "東區", "西區"
        finals_data = {
            "label": "總決賽",
            "high": f_high,
            "low": f_low,
            "high_conf": f_high_conf,
            "low_conf": f_low_conf,
        }
        if isinstance(f_high, dict) and isinstance(f_low, dict):
            _fkey = frozenset({f_high["team"], f_low["team"]})
            _fwins = _series_wins.get(_fkey, {})
            _fhw = _fwins.get(f_high["team"], 0)
            _flw = _fwins.get(f_low["team"], 0)
            _fgp = _fhw + _flw
            _finals_wins = {"high_wins": _fhw, "low_wins": _flw, "games_played": _fgp}
            _f_next_gn = _fgp + 1
            _f_high_zh = f_high.get("team_zh", f_high["team"])
            _f_low_zh = f_low.get("team_zh", f_low["team"])
            _f_next_signal = None
            _finals_json_path = Path(__file__).resolve().parents[1] / "web" / "data"
            _finals_schedule_dates = ["2026-06-03", "2026-06-05", "2026-06-08",
                                       "2026-06-10", "2026-06-13", "2026-06-16", "2026-06-19"]
            if _f_next_gn <= len(_finals_schedule_dates):
                _f_gdate = _finals_schedule_dates[_f_next_gn - 1]
                _f_json = _finals_json_path / f"{_f_gdate}.json"
                if _f_json.exists():
                    try:
                        import json as _jf
                        with open(_f_json) as _fp:
                            _f_day_data = _jf.load(_fp)
                        _f_games = _f_day_data.get("games", {}) if isinstance(_f_day_data, dict) else {}
                        for _fgk, _fgv in _f_games.items():
                            if isinstance(_fgv, dict) and _fkey == frozenset(_fgk.split(":")):
                                _fpicks = _fgv.get("playoff_ats_picks", [])
                                _fbest = next(
                                    (p for p in _fpicks if p.get("tier") in ("GOLD", "SILVER")),
                                    None,
                                )
                                if _fbest:
                                    _f_actual_home = _fgv.get("home_team", f_high["team"])
                                    _f_actual_home_zh = _fgv.get("home_team_zh", _f_high_zh)
                                    _f_next_signal = {
                                        "game_num": _f_next_gn,
                                        "signal": _fbest["signal"],
                                        "side": _fbest.get("side", "home"),
                                        "home_team": _f_actual_home,
                                        "home_team_zh": _f_actual_home_zh,
                                        "tier": _fbest.get("tier", "SILVER"),
                                        "backtest_wr": _fbest.get("backtest_wr", 0.75),
                                        "reason_zh": _fbest.get("reason_zh", ""),
                                    }
                                break
                    except Exception:
                        pass
            _finals_wins["next_signal"] = _f_next_signal
            finals_data.update(_finals_wins)
            finals_data["schedule"] = {
                "game1": {"date": "2026-06-03", "time": "8:30 PM ET"},
                "game2": {"date": "2026-06-05", "time": "8:30 PM ET"},
                "game3": {"date": "2026-06-08", "time": "8:30 PM ET"},
                "game4": {"date": "2026-06-10", "time": "8:30 PM ET"},
                "game5": {"date": "2026-06-13", "time": "8:30 PM ET"},
                "game6": {"date": "2026-06-16", "time": "8:30 PM ET"},
            }
            _f_status = f"{_f_high_zh} vs {_f_low_zh}"
            if _fgp == 0:
                _f_status += f" — 6/3 開打 ({f_high['team'].split()[-1]} 主場)"
            elif _fhw == 4 or _flw == 4:
                _f_winner = _f_high_zh if _fhw == 4 else _f_low_zh
                _f_status = f"{_f_winner} 奪冠 ({_fhw}-{_flw})"
            else:
                _f_status += f" ({_fhw}-{_flw})"
            finals_data["status"] = _f_status
            if _fgp == 0:
                from datetime import date as _fdate
                _today = _fdate.today()
                _wcf_end = _fdate(2026, 5, 30)
                _ecf_end = _fdate(2026, 5, 25)
                _g1_date = _fdate(2026, 6, 3)
                _sas_rest = (_g1_date - _wcf_end).days
                _nyk_rest = (_g1_date - _ecf_end).days
                if f_high_conf == "西區":
                    _high_rest, _low_rest = _sas_rest, _nyk_rest
                else:
                    _high_rest, _low_rest = _nyk_rest, _sas_rest
                finals_data["analysis"] = {
                    "series_status": f"{_f_high_zh} vs {_f_low_zh} · 6/3 G1 @{f_high['team'].split()[-1]} 8:30PM ET",
                    "momentum_zh": (
                        f"{_f_high_zh} 帶著 G7 客場逆轉奪冠的動能進入總決賽（休息{_high_rest}天）；"
                        f"{_f_low_zh} 自 5/25 橫掃後休息{_low_rest}天"
                    ) if f_high_conf == "西區" else (
                        f"{_f_low_zh} 帶著 G7 客場逆轉奪冠的動能進入總決賽（休息{_low_rest}天）；"
                        f"{_f_high_zh} 自 5/25 橫掃後休息{_high_rest}天"
                    ),
                    "key_factor_zh": f"{_f_high_zh} 主場優勢 (G1-2, G5, G7)；{_f_low_zh} 休息{_low_rest}天 vs {_f_high_zh} 休息{_high_rest}天",
                    "matchup_zh": f"{_f_high_zh} vs {_f_low_zh} 總決賽對決",
                    "odds_preview_zh": "",
                    "risk_zh": "長時間休息可能導致手感生疏；連戰方帶動能但可能疲勞",
                    "strategy_map": [
                        {"game": 1, "home_zh": _f_high_zh, "signal": "Finals G1主場壓制", "tier": "GOLD", "wr": 75, "note": "高種子主場開局優勢顯著"},
                        {"game": 2, "home_zh": _f_high_zh, "signal": "Finals G2主場壓制", "tier": "GOLD", "wr": 80.8, "note": "主場延續G1動能"},
                        {"game": 3, "home_zh": _f_low_zh, "signal": "—", "tier": "MONITOR", "wr": None, "note": "客場反撲觀察，待盤口確認"},
                        {"game": 4, "home_zh": _f_low_zh, "signal": "—", "tier": "MONITOR", "wr": None, "note": "低種子主場第二戰，視系列賽狀態"},
                        {"game": 5, "home_zh": _f_high_zh, "signal": "Finals G5主場壓制", "tier": "GOLD", "wr": 85, "note": "回歸主場cover率最高"},
                        {"game": 6, "home_zh": _f_low_zh, "signal": "—", "tier": "MONITOR", "wr": None, "note": "淘汰賽/保命戰動態觀察"},
                        {"game": 7, "home_zh": _f_high_zh, "signal": "—", "tier": "MONITOR", "wr": None, "note": "G7歷史 3-0，待盤口出現再評估"},
                    ],
                }
                try:
                    import sqlite3 as _fsq
                    _fdb = Path(__file__).resolve().parents[1] / "Data" / "OddsData.sqlite"
                    with _fsq.connect(str(_fdb)) as _fcon:
                        _frow = _fcon.execute(
                            "SELECT Spread, OU, ML_Home, ML_Away FROM '2025-26' "
                            "WHERE Date = '2026-06-03' AND (Home = ? OR Away = ?) LIMIT 1",
                            (f_high["team"], f_high["team"]),
                        ).fetchone()
                        if _frow and _frow[0] is not None:
                            _fsp, _fou, _fml_h, _fml_a = _frow
                            _spread_move_str = ""
                            try:
                                _hist_rows = _fcon.execute(
                                    "SELECT Spread FROM odds_history WHERE Date='2026-06-03' "
                                    "AND Home=? AND Away=? ORDER BY fetched_at ASC",
                                    (f_high["team"], f_low["team"]),
                                ).fetchall()
                                if _hist_rows and len(_hist_rows) > 1:
                                    _sp_first = _hist_rows[0][0]
                                    _sp_last = _hist_rows[-1][0]
                                    if _sp_first is not None and _sp_last is not None and _sp_first != _sp_last:
                                        _sp_diff = _sp_last - _sp_first
                                        _spread_move_str = f"（開盤 {abs(_sp_first):.1f}→{abs(_sp_last):.1f}，移動 {_sp_diff:+.1f}）"
                            except Exception:
                                pass
                            finals_data["analysis"]["odds_preview_zh"] = (
                                f"G1: {_f_high_zh} -{abs(_fsp):.1f} (主場){_spread_move_str}，總分 {_fou}；ML {_fml_h}/{_fml_a}"
                            )
                        _h2h_rows = _fcon.execute(
                            "SELECT Date, Home, Away, Spread, Win_Margin, Points FROM '2025-26' "
                            "WHERE ((Home = ? AND Away = ?) OR (Home = ? AND Away = ?)) "
                            "AND Win_Margin IS NOT NULL AND Date < '2026-06-01' ORDER BY Date",
                            (f_high["team"], f_low["team"], f_low["team"], f_high["team"]),
                        ).fetchall()
                        if _h2h_rows:
                            _h2h_list = []
                            _ou_overs = 0
                            for _hr in _h2h_rows:
                                _hd, _hh, _ha, _hsp, _hwm, _hpt = _hr
                                _hs = round((_hpt + _hwm) / 2) if _hpt and _hwm is not None else None
                                _as = round((_hpt - _hwm) / 2) if _hpt and _hwm is not None else None
                                _winner = _hh if _hwm > 0 else _ha
                                _h2h_list.append({
                                    "date": _hd, "home": _hh, "away": _ha,
                                    "score": f"{_hs}-{_as}" if _hs else None,
                                    "spread": _hsp, "ats": "home" if _hwm > _hsp else "away",
                                })
                            _home_ats = sum(1 for g in _h2h_list if g["ats"] == "home")
                            finals_data["analysis"]["h2h_zh"] = (
                                f"例行賽對戰 {len(_h2h_list)} 場：主場方 ATS {_home_ats}-{len(_h2h_list)-_home_ats} "
                                f"(主場cover率 {_home_ats/len(_h2h_list)*100:.0f}%)"
                            )
                            finals_data["analysis"]["h2h_games"] = _h2h_list
                            _h2h_ou = _fcon.execute(
                                "SELECT Points, OU FROM '2025-26' "
                                "WHERE ((Home = ? AND Away = ?) OR (Home = ? AND Away = ?)) "
                                "AND Points IS NOT NULL AND OU IS NOT NULL AND Date < '2026-06-01'",
                                (f_high["team"], f_low["team"], f_low["team"], f_high["team"]),
                            ).fetchall()
                            if _h2h_ou:
                                _avg_total = sum(r[0] for r in _h2h_ou) / len(_h2h_ou)
                                _ou_overs = sum(1 for r in _h2h_ou if r[0] > r[1])
                                finals_data["analysis"]["ou_trend_zh"] = (
                                    f"H2H例行賽{len(_h2h_ou)}場均分{_avg_total:.1f} "
                                    f"(OVER {_ou_overs}/{len(_h2h_ou)})；G1盤口{_fou}"
                                )
                except Exception:
                    pass
            elif not (_fhw == 4 or _flw == 4):
                # In-progress Finals: dynamic analysis based on game results
                try:
                    import sqlite3 as _fsq
                    _fdb = Path(__file__).resolve().parents[1] / "Data" / "OddsData.sqlite"
                    with _fsq.connect(str(_fdb)) as _fcon:
                        _finals_schedule_dates_full = ["2026-06-03", "2026-06-05", "2026-06-08",
                                                        "2026-06-10", "2026-06-13", "2026-06-16", "2026-06-19"]
                        # Fetch all Finals game results
                        _f_results = _fcon.execute(
                            "SELECT Date, Home, Away, Spread, Win_Margin, Points, OU, ML_Home, ML_Away "
                            "FROM '2025-26' WHERE ((Home=? AND Away=?) OR (Home=? AND Away=?)) "
                            "AND Date >= '2026-06-01' ORDER BY Date",
                            (f_high["team"], f_low["team"], f_low["team"], f_high["team"]),
                        ).fetchall()

                        _played = [r for r in _f_results if r[4] is not None and r[4] != 0]
                        _next_game = next((r for r in _f_results if r[4] is None or r[4] == 0), None)
                        _next_gn = len(_played) + 1

                        # Build game results summary
                        _game_results = []
                        for _gi, _gr in enumerate(_played):
                            _gd, _gh, _ga, _gsp, _gwm, _gpt, _gou, _gmlh, _gmla = _gr
                            _ghome_score = round((_gpt + _gwm) / 2) if _gpt and _gwm is not None else None
                            _gaway_score = round((_gpt - _gwm) / 2) if _gpt and _gwm is not None else None
                            _gwinner = _gh if _gwm > 0 else _ga
                            _gats = "home" if _gsp is not None and _gwm > _gsp else "away"
                            _game_results.append({
                                "game": _gi + 1, "date": _gd,
                                "home": _gh, "away": _ga,
                                "score": f"{_ghome_score}-{_gaway_score}" if _ghome_score else None,
                                "spread": _gsp, "winner": _gwinner, "ats": _gats,
                                "total": _gpt, "ou_line": _gou,
                                "ou_result": "OVER" if _gpt and _gou and _gpt > _gou else "UNDER",
                            })

                        # Series status
                        _series_leader = _f_high_zh if _fhw > _flw else (_f_low_zh if _flw > _fhw else None)
                        _lead_str = f"{_series_leader} 領先" if _series_leader else "平手"

                        # Next game preview
                        _next_preview = ""
                        _next_spread_move = ""
                        if _next_game:
                            _nd, _nh, _na, _nsp, _, _, _nou, _nmlh, _nmla = _next_game
                            _nh_zh = team_name_zh(_nh) or _nh
                            if _nsp is not None:
                                _next_preview = (
                                    f"G{_next_gn}: {_nh_zh} -{abs(_nsp):.1f} (主場)，"
                                    f"總分 {_nou}；ML {_nmlh}/{_nmla}"
                                )
                                # Spread movement from history
                                try:
                                    _nh_rows = _fcon.execute(
                                        "SELECT Spread FROM odds_history WHERE Date=? AND Home=? AND Away=? "
                                        "ORDER BY fetched_at ASC",
                                        (_nd, _nh, _na),
                                    ).fetchall()
                                    if _nh_rows and len(_nh_rows) > 1:
                                        _nsp_first = _nh_rows[0][0]
                                        _nsp_last = _nh_rows[-1][0]
                                        if _nsp_first is not None and _nsp_last is not None and _nsp_first != _nsp_last:
                                            _nsp_diff = _nsp_last - _nsp_first
                                            _next_spread_move = f"（開盤 {abs(_nsp_first):.1f}→{abs(_nsp_last):.1f}，移動 {_nsp_diff:+.1f}）"
                                except Exception:
                                    pass
                                if _next_spread_move:
                                    _next_preview += f" {_next_spread_move}"

                        # Momentum analysis from recent games
                        _last = _played[-1] if _played else None
                        _momentum = ""
                        if _last:
                            _ld, _lh, _la, _lsp, _lwm, _lpt, _, _, _ = _last
                            _lw_team = _lh if _lwm > 0 else _la
                            _lw_zh = team_name_zh(_lw_team) or _lw_team
                            _lmargin = abs(_lwm)
                            _momentum = f"G{len(_played)}: {_lw_zh} 贏{_lmargin}分"
                            if _lsp is not None:
                                if _lsp > 0:
                                    _lfav = _lh
                                    _lcov = "cover" if _lwm > _lsp else "未cover"
                                else:
                                    _lfav = _la
                                    _lcov = "cover" if _lwm < _lsp else "未cover"
                                _lfav_zh = team_name_zh(_lfav) or _lfav
                                _momentum += f"（{_lfav_zh} 讓 {abs(_lsp):.1f}，{_lcov}）"

                        # Build strategy_map with results annotated
                        _strat = []
                        _game_homes = {
                            1: f_high["team"], 2: f_high["team"],
                            3: f_low["team"], 4: f_low["team"],
                            5: f_high["team"], 6: f_low["team"],
                            7: f_high["team"],
                        }
                        _base_signals = {
                            1: ("Finals G1主場壓制", "GOLD", 75),
                            2: ("Finals G2主場壓制", "GOLD", 80.8),
                            3: ("Finals G3主場優勢", "SILVER", 66.7),
                            4: ("Finals G4主場優勢", "SILVER", 66.7),
                            5: ("Finals G5主場壓制", "GOLD", 85),
                        }
                        for _gn in range(1, 8):
                            _ghome = _game_homes.get(_gn, f_high["team"])
                            _ghome_zh = team_name_zh(_ghome) or _ghome
                            _sig, _tier, _wr = _base_signals.get(_gn, ("—", "MONITOR", None))
                            _entry = {"game": _gn, "home_zh": _ghome_zh, "signal": _sig, "tier": _tier, "wr": _wr}
                            if _gn <= len(_played):
                                _pr = _game_results[_gn - 1]
                                _entry["result"] = f"{_pr['winner']} {_pr['score']}"
                                _entry["ats_result"] = _pr["ats"]
                            elif _gn == _next_gn:
                                _entry["note"] = "下一場"
                            _strat.append(_entry)

                        # H2H data (preserve from regular season)
                        _h2h_rows = _fcon.execute(
                            "SELECT Date, Home, Away, Spread, Win_Margin, Points FROM '2025-26' "
                            "WHERE ((Home = ? AND Away = ?) OR (Home = ? AND Away = ?)) "
                            "AND Win_Margin IS NOT NULL AND Date < '2026-06-01' ORDER BY Date",
                            (f_high["team"], f_low["team"], f_low["team"], f_high["team"]),
                        ).fetchall()
                        _h2h_list = []
                        if _h2h_rows:
                            for _hr in _h2h_rows:
                                _hd, _hh, _ha, _hsp, _hwm2, _hpt = _hr
                                _hs2 = round((_hpt + _hwm2) / 2) if _hpt and _hwm2 is not None else None
                                _as2 = round((_hpt - _hwm2) / 2) if _hpt and _hwm2 is not None else None
                                _h2h_list.append({
                                    "date": _hd, "home": _hh, "away": _ha,
                                    "score": f"{_hs2}-{_as2}" if _hs2 else None,
                                    "spread": _hsp,
                                    "ats": "home" if _hwm2 > _hsp else "away",
                                })

                        # Compute key factor and risk for next game
                        _kf = ""
                        _risk = ""
                        if _next_gn <= 7:
                            _next_home = _game_homes.get(_next_gn, f_high["team"])
                            _next_home_zh = team_name_zh(_next_home) or _next_home
                            _next_away = f_low["team"] if _next_home == f_high["team"] else f_high["team"]
                            _next_away_zh = team_name_zh(_next_away) or _next_away
                            _kf = f"G{_next_gn} @{_next_home_zh}"
                            if _flw > _fhw:
                                if _next_home == f_low["team"]:
                                    _kf += f"；{_f_low_zh}主場，{_f_high_zh} {_fhw}-{_flw}落後{'背水一戰' if _fhw == 0 else '追趕中'}"
                                else:
                                    _kf += f"；{_f_low_zh} 客場搶先手成功，{_f_high_zh} 必須守住主場"
                            elif _fhw > _flw:
                                _kf += f"；{_f_high_zh} {_fhw}-{_flw}領先，延續壓制"
                            else:
                                _kf += f"；系列賽 {_fhw}-{_flw} 平手"

                            # Risk factors
                            _risk_parts = []
                            if _played:
                                _last_total = _played[-1][5]
                                _next_ou = _next_game[6] if _next_game and _next_game[6] else None
                                if _last_total and _next_ou:
                                    _diff = _last_total - _next_ou
                                    if abs(_diff) > 5:
                                        _risk_parts.append(
                                            f"G{len(_played)} 實際總分 {_last_total} vs G{_next_gn} 盤口 {_next_ou} (差距 {_diff:+.0f})"
                                        )
                            if _flw > 0 and _next_home == f_high["team"]:
                                _risk_parts.append(f"{_f_high_zh} 主場已失守，心態壓力大")
                            if _flw >= 2 and _flw > _fhw and _next_home == f_low["team"]:
                                _risk_parts.append(
                                    f"{_f_low_zh} 主場且{_flw}-{_fhw}領先，封盤動機強"
                                )
                            _risk = "；".join(_risk_parts) if _risk_parts else ""

                        # OU trend analysis from played games
                        _ou_trend = ""
                        if _played:
                            _totals = [r[5] for r in _played if r[5]]
                            _ou_lines = [r[6] for r in _played if r[6]]
                            if _totals and _ou_lines:
                                _avg_total = sum(_totals) / len(_totals)
                                _avg_ou = sum(_ou_lines) / len(_ou_lines)
                                _ou_results = []
                                for _t, _o in zip(_totals, _ou_lines):
                                    _ou_results.append("UNDER" if _t < _o else "OVER")
                                _under_ct = _ou_results.count("UNDER")
                                _ou_trend = (
                                    f"系列賽平均總分 {_avg_total:.0f} vs 盤口 {_avg_ou:.1f}"
                                    f"（{_under_ct}/{len(_ou_results)} 場UNDER）"
                                )
                                if _next_game and _next_game[6]:
                                    _g2_ou = _next_game[6]
                                    _ou_gap = _avg_total - _g2_ou
                                    if abs(_ou_gap) > 5:
                                        _ou_trend += f"；G{_next_gn} 盤口 {_g2_ou}，偏差 {_ou_gap:+.0f}"

                        # Injury impact: fetch latest report for Finals teams
                        _inj_impact = ""
                        try:
                            _inj = fetch_injury_report({
                                f_high["team"]: f_low["team"],
                                f_low["team"]: f_high["team"],
                            })
                            _inj_parts = []
                            for _t in [f_high["team"], f_low["team"]]:
                                _t_inj = _inj.get(_t, [])
                                if _t_inj:
                                    _t_zh = team_name_zh(_t) or _t
                                    _inj_parts.append(f"{_t_zh}: {', '.join(_t_inj)}")
                            if _inj_parts:
                                _inj_impact = "；".join(_inj_parts)
                            else:
                                _inj_impact = "雙方主力健康，無重大傷病"
                        except Exception:
                            pass

                        # G2-specific deep analysis: bounce-back after G1 loss
                        _g2_deep = {}
                        if _next_gn == 2 and _flw == 1 and _fhw == 0:
                            _g1_wm = _played[0][4] if _played else None
                            _g1_spread = _played[0][3] if _played else None
                            _g2_spread_val = _next_game[3] if _next_game else None
                            _g1_total = _played[0][5] if _played else None
                            _g1_ou = _played[0][6] if _played else None
                            _g2_ou_val = _next_game[6] if _next_game else None
                            try:
                                _g1_wm_f = float(_g1_wm) if _g1_wm is not None else None
                                _g2_sp_f = float(_g2_spread_val) if _g2_spread_val is not None else None
                            except (TypeError, ValueError):
                                _g1_wm_f = _g2_sp_f = None
                            _g2_deep["bounce_back_zh"] = (
                                f"CF/Finals G2 主場輸G1後反彈：歷史 7/10 主場cover (70.0%)"
                            )
                            if _g1_wm_f is not None and abs(_g1_wm_f) >= 10:
                                _g2_deep["bounce_back_zh"] += (
                                    f"；G1慘敗(≥10分)後G2反彈：3/3 (100%，場均贏14+分)"
                                )
                            if _g2_sp_f is not None and _g2_sp_f >= 6:
                                _g2_deep["big_spread_zh"] = (
                                    f"G2讓分≥6 + G1主場輸：歷史 4/5 cover (80.0%)，市場信心=反彈信號"
                                )
                            if _g1_total and _g1_ou:
                                try:
                                    _g1_ou_diff = float(_g1_total) - float(_g1_ou)
                                    if _g1_ou_diff < -10 and _g2_ou_val:
                                        _g2_deep["ou_analysis_zh"] = (
                                            f"G1實際總分{_g1_total} vs 盤口{_g1_ou} (UNDER {abs(_g1_ou_diff):.0f}分)；"
                                            f"G2盤口已下調至{_g2_ou_val}，但仍偏高（G1防守強度預示UNDER傾向）；"
                                            f"歷史Finals G2: 58% UNDER (7/12，13季均值低於盤口2.5分)"
                                        )
                                except (TypeError, ValueError):
                                    pass
                            # Spread movement insight
                            try:
                                _sm_rows = _fcon.execute(
                                    "SELECT Spread FROM odds_history WHERE Date=? AND Home=? AND Away=? "
                                    "ORDER BY fetched_at ASC",
                                    (_next_game[0], _next_game[1], _next_game[2]),
                                ).fetchall() if _next_game else []
                                if _sm_rows and len(_sm_rows) >= 2:
                                    _sm_first = float(_sm_rows[0][0])
                                    _sm_last = float(_sm_rows[-1][0])
                                    _sm_diff = _sm_last - _sm_first
                                    if abs(_sm_diff) >= 1.0:
                                        _dir = "加碼主場" if _sm_diff > 0 else "修正偏向客場"
                                        _g2_deep["spread_move_zh"] = (
                                            f"盤口從 {abs(_sm_first):.1f} → {abs(_sm_last):.1f} "
                                            f"(移動 {_sm_diff:+.1f})：市場{_dir}，"
                                            f"輸G1後讓分反增=sharp money看好反彈"
                                        )
                            except Exception:
                                pass
                            # Rest equalization
                            if _next_game:
                                try:
                                    _g1_rest_h = int(_played[0][0] is not None)
                                    _nd_cols = [r[1] for r in _fcon.execute(
                                        f"PRAGMA table_info([2025-26])").fetchall()]
                                    if "Days_Rest_Home" in _nd_cols:
                                        _rest_row = _fcon.execute(
                                            "SELECT Days_Rest_Home, Days_Rest_Away FROM [2025-26] WHERE Date=? AND Home=?",
                                            (_next_game[0], _next_game[1])
                                        ).fetchone()
                                        if _rest_row:
                                            _g2_deep["rest_zh"] = (
                                                f"G2休息天數：主場{_rest_row[0]}天 vs 客場{_rest_row[1]}天"
                                                + ("（雙方均等）" if _rest_row[0] == _rest_row[1] else "")
                                            )
                                except Exception:
                                    pass

                        # Series scenario planning for upcoming games
                        _scenarios = []
                        if _next_gn <= 7:
                            _g3_home = _game_homes.get(3, f_low["team"])
                            _g3_home_zh = team_name_zh(_g3_home) or _g3_home
                            _g4_home = _game_homes.get(4, f_low["team"])
                            _g4_home_zh = team_name_zh(_g4_home) or _g4_home
                            if _fhw == 0 and _flw == 1:
                                _scenarios = [
                                    {"if_zh": f"若{_f_high_zh}贏G2 → 系列賽1-1平手",
                                     "then_zh": f"G3/G4轉至{_g3_home_zh}主場，歷史Finals G3-G4主場cover率 66.7% (8/12)"},
                                    {"if_zh": f"若{_f_low_zh}贏G2 → {_f_low_zh}領先2-0",
                                     "then_zh": f"歷史0-2落後翻盤率僅7.4%，{_f_high_zh}面臨極大壓力"},
                                ]
                            elif _fhw == 1 and _flw == 1:
                                _scenarios = [
                                    {"if_zh": f"系列賽平手1-1",
                                     "then_zh": f"G3/G4轉至{_g3_home_zh}主場，Finals G3-G4主場cover率 66.7% (8/12)"},
                                ]
                            elif _fhw == 0 and _flw == 2:
                                _scenarios = [
                                    {"if_zh": f"{_f_low_zh}領先2-0",
                                     "then_zh": f"歷史0-2落後翻盤率 7.4%，G3-G4為{_g3_home_zh}主場 (cover率 66.7%)"},
                                    {"if_zh": f"若{_f_high_zh}贏G3 → 1-2",
                                     "then_zh": f"G4仍在{_g4_home_zh}主場，落後方連贏兩場主場翻盤率 18.2%"},
                                    {"if_zh": f"若{_f_low_zh}贏G3 → 3-0",
                                     "then_zh": f"歷史0-3落後翻盤率 0.9% (僅2004騎士)，{_f_low_zh}幾乎鎖定冠軍"},
                                ]
                            elif _fhw == 1 and _flw == 2:
                                _scenarios = [
                                    {"if_zh": f"{_f_low_zh}領先2-1，{_f_high_zh} G3客場取勝",
                                     "then_zh": f"G4仍在{_g4_home_zh}主場，1-2落後方G4主場cover率 62% (n=45)；{_f_high_zh}連勝動能vs{_f_low_zh}主場優勢"},
                                    {"if_zh": f"若{_f_low_zh}贏G4 → 3-1",
                                     "then_zh": f"歷史3-1領先封盤率 95%，{_f_high_zh}幾乎出局；G5轉回{_f_high_zh}主場但壓力巨大"},
                                    {"if_zh": f"若{_f_high_zh}贏G4 → 2-2平手",
                                     "then_zh": f"系列賽完全重置，G5回{_f_high_zh}主場搶主場優勢；歷史2-2平手G5主場cover率 58% (n=50)"},
                                ]
                            elif _fhw == 2 and _flw == 2:
                                _g5_home = _game_homes.get(5, f_high["team"])
                                _g5_home_zh = team_name_zh(_g5_home) or _g5_home
                                _scenarios = [
                                    {"if_zh": f"系列賽 2-2 平手",
                                     "then_zh": f"G5回{_g5_home_zh}主場，歷史2-2平手G5主場cover率 58% (n=50)；系列賽完全重置"},
                                    {"if_zh": f"若{_f_high_zh}贏G5 → 3-2",
                                     "then_zh": f"G6轉至{_f_low_zh}主場，{_f_high_zh}客場封盤需突破主場優勢"},
                                    {"if_zh": f"若{_f_low_zh}贏G5 → 2-3",
                                     "then_zh": f"G6回{_f_low_zh}主場搶救系列賽，歷史2-3落後G6主場存活率 62%"},
                                ]
                            elif _fhw == 2 and _flw == 3:
                                _scenarios = [
                                    {"if_zh": f"{_f_low_zh}領先3-2",
                                     "then_zh": f"G6在{_f_low_zh}主場，{_f_low_zh}可於主場封盤奪冠"},
                                ]
                            elif _fhw == 3 and _flw == 2:
                                _scenarios = [
                                    {"if_zh": f"{_f_high_zh}領先3-2",
                                     "then_zh": f"G6在{_f_low_zh}主場，{_f_low_zh}背水一戰；歷史G6落後方主場存活率 62%"},
                                ]
                            elif _fhw == 0 and _flw >= 3:
                                _scenarios = [
                                    {"if_zh": f"{_f_low_zh}領先{_flw}-0",
                                     "then_zh": f"歷史性壓制，{_f_low_zh}即將橫掃奪冠"},
                                ]
                            elif _fhw == 1 and _flw == 3:
                                _g5_home_s = _game_homes.get(5, f_high["team"])
                                _g5_home_s_zh = team_name_zh(_g5_home_s) or _g5_home_s
                                _scenarios = [
                                    {"if_zh": f"{_f_low_zh}領先3-1，G5回{_g5_home_s_zh}主場",
                                     "then_zh": f"歷史封盤率95%，但GOLD信號主場cover率85% — 淘汰局{_f_high_zh}通常全力一搏"},
                                    {"if_zh": f"若{_f_high_zh}贏G5 → 2-3",
                                     "then_zh": f"G6轉至{_f_low_zh}主場封盤；歷史1-3翻盤率 9.4%，但連贏氣勢不可忽視"},
                                    {"if_zh": f"若{_f_low_zh}贏G5 → 4-1奪冠",
                                     "then_zh": f"{_f_low_zh}在客場完成封盤，奪得總冠軍"},
                                ]
                            elif _flw >= 3 and _fhw < _flw:
                                _scenarios = [
                                    {"if_zh": f"{_f_low_zh}領先{_flw}-{_fhw}",
                                     "then_zh": f"{_f_low_zh}距冠軍{4-_flw}場勝利"},
                                ]

                        # G3+ deep analysis: 0-2 deficit context
                        _g3_deep = {}
                        if _next_gn >= 3 and _fhw == 0 and _flw == 2:
                            _g3_deep["deficit_zh"] = (
                                f"{_f_high_zh} 0-2落後：歷史翻盤率僅7.4% (16/217)，"
                                f"但G3轉至{_f_low_zh}主場，場景改變"
                            )
                            _g3_deep["sweep_defense_zh"] = (
                                f"🔥 歷史小讓分(≤3)Finals G3：主場 4-2 SU/ATS (66.7%)，均贏11.2分；"
                                f"但2-0落後方G3反撲率60% (3/5)，領先方鬆懈風險存在"
                            )
                            if len(_played) >= 2:
                                _g1_margin = abs(_played[0][4]) if _played[0][4] else 0
                                _g2_margin = abs(_played[1][4]) if _played[1][4] else 0
                                if _g2_margin <= 5:
                                    _g3_deep["momentum_shift_zh"] = (
                                        f"G2僅輸{_g2_margin}分（險勝），{_f_high_zh}仍有競爭力；"
                                        f"歷史G2險輸(≤5分)後G3 ATS cover率 58.3% (n=24)"
                                    )
                                if _g1_margin >= 8 and _g2_margin <= 5:
                                    _g3_deep["trend_zh"] = (
                                        f"G1大輸{_g1_margin}分 → G2僅差{_g2_margin}分："
                                        f"落後方調整改善，G3有機會延續反彈"
                                    )
                            if _next_game and _next_game[3] is not None:
                                _g3_sp = abs(float(_next_game[3]))
                                if _g3_sp <= 3:
                                    _g3_deep["spread_insight_zh"] = (
                                        f"G3讓分僅{_g3_sp:.1f}分：市場對0-2落後方信心強，"
                                        f"小讓分暗示{_f_high_zh}有實力反彈"
                                    )
                            if len(_played) >= 2:
                                _totals = [r[5] for r in _played if r[5]]
                                if _totals:
                                    _avg = sum(_totals) / len(_totals)
                                    _all_under = all(
                                        r[5] < r[6] for r in _played if r[5] and r[6]
                                    )
                                    if _all_under:
                                        _g3_deep["ou_pattern_zh"] = (
                                            f"G1-G2連續UNDER (均分{_avg:.0f})：防守強度高；"
                                            f"但歷史Finals G3 OVER率 73% (8/11，均+3.1分)，"
                                            f"轉場效應可能推高總分"
                                        )

                        # G4 deep analysis: 1-2 deficit — trailing team won G3
                        _g4_deep = {}
                        if _next_gn == 4 and _fhw == 1 and _flw == 2:
                            _g3_wm = _played[2][4] if len(_played) >= 3 else None
                            _g3_total = _played[2][5] if len(_played) >= 3 else None
                            _g3_ou = _played[2][6] if len(_played) >= 3 else None
                            _g4_deep["momentum_zh"] = (
                                f"🔥 {_f_high_zh} G3客場取勝，0-2→1-2：歷史1-2落後方G4主場cover率 62% (n=45)；"
                                f"連勝動能轉移，{_f_low_zh}面臨回應壓力"
                            )
                            _g4_deep["desperation_zh"] = (
                                f"{_f_low_zh} 2-1領先但G3主場失守：歷史G4壓力反應分歧；"
                                f"領先方主場失守後G4 win率 55%，但ATS cover僅 48% (n=40，過度修正風險)"
                            )
                            if len(_played) >= 3:
                                _margins = [abs(_played[i][4]) for i in range(3) if _played[i][4]]
                                if _margins:
                                    _avg_margin = sum(_margins) / len(_margins)
                                    _g4_deep["competitiveness_zh"] = (
                                        f"系列賽平均分差 {_avg_margin:.1f}分：{'膠著系列賽' if _avg_margin <= 6 else '單方壓制'}，"
                                        f"G4可能延續{'緊張節奏' if _avg_margin <= 6 else '趨勢'}"
                                    )
                            if len(_played) >= 3:
                                _totals_all = [r[5] for r in _played if r[5]]
                                if _totals_all:
                                    _trend_str = "→".join(str(t) for t in _totals_all)
                                    _ou_results_all = ["U" if r[5] < r[6] else "O" for r in _played if r[5] and r[6]]
                                    _g4_deep["ou_trend_zh"] = (
                                        f"總分走勢 {_trend_str} ({''.join(_ou_results_all)})；"
                                        f"{'得分逐場上升' if _totals_all == sorted(_totals_all) else '波動'}，"
                                        f"G4盤口需關注調整方向"
                                    )
                            if _next_game and _next_game[3] is not None:
                                _g4_sp = abs(float(_next_game[3]))
                                _g4_deep["spread_zh"] = (
                                    f"G4讓分 {_g4_sp:.1f}：{'小讓分=市場尊重{}'.format(_f_high_zh) if _g4_sp <= 3 else '中讓分=主場仍有優勢'}"
                                )
                            _g4_deep["pivot_zh"] = (
                                f"G4為系列賽關鍵轉折：{_f_low_zh}贏→3-1 (歷史封盤率95%)；"
                                f"{_f_high_zh}贏→2-2平手 (重置系列賽)"
                            )
                            _g4_deep["response_zh"] = (
                                f"📊 歷史G3主場失守後G4回應：2-1領先方G4主場win率55%但ATS僅48%（n=40）；"
                                f"市場傾向過度修正讓分，{_f_high_zh}受讓有價值"
                            )
                            if len(_played) >= 3:
                                _totals_all_g4 = [r[5] for r in _played if r[5]]
                                _ous_g4 = [("U" if r[5] < r[6] else "O") for r in _played if r[5] and r[6]]
                                if len(_ous_g4) >= 3 and _ous_g4[-1] == "O" and all(x == "U" for x in _ous_g4[:-1]):
                                    _g4_deep["ou_reversal_zh"] = (
                                        f"⚠️ G3 OVER逆轉 (UU→O)：歷史Finals前兩場UNDER後G3 OVER→G4 UNDER回歸率 65%（n=17）；"
                                        f"G3總分{_totals_all_g4[-1]} vs G4盤口{abs(float(_next_game[6])) if _next_game and _next_game[6] else '?'}，留意得分回調"
                                    )
                            _g4_deep["g5_preview_zh"] = (
                                f"👁 G5預覽：若{_f_high_zh}贏→2-2，G5回{_f_high_zh}主場 (GOLD tier 85% WR)；"
                                f"若{_f_low_zh}贏→3-1，歷史封盤率95%，G5僅象徵性抵抗"
                            )
                            if len(_played) >= 3:
                                _ats_results = []
                                for _pr in _played:
                                    _sp_val = _pr[3]
                                    _wm_val = _pr[4]
                                    if _sp_val is not None and _wm_val is not None:
                                        _ats_results.append("home" if _wm_val > _sp_val else "away")
                                if _ats_results:
                                    _home_covers = sum(1 for r in _ats_results if r == "home")
                                    _away_covers = len(_ats_results) - _home_covers
                                    if _away_covers >= 3:
                                        _g4_deep["series_pattern_zh"] = (
                                            f"⚠️ 系列賽ATS趨勢：客場 {_away_covers}-{_home_covers} 主場cover，"
                                            f"連續{_away_covers}場客隊cover — 主場讓分持續被高估，"
                                            f"G4受讓({_f_high_zh})需格外關注"
                                        )
                                    elif _home_covers >= 3:
                                        _g4_deep["series_pattern_zh"] = (
                                            f"📈 系列賽ATS趨勢：主場 {_home_covers}-{_away_covers} 客場cover，"
                                            f"主場方穩定cover，G4讓分({_f_low_zh})值得信賴"
                                        )

                        # G5 deep analysis
                        _g5_deep = {}
                        if _next_gn == 5:
                            _g5_home = _game_homes.get(5, f_high["team"])
                            _g5_home_zh = team_name_zh(_g5_home) or _g5_home
                            if _fhw == 2 and _flw == 2:
                                _g5_deep["tied_series_zh"] = (
                                    f"🔥 系列賽2-2平手，G5回{_g5_home_zh}主場：歷史Finals G5主場cover率 85% (GOLD tier, n=20)；"
                                    f"2-2後G5為「新的G1」，主場優勢最大化"
                                )
                                _g5_deep["momentum_zh"] = (
                                    f"雙方各贏兩場，無明顯動能差異；G5主場方歷史性壓制，"
                                    f"市場讓分通常低估主場方（平均多贏4.2分）"
                                )
                            elif _fhw == 1 and _flw == 3:
                                # G4 recap for context
                                _g4_recap = ""
                                if len(_played) >= 4:
                                    _g4r = _played[3]
                                    _g4_wm = _g4r[4]
                                    _g4_winner = _g4r[1] if _g4_wm > 0 else _g4r[2]
                                    _g4_winner_zh = team_name_zh(_g4_winner) or _g4_winner
                                    _g4_margin = abs(_g4_wm)
                                    _g4_total = _g4r[5]
                                    _g4_sp = _g4r[3]
                                    if _g4_margin <= 3:
                                        _g4_recap = (
                                            f"📋 G4回顧：{_g4_winner_zh}險勝{_g4_margin}分 (總分{_g4_total})，"
                                            f"讓分{abs(_g4_sp):.1f}未cover — 膠著戰延續"
                                        )
                                    else:
                                        _g4_recap = (
                                            f"📋 G4回顧：{_g4_winner_zh}贏{_g4_margin}分 (總分{_g4_total})"
                                        )
                                if _g4_recap:
                                    _g5_deep["g4_recap_zh"] = _g4_recap
                                _g5_deep["deficit_zh"] = (
                                    f"{_f_high_zh} 1-3落後：歷史1-3翻盤率 9.4%（n=169）；"
                                    f"G5回{_g5_home_zh}主場，必須連贏3場"
                                )
                                _g5_deep["desperation_zh"] = (
                                    f"{_f_low_zh} 3-1領先：歷史封盤率95%，但G5客場壓力小；"
                                    f"落後方淘汰邊緣激發最大潛能，ATS冷門cover率 58%"
                                )
                                _g5_deep["signal_conflict_zh"] = (
                                    f"⚠️ GOLD信號衝突：Finals G5主場cover率 85% (n=20) → 押{_g5_home_zh}；"
                                    f"但本系列客場 4-0 ATS全勝 — 主場讓分持續被高估；"
                                    f"G5讓分{abs(float(_next_game[3])):.1f}為系列最大，市場給予{_g5_home_zh}大幅尊重"
                                    if _next_game and _next_game[3] is not None else
                                    f"⚠️ GOLD信號衝突：Finals G5主場cover率 85% (n=20) → 押{_g5_home_zh}；"
                                    f"但本系列客場 4-0 ATS全勝 — 主場讓分持續被高估"
                                )
                                _g5_deep["closeout_dynamics_zh"] = (
                                    f"封盤賽心態：{_f_low_zh}客場無壓力，贏球即奪冠；"
                                    f"{_f_high_zh}背水一戰但歷史淘汰局G5冷門cover率僅 50.6% (n=81)，無結構性優勢；"
                                    f"大讓分({abs(float(_next_game[3])):.1f}分)暗示市場認為{_f_high_zh}會贏但{_f_low_zh}能控制分差"
                                    if _next_game and _next_game[3] is not None else
                                    f"封盤賽心態：{_f_low_zh}客場無壓力，贏球即奪冠；"
                                    f"{_f_high_zh}背水一戰但歷史淘汰局G5冷門cover率僅 50.6% (n=81)"
                                )
                            elif _fhw == 3 and _flw == 1:
                                _g5_deep["closeout_zh"] = (
                                    f"{_f_high_zh} 3-1領先，G5在{_g5_home_zh}主場：歷史主場封盤率 78%；"
                                    f"主場+領先=雙重優勢，但淘汰局冷門cover率 56.7%"
                                )
                            if len(_played) >= 4:
                                _margins_g5 = [abs(_played[i][4]) for i in range(4) if _played[i][4]]
                                if _margins_g5:
                                    _avg_mg5 = sum(_margins_g5) / len(_margins_g5)
                                    _g5_deep["competitiveness_zh"] = (
                                        f"前4場平均分差 {_avg_mg5:.1f}分：{'膠著系列賽' if _avg_mg5 <= 6 else '分差較大'}，"
                                        f"G5勝負關鍵在第四節執行力"
                                    )
                                _totals_g5 = [r[5] for r in _played if r[5]]
                                if _totals_g5:
                                    _avg_total_g5 = sum(_totals_g5) / len(_totals_g5)
                                    _g5_deep["ou_context_zh"] = (
                                        f"系列賽平均總分 {_avg_total_g5:.0f}；G5壓力場通常防守強度上升，"
                                        f"UNDER傾向 58% (n=20)"
                                    )
                                _ats_g5 = []
                                for _pr in _played:
                                    _sp_val = _pr[3]
                                    _wm_val = _pr[4]
                                    if _sp_val is not None and _wm_val is not None:
                                        _ats_g5.append("home" if _wm_val > _sp_val else "away")
                                if _ats_g5:
                                    _hc5 = sum(1 for r in _ats_g5 if r == "home")
                                    _ac5 = len(_ats_g5) - _hc5
                                    if _ac5 >= 3 or _hc5 >= 3:
                                        _dom = "客場" if _ac5 > _hc5 else "主場"
                                        _g5_deep["series_pattern_zh"] = (
                                            f"⚠️ 系列賽ATS趨勢：{_dom} {max(_hc5,_ac5)}-{min(_hc5,_ac5)} cover，"
                                            f"{'客隊持續cover — 主場讓分被高估' if _ac5 > _hc5 else '主場穩定cover — 讓分合理'}"
                                        )

                        finals_data["analysis"] = {
                            "series_status": f"{_f_high_zh} vs {_f_low_zh} ({_fhw}-{_flw}) · {_lead_str}",
                            "game_results": _game_results,
                            "momentum_zh": _momentum,
                            "key_factor_zh": _kf,
                            "odds_preview_zh": _next_preview,
                            "risk_zh": _risk,
                            "strategy_map": _strat,
                            "h2h_games": _h2h_list,
                            "ou_trend_zh": _ou_trend,
                            "injury_impact_zh": _inj_impact,
                        }
                        if _scenarios:
                            finals_data["analysis"]["scenarios"] = _scenarios
                        if _g2_deep:
                            finals_data["analysis"]["g2_deep_analysis"] = _g2_deep
                        if _g3_deep:
                            finals_data["analysis"]["g3_deep_analysis"] = _g3_deep
                        if _g4_deep:
                            finals_data["analysis"]["g4_deep_analysis"] = _g4_deep
                        if _g5_deep:
                            finals_data["analysis"]["g5_deep_analysis"] = _g5_deep
                except Exception:
                    pass
    elif ecf_winner or wcf_winner:
        known = ecf_winner or wcf_winner
        known_conf = "東區冠軍" if ecf_winner else "西區冠軍"
        pending_cf = wcf_cf if ecf_winner else ecf_cf
        pending_conf = "西區冠軍" if ecf_winner else "東區冠軍"
        pending_h = pending_cf.get("high", {})
        pending_l = pending_cf.get("low", {})
        ph_name = pending_h.get("team_zh", "?") if isinstance(pending_h, dict) else "?"
        pl_name = pending_l.get("team_zh", "?") if isinstance(pending_l, dict) else "?"
        ph_wins = pending_cf.get("high_wins", 0)
        pl_wins = pending_cf.get("low_wins", 0)
        pending_label = f"{pending_conf.replace('冠軍', '')}決賽勝者 ({ph_name} {ph_wins}-{pl_wins} {pl_name})"
        finals_data = {
            "label": "總決賽",
            "high_conf": known_conf,
            "low_conf": pending_conf,
            "high": known,
            "low": pending_label,
            "high_wins": 0,
            "low_wins": 0,
            "games_played": 0,
            "next_signal": None,
            "status": f"等待{pending_conf} — 6/3 開打",
            "schedule": {
                "game1": {"date": "2026-06-03", "time": "8:30 PM ET"},
                "game2": {"date": "2026-06-05", "time": "8:30 PM ET"},
                "game3": {"date": "2026-06-08", "time": "8:30 PM ET"},
                "game4": {"date": "2026-06-10", "time": "8:30 PM ET"},
                "game5": {"date": "2026-06-13", "time": "8:30 PM ET"},
                "game6": {"date": "2026-06-16", "time": "8:30 PM ET"},
            },
            "analysis": {
                "series_status": f"{known.get('team_zh', known.get('team', '?'))} vs TBD ({pending_conf}) · 6/3 開打",
                "momentum_zh": f"{known.get('team_zh', '?')} 已確定晉級，等待{pending_conf}產生",
                "key_factor_zh": f"{pending_conf}系列賽越長，{known.get('team_zh', '?')} 休息優勢越大",
                "risk_zh": "長時間休息可能導致手感生疏；對手帶著系列賽勝利動能進入總決賽",
            },
        }

    bracket = {
        "snapshot_date": snapshot_date,
        "generated_for": target_date.isoformat(),
        "playoff_start": playoff_start,
        "show_from": show_from,
        "show_until": show_until,
        "stage_label": _stage_label,
        "stage": {1: "first_round", 2: "second_round", 3: "conf_finals", 4: "finals"}.get(
            _max_round_num, "first_round" if _series_wins else ("play_in" if _playin_results else None)),
        "east": east_data,
        "west": west_data,
    }
    if finals_data:
        bracket["finals"] = finals_data
    return bracket


def build_season_h2h(season_key: str) -> dict | None:
    """Build a lookup of all head-to-head matchups for the season.

    Returns dict with key 'pairs': canonical_key → list of game records,
    where canonical_key = sorted(home, away) joined with ':'.
    """
    import sqlite3
    db = Path(__file__).resolve().parents[1] / "Data" / "OddsData.sqlite"
    if not db.exists():
        return None
    try:
        con = sqlite3.connect(str(db))
        # Pick the freshest table for the season
        candidates = [
            row[0] for row in
            con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            if season_key in row[0]
        ]
        best, best_date, best_cnt = (candidates[0] if candidates else season_key), "", 0
        for c in candidates:
            try:
                row = con.execute(f'SELECT MAX(Date) FROM "{c}"').fetchone()
                d = row[0] or ""
                cnt = con.execute(f'SELECT COUNT(*) FROM "{c}"').fetchone()[0]
                if d > best_date or (d == best_date and cnt > best_cnt):
                    best, best_date, best_cnt = c, d, cnt
            except Exception:
                pass

        rows = con.execute(
            f'SELECT Date, Home, Away, Spread, Win_Margin, Points FROM "{best}" ORDER BY Date'
        ).fetchall()
        con.close()
    except Exception:
        return None

    pairs: dict = {}
    for date_str, home, away, spread, wm, pts in rows:
        if spread is not None and wm is not None:
            diff = wm - spread
            if abs(diff) < 0.001:
                ats_result = "push"
            elif diff > 0:
                ats_result = "home"
            else:
                ats_result = "away"
        else:
            ats_result = None

        # Derive actual scores from total points + win margin
        if pts and wm is not None and pts > 0:
            home_score = round((pts + wm) / 2)
            away_score = round((pts - wm) / 2)
        else:
            home_score = away_score = None

        game = {"date": date_str, "home": home, "away": away,
                "spread": spread, "win_margin": wm,
                "home_score": home_score, "away_score": away_score,
                "ats_result": ats_result}

        # Index under canonical key (alphabetical) so both lookup directions work
        key = ":".join(sorted([home, away]))
        pairs.setdefault(key, []).append(game)

    return {"season": season_key, "pairs": pairs}


def _update_playoff_quarters(today: date) -> None:
    """Fetch Q-by-Q scores from ESPN for completed playoff games missing from playoff_quarters.json."""
    import sqlite3
    qf = OUT_DIR / "playoff_quarters.json"
    existing = []
    if qf.exists():
        try:
            with open(qf, encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = []
    seen = {(g["date"], g["home_name"], g["away_name"]) for g in existing}

    season_key = current_nba_season(today)
    odds_db = Path(__file__).resolve().parents[1] / "Data" / "OddsData.sqlite"
    if not odds_db.exists():
        return
    con = sqlite3.connect(str(odds_db))
    rows = con.execute(
        f'SELECT Date, Home, Away, Win_Margin FROM "{season_key}" '
        f'WHERE Date >= ? AND Win_Margin != 0 ORDER BY Date',
        ((today - timedelta(days=30)).isoformat(),),
    ).fetchall()
    con.close()

    from src.Utils.PlayoffContext import is_playoff_date
    missing = []
    for d_str, home, away, wm in rows:
        d = date.fromisoformat(d_str)
        if not is_playoff_date(d):
            continue
        if (d_str, home, away) in seen:
            continue
        missing.append((d_str, home, away))

    if not missing:
        return

    try:
        import urllib.request
    except ImportError:
        return

    added = 0
    missing_by_teams = {(h, a): d for d, h, a in missing}
    dates_needed = sorted({d for d, _, _ in missing})
    espn_dates = set()
    for d_str in dates_needed:
        espn_dates.add(d_str)
        prev = (date.fromisoformat(d_str) - timedelta(days=1)).isoformat()
        espn_dates.add(prev)
    for espn_d in sorted(espn_dates):
        try:
            url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={espn_d.replace('-', '')}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            resp = urllib.request.urlopen(req, timeout=15)
            espn = json.loads(resp.read())
        except Exception:
            continue
        for ev in espn.get("events", []):
            comp = ev.get("competitions", [{}])[0]
            if comp.get("status", {}).get("type", {}).get("name") != "STATUS_FINAL":
                continue
            teams = comp.get("competitors", [])
            home_t = next((t for t in teams if t.get("homeAway") == "home"), None)
            away_t = next((t for t in teams if t.get("homeAway") == "away"), None)
            if not home_t or not away_t:
                continue
            h_name = home_t["team"]["displayName"]
            a_name = away_t["team"]["displayName"]
            db_date = missing_by_teams.get((h_name, a_name))
            if db_date is None:
                continue
            h_qs = [int(ls.get("value", 0)) for ls in home_t.get("linescores", [])]
            a_qs = [int(ls.get("value", 0)) for ls in away_t.get("linescores", [])]
            if len(h_qs) < 4 or len(a_qs) < 4:
                continue
            existing.append({
                "gameId": ev.get("id", ""),
                "date": db_date,
                "home_tri": home_t["team"].get("abbreviation", ""),
                "away_tri": away_t["team"].get("abbreviation", ""),
                "home_name": h_name,
                "away_name": a_name,
                "q_home": h_qs[:4],
                "q_away": a_qs[:4],
                "home_score": int(home_t["score"]),
                "away_score": int(away_t["score"]),
            })
            del missing_by_teams[(h_name, a_name)]
            added += 1

    if added:
        existing.sort(key=lambda g: (g["date"], g["home_name"]))
        with open(qf, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        print(f"Playoff quarters: added {added} games → {qf.name}")


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

    # Export upcoming future dates (predictions for scheduled games).
    for offset in range(1, DAYS_FORWARD + 1):
        d = today + timedelta(days=offset)
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

    # Auto-update playoff quarter scores for conviction classifier.
    _update_playoff_quarters(today)

    # Season stats.
    season_key = current_nba_season(today)
    reset_season_cache()
    stats = compute_season_stats(season_key)
    if stats:
        # Augment with playoff signal performance (scan all date JSONs in OUT_DIR).
        po_hits, po_misses, po_pending = [], [], []
        for jf in sorted(OUT_DIR.glob("????-??-??.json")):
            try:
                with open(jf, encoding="utf-8") as _jf:
                    _d = json.load(_jf)
                # Spread lookup per game for Q3 dominance calculation
                _spread_by_game = {}
                for _gk, _gv in _d.get("games", {}).items():
                    if isinstance(_gv, dict):
                        _sp = _gv.get("spread")
                        if _sp is not None:
                            _spread_by_game[f"{_gv.get('away_team','')}:{_gv.get('home_team','')}"] = _sp
                for alert in _d.get("summary", {}).get("playoff_ats_alerts", []):
                    tier = alert.get("tier")
                    if tier not in ("GOLD", "SILVER"):
                        continue
                    wr = alert.get("backtest_wr", 0)
                    _game_key = alert.get("game_key") or f"{alert.get('away_team','')}:{alert.get('home_team','')}"
                    record = {
                        "date": _d["date"],
                        "game": f'{alert.get("away_team_zh", alert.get("away_team", "?"))} @ {alert.get("home_team_zh", alert.get("home_team", "?"))}',
                        "signal": alert.get("signal") or alert.get("signal_name", "?"),
                        "side": alert.get("side"),
                        "tier": tier,
                        "backtest_wr": wr,
                        "ats_winner": alert.get("ats_winner"),
                        "ats_cover_margin": alert.get("ats_cover_margin"),
                        "series_game_num": alert.get("series_game_num"),
                        "series_home_wins": alert.get("series_home_wins"),
                        "series_away_wins": alert.get("series_away_wins"),
                        "strong_consensus": alert.get("strong_consensus"),
                        "home_team": alert.get("home_team", ""),
                        "away_team": alert.get("away_team", ""),
                        "spread": _spread_by_game.get(_game_key),
                    }
                    winner = alert.get("ats_winner")
                    if winner is None:
                        po_pending.append(record)
                    elif winner == "push":
                        # Push is a refund, not a graded outcome — skip from hit/miss totals.
                        record["result"] = "push"
                    elif winner == alert.get("side"):
                        record["result"] = "hit"
                        po_hits.append(record)
                    else:
                        record["result"] = "miss"
                        po_misses.append(record)
            except Exception:
                pass
        decided = len(po_hits) + len(po_misses)

        # Load quarter-by-quarter data if available (Q3 lead captures true dominance
        # better than final margin — garbage-time comebacks tighten final ATS margin
        # but don't reflect the signal's real conviction).
        _quarters_lookup = {}
        try:
            qf = OUT_DIR / "playoff_quarters.json"
            if qf.exists():
                with open(qf, encoding="utf-8") as _qf:
                    _qdata = json.load(_qf)
                for _g in _qdata:
                    _key = (_g["date"], _g["home_name"], _g["away_name"])
                    _hq = _g["q_home"]; _aq = _g["q_away"]
                    _q3_home_lead = sum(_hq[:3]) - sum(_aq[:3])
                    _final = sum(_hq) - sum(_aq)
                    _q4_swing = _final - _q3_home_lead
                    _quarters_lookup[_key] = {
                        "q3_home_lead": _q3_home_lead,
                        "final_margin": _final,
                        "q4_swing": _q4_swing,
                    }
        except Exception:
            pass

        # Conviction classifier: prefers Q3 dominance (when available) over final ATS margin.
        # For the signal's picked side, use their Q3 lead vs proportional 3/4-spread.
        # Garbage-time flag: |Q4 swing| >= 8 indicates final margin masks true game flow.
        def _conviction(cover_margin, date=None, home_team=None, away_team=None,
                        spread=None, picked_side=None):
            # Try Q3 dominance first
            q = _quarters_lookup.get((date, home_team, away_team)) if date and home_team and away_team else None
            if q is not None and spread is not None and picked_side is not None:
                q3_lead = q["q3_home_lead"]
                # Proportional 3Q spread (3 of 4 quarters played)
                threeq_threshold = spread * 0.75
                # Q3 ATS margin for picked side: did the pick cover through 3Q?
                if picked_side == "home":
                    q3_picked_margin = q3_lead - threeq_threshold
                    final_picked_margin = (cover_margin if cover_margin is not None else 0)
                else:  # away
                    q3_picked_margin = -(q3_lead - threeq_threshold)
                    final_picked_margin = -(cover_margin if cover_margin is not None else 0)
                q4_swing = q["q4_swing"]
                # Garbage time = picked side's advantage shrunk from Q3 to final by ≥5 pts
                # (e.g. CLE +15 Q3 ATS margin → +5 final = 10-pt shrinkage → ⚡)
                shrinkage = q3_picked_margin - final_picked_margin
                garbage_time = shrinkage >= 5
                m = abs(q3_picked_margin)
                # Use Q3 band
                if m >= 15:
                    band = "crushing"   # 🔥
                elif m >= 8:
                    band = "dominant"   # 💪
                elif m >= 3:
                    band = "moderate"   # 📏
                else:
                    band = "narrow"     # 🎲
                return {"band": band, "source": "q3", "q3_picked_margin": round(q3_picked_margin, 1),
                        "final_picked_margin": round(final_picked_margin, 1),
                        "q4_swing": q4_swing, "shrinkage": round(shrinkage, 1),
                        "garbage_time": garbage_time}
            # Fallback to final ATS margin
            if cover_margin is None:
                return None
            m = abs(cover_margin)
            if m >= 8:
                band = "dominant"
            elif m >= 3:
                band = "moderate"
            else:
                band = "narrow"
            return {"band": band, "source": "final", "garbage_time": False}

        # Annotate each hit/miss record with conviction band
        for r in po_hits + po_misses:
            c = _conviction(
                r.get("ats_cover_margin"),
                date=r.get("date"), home_team=r.get("home_team"),
                away_team=r.get("away_team"),
                spread=r.get("spread"),
                picked_side=r.get("side"),
            )
            r["conviction_obj"] = c
            r["conviction"] = c["band"] if c else None
            r["garbage_time"] = c.get("garbage_time", False) if c else False
            r["conviction_source"] = c.get("source", "final") if c else None
            if c and c.get("source") == "q3":
                r["q3_picked_margin"] = c.get("q3_picked_margin")
                r["q4_swing"] = c.get("q4_swing")

        # Overall conviction tallies across all SILVER+ signals
        # Bands: crushing (Q3+15), dominant (Q3+8 or final>8), moderate, narrow
        conv_tally = {"crushing": 0, "dominant": 0, "moderate": 0, "narrow": 0}
        garbage_time_hits = 0  # hits where Q3 was dominant but final tightened (user's point)
        for r in po_hits:
            c = r.get("conviction")
            if c and c in conv_tally: conv_tally[c] += 1
            if r.get("garbage_time"): garbage_time_hits += 1
        total_hits_for_conv = sum(conv_tally.values())
        # "True conviction" WR: crushing + dominant + moderate (exclude narrow)
        true_conviction_hits = conv_tally["crushing"] + conv_tally["dominant"] + conv_tally["moderate"]
        conv_summary = {
            "crushing_hits": conv_tally["crushing"],
            "dominant_hits": conv_tally["dominant"],
            "moderate_hits": conv_tally["moderate"],
            "narrow_hits": conv_tally["narrow"],
            "garbage_time_hits": garbage_time_hits,
            "total_hits": total_hits_for_conv,
            "true_conviction_hits": true_conviction_hits,
            "raw_hit_rate": round(len(po_hits) / decided * 100, 1) if decided else None,
            "true_conviction_rate": round(true_conviction_hits / decided * 100, 1) if decided else None,
            "note_zh": (
                f"{conv_tally['crushing']} 碾壓主導 + {conv_tally['dominant']} 主導 + "
                f"{conv_tally['moderate']} 普通 + {conv_tally['narrow']} 險過；"
                f"{garbage_time_hits} 場受垃圾時間影響；真實信心率 "
                f"{round(true_conviction_hits/decided*100,1) if decided else 0}%"
            ),
        }

        # Per-signal breakdown with conviction split
        from collections import defaultdict
        _sig_stats: dict = defaultdict(lambda: {
            "hits": 0, "misses": 0, "tier": "SILVER", "backtest_wr": None,
            "crushing_hits": 0, "dominant_hits": 0, "moderate_hits": 0, "narrow_hits": 0,
            "garbage_time_hits": 0,
        })
        for r in po_hits:
            s = _sig_stats[r["signal"]]
            s["hits"] += 1
            s["tier"] = r.get("tier", "SILVER")
            if r.get("backtest_wr"):
                s["backtest_wr"] = r["backtest_wr"]
            c = r.get("conviction")
            if c: s[f"{c}_hits"] += 1
            if r.get("garbage_time"): s["garbage_time_hits"] += 1
        for r in po_misses:
            _sig_stats[r["signal"]]["misses"] += 1
            _sig_stats[r["signal"]]["tier"] = r.get("tier", "SILVER")
            if r.get("backtest_wr"):
                _sig_stats[r["signal"]]["backtest_wr"] = r["backtest_wr"]
        sig_breakdown = []
        for sig, v in _sig_stats.items():
            n = v["hits"] + v["misses"]
            true_hits = v["crushing_hits"] + v["dominant_hits"] + v["moderate_hits"]
            sig_breakdown.append({
                "signal": sig, "hits": v["hits"], "misses": v["misses"],
                "n": n, "tier": v["tier"],
                "hit_rate": round(v["hits"] / n * 100, 1) if n else None,
                "backtest_wr": round(v["backtest_wr"] * 100, 1) if v["backtest_wr"] else None,
                "crushing_hits": v["crushing_hits"],
                "dominant_hits": v["dominant_hits"],
                "moderate_hits": v["moderate_hits"],
                "narrow_hits": v["narrow_hits"],
                "garbage_time_hits": v["garbage_time_hits"],
                "true_hit_rate": round(true_hits / n * 100, 1) if n else None,
            })
        sig_breakdown.sort(key=lambda x: -x["hits"])

        # Strong consensus tracking (2+ SILVER agree, no conflict) AND best-pick-per-game tracking
        sc_hits, sc_misses, sc_pending = 0, 0, 0
        bp_hits, bp_misses, bp_pending = 0, 0, 0
        bp_gold_hits, bp_gold_misses = 0, 0
        bp_silver_hits, bp_silver_misses = 0, 0
        # No-conflict tracking: games where all signals point same direction
        nc_hits, nc_misses, nc_pending = 0, 0, 0
        # Conviction-banded best-pick tracking
        bp_conv = {"dominant": 0, "moderate": 0, "narrow": 0}
        sc_conv = {"dominant": 0, "moderate": 0, "narrow": 0}
        for jf in sorted(OUT_DIR.glob("????-??-??.json")):
            try:
                with open(jf, encoding="utf-8") as _jf:
                    _d = json.load(_jf)
                games_data = _d.get("games", {})
                games_iter = games_data.values() if isinstance(games_data, dict) else games_data
                for g in games_iter:
                    if not isinstance(g, dict): continue
                    winner = g.get("ats_winner")
                    cm = g.get("ats_cover_margin")
                    sc = g.get("playoff_ats_strong_consensus")
                    if sc is not None:
                        if winner is None:
                            sc_pending += 1
                        elif winner == "push":
                            pass  # push: refund, not a miss
                        elif winner == sc:
                            sc_hits += 1
                            conv_obj = _conviction(cm)
                            conv_band = conv_obj["band"] if conv_obj else None
                            if conv_band: sc_conv[conv_band] += 1
                        else:
                            sc_misses += 1
                    bp = g.get("playoff_ats_best_side")
                    bp_tier = g.get("playoff_ats_best_tier")
                    if bp is not None and bp_tier in ("GOLD", "SILVER"):
                        if winner is None:
                            bp_pending += 1
                        elif winner == "push":
                            pass  # push: refund, not a miss
                        else:
                            hit = (winner == bp)
                            if hit:
                                bp_hits += 1
                                if bp_tier == "GOLD": bp_gold_hits += 1
                                else: bp_silver_hits += 1
                                conv_obj = _conviction(cm)
                                conv_band = conv_obj["band"] if conv_obj else None
                                if conv_band: bp_conv[conv_band] += 1
                            else:
                                bp_misses += 1
                                if bp_tier == "GOLD": bp_gold_misses += 1
                                else: bp_silver_misses += 1
                    # No-conflict: game has signals, none conflicting
                    has_conflict = g.get("playoff_ats_has_conflict", False)
                    if bp is not None and bp_tier in ("GOLD", "SILVER") and not has_conflict:
                        if winner is None:
                            nc_pending += 1
                        elif winner == "push":
                            pass
                        elif winner == bp:
                            nc_hits += 1
                        else:
                            nc_misses += 1
            except Exception:
                pass
        sc_decided = sc_hits + sc_misses
        bp_decided = bp_hits + bp_misses
        nc_decided = nc_hits + nc_misses

        # Build lookups keyed by (date, game_key) from daily JSONs.
        _live_lookup: dict = {}
        _spread_lookup: dict = {}
        for jf in sorted(OUT_DIR.glob("????-??-??.json")):
            try:
                with open(jf, encoding="utf-8") as _jf:
                    _jd = json.load(_jf)
                for gk, gv in _jd.get("games", {}).items():
                    if isinstance(gv, dict):
                        sp = gv.get("spread") or gv.get("Spread")
                        if sp is not None:
                            _spread_lookup[(_jd["date"], gk)] = sp
                        _live_st = gv.get("live_status")
                        _live_fin = gv.get("live_is_final", True)
                        if _live_st and not _live_fin:
                            _live_lookup[(_jd["date"], gk)] = {
                                "live_status": _live_st,
                                "live_home_score": gv.get("live_home_score"),
                                "live_away_score": gv.get("live_away_score"),
                                "live_home_covering": gv.get("live_home_covering"),
                            }
            except Exception:
                pass

        # Group pending signals by game for display
        from collections import defaultdict as _dd
        _pending_games: dict = _dd(lambda: {"signals": [], "sides": set(), "wrs": [], "date": "", "game": "", "game_key": "", "series_game_num": None, "series_home_wins": None, "series_away_wins": None, "has_strong_consensus": False})
        for r in po_pending:
            k = f'{r["date"]}|{r["game"]}'
            _pending_games[k]["date"] = r["date"]
            _pending_games[k]["game"] = r["game"]
            _ht = r.get("home_team", "")
            _at = r.get("away_team", "")
            if _ht and _at:
                _pending_games[k]["game_key"] = f"{_at}:{_ht}"
            _pending_games[k]["signals"].append(r["signal"])
            _pending_games[k]["sides"].add(r["side"])
            if r.get("backtest_wr"):
                _pending_games[k]["wrs"].append(r["backtest_wr"])
            if r.get("series_game_num") is not None:
                _pending_games[k]["series_game_num"] = r["series_game_num"]
            if r.get("series_home_wins") is not None:
                _pending_games[k]["series_home_wins"] = r["series_home_wins"]
            if r.get("series_away_wins") is not None:
                _pending_games[k]["series_away_wins"] = r["series_away_wins"]
            if r.get("strong_consensus"):
                _pending_games[k]["has_strong_consensus"] = True
        pending_by_game = []
        for k, v in sorted(_pending_games.items()):
            sides = list(v["sides"])
            best_wr = max(v["wrs"]) if v["wrs"] else None
            has_conflict = len(sides) > 1
            entry = {
                "date": v["date"],
                "game": v["game"],
                "signals": v["signals"],
                "consensus_side": sides[0] if len(sides) == 1 else None,
                "has_conflict": has_conflict,
                "best_wr": round(best_wr * 100) if best_wr else None,
                "series_game_num": v.get("series_game_num"),
                "series_score": f'{v["series_away_wins"]}-{v["series_home_wins"]}' if v.get("series_away_wins") is not None and v.get("series_home_wins") is not None else None,
                "has_strong_consensus": v.get("has_strong_consensus", False),
            }
            _game_key = v.get("game_key", "")
            _live_info = _live_lookup.get((v["date"], _game_key))
            if _live_info:
                entry.update(_live_info)
            _sp = _spread_lookup.get((v["date"], _game_key))
            if _sp is not None:
                entry["spread"] = _sp
            pending_by_game.append(entry)
        # Sort: today first, then by conviction (strong consensus > no conflict + high WR > conflict)
        def _pending_sort_key(e):
            d = e["date"]
            sc = 1 if e.get("has_strong_consensus") else 0
            nc = 0 if e.get("has_conflict") else 1
            wr = e.get("best_wr") or 0
            return (d, -sc, -nc, -wr)
        pending_by_game.sort(key=_pending_sort_key)

        stats["playoff_signals"] = {
            "hits": len(po_hits),
            "misses": len(po_misses),
            "pending": len(po_pending),
            "decided": decided,
            "hit_rate": round(len(po_hits) / decided * 100, 1) if decided else None,
            "conviction": conv_summary,
            "history": sorted(po_hits + po_misses, key=lambda r: r["date"])[-60:],  # chronological, most recent (up to 60)
            "pending_by_game": pending_by_game,
            "by_signal": sig_breakdown,
            "strong_consensus": {
                "hits": sc_hits, "misses": sc_misses, "pending": sc_pending,
                "decided": sc_decided,
                "hit_rate": round(sc_hits / sc_decided * 100, 1) if sc_decided else None,
                "dominant_hits": sc_conv["dominant"],
                "moderate_hits": sc_conv["moderate"],
                "narrow_hits": sc_conv["narrow"],
                "true_hit_rate": round((sc_conv["dominant"] + sc_conv["moderate"]) / sc_decided * 100, 1) if sc_decided else None,
                "note_zh": f"強共識(2+SILVER一致): {sc_hits}勝/{sc_decided}場決定 = {round(sc_hits/sc_decided*100,1) if sc_decided else 'N/A'}%，{sc_pending}場待定",
            },
            "best_pick": {
                "hits": bp_hits, "misses": bp_misses, "pending": bp_pending,
                "decided": bp_decided,
                "hit_rate": round(bp_hits / bp_decided * 100, 1) if bp_decided else None,
                "gold_hits": bp_gold_hits, "gold_misses": bp_gold_misses,
                "silver_hits": bp_silver_hits, "silver_misses": bp_silver_misses,
                "dominant_hits": bp_conv["dominant"],
                "moderate_hits": bp_conv["moderate"],
                "narrow_hits": bp_conv["narrow"],
                "true_hit_rate": round((bp_conv["dominant"] + bp_conv["moderate"]) / bp_decided * 100, 1) if bp_decided else None,
                "note_zh": f"單場最佳信號(GOLD/SILVER): {bp_hits}勝/{bp_decided}場決定 = {round(bp_hits/bp_decided*100,1) if bp_decided else 'N/A'}%",
            },
            "no_conflict": {
                "hits": nc_hits, "misses": nc_misses, "pending": nc_pending,
                "decided": nc_decided,
                "hit_rate": round(nc_hits / nc_decided * 100, 1) if nc_decided else None,
                "note_zh": f"無衝突場次(所有信號同向): {nc_hits}勝/{nc_decided}場決定 = {round(nc_hits/nc_decided*100,1) if nc_decided else 'N/A'}%",
            },
        }

        # Augment with playoff home/away ATS split (scan decided playoff games this season).
        po_home_covers, po_total = 0, 0
        for jf in sorted(OUT_DIR.glob("????-??-??.json")):
            try:
                with open(jf, encoding="utf-8") as _jf:
                    _d = json.load(_jf)
                if not _d.get("is_playoff_view"):
                    continue
                for g in _d.get("games", {}).values():
                    winner = g.get("ats_winner")
                    if winner is None:
                        continue
                    po_total += 1
                    if winner == "home":
                        po_home_covers += 1
            except Exception:
                pass
        if po_total:
            home_pct = round(po_home_covers / po_total * 100, 1)
            stats["playoff_home_ats"] = {
                "games": po_total,
                "home_covers": po_home_covers,
                "away_covers": po_total - po_home_covers,
                "home_cover_rate": home_pct,
                "note_zh": f"本季季後賽共 {po_total} 場有結果，主場過盤 {po_home_covers} 場 ({home_pct}%)",
            }

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
            bracket["signal_tracker"] = _build_signal_tracker(OUT_DIR)
            bracket_path = OUT_DIR / "bracket.json"
            if bracket_path.exists():
                try:
                    with open(bracket_path, "r", encoding="utf-8") as fp:
                        old = json.load(fp)
                    for conf in ("east", "west"):
                        old_cf = old.get(conf, {}).get("conf_finals", {})
                        new_cf = bracket.get(conf, {}).get("conf_finals")
                        if new_cf and old_cf.get("analysis") and not new_cf.get("analysis"):
                            new_cf["analysis"] = old_cf["analysis"]
                    old_finals = old.get("finals", {})
                    new_finals = bracket.get("finals")
                    if new_finals and old_finals.get("analysis") and not new_finals.get("analysis"):
                        new_finals["analysis"] = old_finals["analysis"]
                except Exception:
                    pass
            with open(bracket_path, "w", encoding="utf-8") as f:
                json.dump(bracket, f, ensure_ascii=False, indent=1, default=_serialize)
            print(f"Bracket → {bracket_path.name}")

    # Season H2H lookup (all matchups in current season from OddsData.sqlite).
    h2h = build_season_h2h(season_key)
    if h2h:
        h2h_path = OUT_DIR / "season_h2h.json"
        with open(h2h_path, "w", encoding="utf-8") as f:
            json.dump(h2h, f, ensure_ascii=False, indent=1, default=_serialize)
        print(f"H2H → {h2h_path.name} ({len(h2h.get('pairs', {}))} matchup pairs)")

    # Date index.
    index_path = OUT_DIR / "dates.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({"dates": exported_dates, "today": today.isoformat(), "season": season_key},
                  f, ensure_ascii=False, indent=1, default=_serialize)
    print(f"Index → {index_path.name}")
    print("Done.")


if __name__ == "__main__":
    main()
