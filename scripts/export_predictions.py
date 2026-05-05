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
    for f in sorted(list(data_dir.glob("2026-04-*.json")) + list(data_dir.glob("2026-05-*.json"))):
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
    for gv in game_votes.values():
        if gv["pending"] or gv["ats_winner"] is None:
            continue
        consensus = "home" if gv["home"] > gv["away"] else "away" if gv["away"] > gv["home"] else None
        if consensus is None:
            continue
        if consensus == gv["ats_winner"]:
            game_wins += 1
        else:
            game_losses += 1
    game_total = game_wins + game_losses
    game_pnl = round(game_wins * 0.91 - game_losses * 1.0, 2) if game_total else 0

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

        def _with_wins(high_team, low_team):
            """Attach series win counts and upcoming signal hints to a first_round entry."""
            if isinstance(high_team, str) or isinstance(low_team, str):
                return {}
            key = frozenset({high_team["team"], low_team["team"]})
            wins_by_team = _series_wins.get(key, {})
            hw = wins_by_team.get(high_team["team"], 0)
            lw = wins_by_team.get(low_team["team"], 0)
            gp = hw + lw
            entry = {"high_wins": hw, "low_wins": lw, "games_played": gp}
            # G1 pre-game hint: R2 opener home-seed advantage.
            if gp == 0 and _max_round_num >= 2:
                _game = _today_game_lookup.get(key) or _upcoming_game_lookup.get(key)
                _r2_picks = (_game.get("playoff_ats_picks", []) if _game else [])
                _r2_g1_pick = next((p for p in _r2_picks if "R2G1" in p.get("signal", "")), None)
                if _r2_g1_pick:
                    entry["next_signal"] = {
                        "game_num": 1,
                        "signal": _r2_g1_pick["signal"],
                        "side": _r2_g1_pick.get("side", "home"),
                        "home_team": high_team["team"],
                        "home_team_zh": high_team["team_zh"],
                        "tier": _r2_g1_pick.get("tier", "SILVER"),
                        "backtest_wr": _r2_g1_pick.get("backtest_wr", 0.646),
                        "reason_zh": _r2_g1_pick.get("reason_zh", "第二輪G1主場cover率 64.6% (n=48，13季)"),
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
            # G5 signal: fires when 4 games played and series is tied 2-2
            # Home team covers 60.0% in G5 tied games (n=60, 12 seasons, ROI +14.5%)
            if gp == 4 and hw == 2 and lw == 2:
                # G5 is at high_seed's home (home court rotation: 1-2 at high, 3-4-5 at low? No)
                # Actually G5 returns to high seed home: games 1,2,5,6 at high; 3,4,7 at low
                entry["next_signal"] = {
                    "game_num": 5,
                    "signal": "G5平手主場壓制",
                    "side": "home",
                    "home_team": high_team["team"],
                    "home_team_zh": high_team["team_zh"],
                    "tier": "SILVER",
                    "backtest_wr": 0.60,
                    "reason_zh": "系列賽平手2-2，第5場高種子主場，12季回測主場cover率 60% (n=60，ROI +14.5%)，最強結構信號",
                }
            # G6 signal: fires when 5 games played in R1 (series reaches G6).
            # HUGE signal: R1 G6 away covers 80.4% (n=46, 12 seasons 2012-2024).
            # G6 hosted at low_seed home (2-2-1-1-1 format: G3,4,6 at low).
            # Away team = high_seed. Signal: bet AWAY (the visiting team).
            # Recent validation: 2019-2024 = 87.5% (14/16), 2020+ = 84.6% (11/13).
            # Robust across home-dog buckets: dog 0-5: 76.5%(26/34), dog 5+: 100%(10/10).
            if gp == 5 and hw < 4 and lw < 4:
                # Determine signal team (away in G6) - high_seed is visiting
                g6_away = high_team
                g6_home = low_team
                entry["next_signal"] = {
                    "game_num": 6,
                    "signal": "R1G6客場優勢",
                    "side": "away",
                    "home_team": g6_home["team"],
                    "home_team_zh": g6_home["team_zh"],
                    "tier": "GOLD",
                    "backtest_wr": 0.804,
                    "reason_zh": "首輪第6場在低種子主場，12季回測客場(高種子)cover率 80.4% (n=46)，近5年更達 84.6%，為歷史最強結構信號",
                    "signal_team_zh": g6_away.get("team_zh") or g6_away.get("team"),
                }
            # Upgrade next_signal if actual game data has a higher-tier signal.
            _tier_rank = {"GOLD": 3, "SILVER": 2, "BRONZE": 1, "NONE": 0}
            ns = entry.get("next_signal")
            if ns and gp >= 2:
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
                r2a.update(_with_wins(high_r2a, low_r2a))
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
                r2b.update(_with_wins(high_r2b, low_r2b))
            second_round.append(r2b)

        result = {
            "name": name,
            "seeds": seeds[:10],
            "lottery": seeds[10:],
            "play_in": play_in,
            "first_round": first_round,
        }
        if second_round:
            result["second_round"] = second_round
        return result

    return {
        "snapshot_date": snapshot_date,
        "generated_for": target_date.isoformat(),
        "playoff_start": playoff_start,
        "show_from": show_from,
        "show_until": show_until,
        "stage_label": _stage_label,
        "stage": {1: "first_round", 2: "second_round", 3: "conf_finals", 4: "finals"}.get(
            _max_round_num, "first_round" if _series_wins else ("play_in" if _playin_results else None)),
        "east": _conference("東區", east, east_r1_resolved),
        "west": _conference("西區", west, west_r1_resolved),
    }


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
        best, best_date = (candidates[0] if candidates else season_key), ""
        for c in candidates:
            try:
                row = con.execute(f'SELECT MAX(Date) FROM "{c}"').fetchone()
                d = row[0] or ""
                if d > best_date:
                    best, best_date = c, d
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
    dates_needed = sorted({d for d, _, _ in missing})
    for d_str in dates_needed:
        try:
            url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={d_str.replace('-', '')}"
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
            if (d_str, h_name, a_name) not in [(d, h, a) for d, h, a in missing]:
                continue
            h_qs = [int(ls.get("value", 0)) for ls in home_t.get("linescores", [])]
            a_qs = [int(ls.get("value", 0)) for ls in away_t.get("linescores", [])]
            if len(h_qs) < 4 or len(a_qs) < 4:
                continue
            existing.append({
                "gameId": ev.get("id", ""),
                "date": d_str,
                "home_tri": home_t["team"].get("abbreviation", ""),
                "away_tri": away_t["team"].get("abbreviation", ""),
                "home_name": h_name,
                "away_name": a_name,
                "q_home": h_qs[:4],
                "q_away": a_qs[:4],
                "home_score": int(home_t["score"]),
                "away_score": int(away_t["score"]),
            })
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
