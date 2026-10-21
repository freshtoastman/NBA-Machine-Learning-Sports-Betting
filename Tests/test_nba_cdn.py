"""cdn.nba.com schedule/boxscore helper (src/Utils/nba_cdn) and the two scripts
that fall back to it now that stats.nba.com scoreboardv2 answers 500."""
import importlib.util
import sqlite3
import sys
from datetime import date
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.Utils import nba_cdn  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SCHEDULE = {"leagueSchedule": {"seasonYear": "2026-27", "gameDates": [
    {"gameDate": "10/10/2026 00:00:00", "games": [
        {"gameId": "0012600006", "gameStatus": 3,
         "homeTeam": {"teamId": 1, "teamCity": "Indiana", "teamName": "Pacers", "score": 100},
         "awayTeam": {"teamId": 2, "teamCity": "Atlanta", "teamName": "Hawks", "score": 90}}]},
    {"gameDate": "10/20/2026 00:00:00", "games": [
        {"gameId": "0022600001", "gameStatus": 3,
         "homeTeam": {"teamId": 1610612765, "teamCity": "Detroit", "teamName": "Pistons", "score": 121},
         "awayTeam": {"teamId": 1610612738, "teamCity": "Boston", "teamName": "Celtics", "score": 113}},
        {"gameId": "0022600003", "gameStatus": 1,
         "homeTeam": {"teamId": 1610612746, "teamCity": "Los Angeles", "teamName": "Clippers", "score": 0},
         "awayTeam": {"teamId": 1610612759, "teamCity": "San Antonio", "teamName": "Spurs", "score": 0}}]},
]}}

BOX = {"game": {"gameId": "0022600001", "gameStatus": 3, "gameStatusText": "Final",
                "homeTeam": {"teamCity": "Detroit", "teamName": "Pistons", "score": 121},
                "awayTeam": {"teamCity": "Boston", "teamName": "Celtics", "score": 113},
                "officials": [{"personId": 202041, "name": "Kevin Scott", "assignment": "OFFICIAL1"},
                              {"personId": 2530, "name": "Ed Malloy", "assignment": "OFFICIAL2"},
                              {"personId": 1626302, "name": "Aaron Smith", "assignment": "OFFICIAL3"}]}}


class _Resp:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"{self.status_code}")

    def json(self):
        return self._payload


def setup_function(_):
    nba_cdn._schedule_cache = None


def _cdn_only(payload):
    """requests.get stand-in: cdn.nba.com gets `payload`, stats.nba.com is a failure."""
    def get(url, *a, **k):
        if "cdn.nba.com" in url:
            return _Resp(payload)
        raise AssertionError(f"stats.nba.com must not be called: {url}")
    return get


def test_schedule_is_keyed_by_iso_date_drops_preseason_and_is_fetched_once():
    with mock.patch.object(nba_cdn.requests, "get", return_value=_Resp(SCHEDULE)) as get:
        games = nba_cdn.games_for_date(date(2026, 10, 20))
        assert [g["game_id"] for g in games] == ["0022600001", "0022600003"]
        assert games[0]["home"] == "Detroit Pistons" and games[0]["status"] == 3
        assert games[1]["home"] == "LA Clippers"  # Los Angeles Clippers normalised
        assert nba_cdn.games_for_date(date(2026, 10, 10)) == []  # preseason id 001...
        assert len(nba_cdn.games_for_date(date(2026, 10, 10), season_only=False)) == 1
        assert nba_cdn.games_for_date(date(2026, 10, 21)) == []
        assert get.call_count == 1
        assert get.call_args.kwargs["headers"]["Referer"].startswith("https://www.nba.com")


def test_boxscore_returns_officials_and_none_before_tipoff():
    with mock.patch.object(nba_cdn.requests, "get", return_value=_Resp(BOX)):
        box = nba_cdn.fetch_boxscore("0022600001")
    assert box["home_score"] == 121 and box["away"] == "Boston Celtics"
    assert box["officials"] == [(202041, "Kevin Scott"), (2530, "Ed Malloy"), (1626302, "Aaron Smith")]
    with mock.patch.object(nba_cdn.requests, "get", return_value=_Resp({}, status=403)):
        assert nba_cdn.fetch_boxscore("0022600004") is None


def test_patch_scores_game_ids_use_schedule_and_skip_unfinished():
    patch_scores = _load("patch_scores")
    with mock.patch.object(nba_cdn.requests, "get", side_effect=_cdn_only(SCHEDULE)):
        assert patch_scores._game_ids_for_date(date(2026, 10, 20)) == ["0022600001"]


def test_patch_scores_falls_back_to_scoreboardv2_when_schedule_fails():
    patch_scores = _load("patch_scores")
    sb = {"resultSets": [{"name": "GameHeader", "headers": ["GAME_ID"], "rowSet": [["0022600099"]]}]}
    with mock.patch.object(nba_cdn, "games_for_date", side_effect=RuntimeError("cdn down")), \
         mock.patch.object(patch_scores.requests, "get", return_value=_Resp(sb)):
        assert patch_scores._game_ids_for_date(date(2026, 10, 20)) == ["0022600099"]


def test_referee_collector_uses_cdn_for_ids_and_officials():
    ref = _load("collect_referee_data")
    with mock.patch.object(nba_cdn.requests, "get", side_effect=_cdn_only(SCHEDULE)):
        ids = ref._get_game_ids_for_date(date(2026, 10, 20))
    assert ids == [("0022600001", 1610612765, 1610612738)]  # finished game only
    with mock.patch.object(nba_cdn.requests, "get", side_effect=_cdn_only(BOX)):
        assert ref._fetch_game_officials("0022600001")[0] == (202041, "Kevin Scott")


def test_referee_backfill_fills_only_null_season_games():
    ref = _load("collect_referee_data")
    con_ref = sqlite3.connect(":memory:")
    ref._init_db(con_ref)
    con_ref.executemany(
        "INSERT INTO referee_games (game_id, game_date, home_team, away_team, official1_name) VALUES (?,?,?,?,?)",
        [("0022500001", "2025-10-21", "Oklahoma City Thunder", "Houston Rockets", None),
         ("0012600006", "2026-10-10", "Indiana Pacers", "Atlanta Hawks", None),      # preseason: skipped
         ("0022500002", "2025-10-21", "Los Angeles Lakers", "Golden State Warriors", "Zach Zarba")])
    con_odds = sqlite3.connect(":memory:")
    con_odds.execute('CREATE TABLE "2025-26" (Date, Home, Away, Spread REAL, Win_Margin REAL)')
    con_odds.execute('INSERT INTO "2025-26" VALUES ("2025-10-21","Oklahoma City Thunder","Houston Rockets",-6.5,1)')
    fetched = []

    def officials(game_id):
        fetched.append(game_id)
        return [(1, "A One"), (2, "B Two"), (3, "C Three")]

    with mock.patch.object(ref, "_fetch_game_officials", officials):
        assert ref.backfill_null_officials(con_ref, con_odds, sleep_s=0) == 1
    assert fetched == ["0022500001"]
    row = con_ref.execute("SELECT official1_name, official3_name, spread, home_covered FROM referee_games "
                          "WHERE game_id='0022500001'").fetchone()
    assert row == ("A One", "C Three", -6.5, 1)
