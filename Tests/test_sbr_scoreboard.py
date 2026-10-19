"""src/Utils/sbr_scoreboard must survive what broke sbrscrape on 2026-10-19:
SBR answering 403 to a python user agent. Every request carries a browser UA and a
timeout, a 403 degrades to ``games == []`` (never an AttributeError), and a 200
parses into the sbrscrape-shaped dicts Get_Odds_Data / update_odds / patch_scores
read (``away_spread['fanduel']``, ``total['fanduel']``, ``home_ml`` ...)."""
import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.Utils import sbr_scoreboard as sbr  # noqa: E402


def _row(game_id, home, away, status="7:00 PM", hs=0, as_=0, **line):
    return {
        "gameView": {
            "gameId": game_id, "startDate": "2026-10-20T23:00:00+00:00", "gameStatusText": status,
            "homeTeam": {"fullName": home, "displayName": home.split()[0], "shortName": home[:3].upper(), "rank": 0},
            "awayTeam": {"fullName": away, "displayName": away.split()[0], "shortName": away[:3].upper(), "rank": 0},
            "homeTeamScore": hs, "awayTeamScore": as_,
        },
        "oddsViews": [None, {"sportsbook": "fanduel", "currentLine": line, "openingLine": line}],
    }


def _payload(rows):
    return {"pageProps": {"oddsTables": [{"oddsTableModel": {"gameRows": rows}}]}}


class FakeResponse:
    def __init__(self, status=200, text="", payload=None):
        self.status_code = status
        self.text = text
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        return self._payload


class FakeSession:
    """Routes by URL; records every call so the test can check timeout/headers."""

    def __init__(self, routes, status_for_page=200):
        self.routes = routes
        self.calls = []
        self.headers = dict(sbr.HEADERS)
        self.status_for_page = status_for_page

    def get(self, url, timeout=None, **kw):
        self.calls.append((url, timeout))
        if "/_next/data/" not in url:
            page = '<script id="__NEXT_DATA__" type="application/json">{"buildId": "B1"}</script>'
            return FakeResponse(self.status_for_page, text=page)
        for key, resp in self.routes.items():
            if key in url:
                return resp
        return FakeResponse(404)


def setup_function(_):
    sbr._build_id_cache.clear()


def test_headers_are_a_browser_not_python_requests():
    assert sbr.HEADERS["User-Agent"].startswith("Mozilla/5.0")
    assert "python-requests" not in sbr.make_session().headers["User-Agent"]
    assert sbr.REQUEST_TIMEOUT_SECONDS <= 60


def test_403_on_page_gives_empty_games_not_attribute_error():
    sess = FakeSession({}, status_for_page=403)
    sb = sbr.Scoreboard(date="2026-10-20", session=sess)
    assert sb.games == []
    assert "403" in sb.error
    assert all(t == sbr.REQUEST_TIMEOUT_SECONDS for _, t in sess.calls)


def test_parses_opening_night_into_sbrscrape_shape():
    spread = _row(1, "Detroit Pistons", "Boston Celtics", homeSpread=-1.5, awaySpread=1.5, homeOdds=-102, awayOdds=-120)
    ml = _row(1, "Detroit Pistons", "Boston Celtics", homeOdds=-108, awayOdds=-108)
    tot = _row(1, "Detroit Pistons", "Boston Celtics", total=221.5, overOdds=-118, underOdds=-104)
    sess = FakeSession({
        "/money-line/": FakeResponse(payload=_payload([ml])),
        "/totals/": FakeResponse(payload=_payload([tot])),
        "nba-basketball.json": FakeResponse(payload=_payload([spread])),
    })
    sb = sbr.Scoreboard(date="2026-10-20", session=sess)
    assert sb.error is None
    assert len(sb.games) == 1
    g = sb.games[0]
    assert (g["home_team"], g["away_team"]) == ("Detroit Pistons", "Boston Celtics")
    assert g["away_spread"]["fanduel"] == 1.5 and g["home_spread"]["fanduel"] == -1.5
    assert g["total"]["fanduel"] == 221.5
    assert g["home_ml"]["fanduel"] == -108 and g["away_ml"]["fanduel"] == -108
    assert g["home_score"] == 0 and g["status"] == "7:00 PM"
    # Get_Odds_Data.append_game_rows reads exactly these keys
    for key in ("total", "away_spread", "home_ml", "away_ml", "away_score", "home_score"):
        assert key in g
    # one page fetch (buildId) + three JSON fetches, all with a timeout
    assert len(sess.calls) == 4
    assert all(t == sbr.REQUEST_TIMEOUT_SECONDS for _, t in sess.calls)


def test_build_id_cached_across_dates_and_refreshed_on_stale():
    ok = FakeResponse(payload=_payload([]))
    sess = FakeSession({"nba-basketball.json": ok})
    sbr.Scoreboard(date="2026-10-20", session=sess)
    sbr.Scoreboard(date="2026-10-21", session=sess)
    pages = [u for u, _ in sess.calls if "/_next/data/" not in u]
    assert len(pages) == 1, "buildId must be fetched once per process, not once per date"

    # stale buildId: the JSON endpoint 404s once, then the page is re-read and retried
    sbr._build_id_cache.clear()
    seq = iter([FakeResponse(404), ok])

    class Stale(FakeSession):
        def get(self, url, timeout=None, **kw):
            if "nba-basketball.json" in url:
                self.calls.append((url, timeout))
                return next(seq)
            return super().get(url, timeout=timeout)

    s2 = Stale({})
    sb = sbr.Scoreboard(date="2026-10-20", session=s2)
    assert sb.error is None and sb.games == []
    assert len([u for u, _ in s2.calls if "/_next/data/" not in u]) == 2


def test_accepts_date_objects_like_get_odds_data_passes():
    from datetime import date
    sess = FakeSession({"nba-basketball.json": FakeResponse(payload=_payload([]))})
    sbr.Scoreboard(date=date(2026, 10, 20), session=sess)
    assert any("date=2026-10-20" in u for u, _ in sess.calls)


def test_fixture_round_trips_json():
    # guard against the fake payload drifting from what the parser expects
    json.dumps(_payload([_row(1, "A B", "C D", homeSpread=-1.0)]))
