"""patch_scores.main must finish even when a score source raises, and must not
query sources for a US date whose games have not tipped off yet.

2026-10-20 06:00 Taipei (opening day): stats.nba.com answered 500 for the
2026-10-20 slate (still 18:00 ET the day before), `_game_ids_for_date` raised
and the whole daily pipeline stopped before Export_JSON."""
import importlib.util
import sqlite3
import sys
from datetime import date
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("patch_scores", ROOT / "scripts" / "patch_scores.py")
patch_scores = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch_scores)


def _db_with_opener():
    con = sqlite3.connect(":memory:")
    con.execute('CREATE TABLE "2026-27" (Date DATE, Home TEXT, Away TEXT, OU REAL, Spread REAL, '
                "ML_Home INTEGER, ML_Away INTEGER, Points INTEGER, Win_Margin INTEGER, "
                "Days_Rest_Home INTEGER, Days_Rest_Away INTEGER)")
    con.execute('INSERT INTO "2026-27" VALUES ("2026-10-20","Detroit Pistons","Boston Celtics",'
                "221.5,-1.5,100,-118,0,0,7,7)")
    return con


def test_raising_source_does_not_abort_and_later_sources_still_run():
    con = _db_with_opener()
    calls = []

    def boom(con, d_str):
        calls.append("cdn")
        raise RuntimeError("500 Server Error")

    def espn(con, d_str):
        calls.append("espn")
        con.execute('UPDATE "2026-27" SET Points=212, Win_Margin=4 WHERE Date=?', (d_str,))
        return 1

    with mock.patch.object(patch_scores, "_patch_from_nba_live_scoreboard", return_value=0), \
         mock.patch.object(patch_scores, "_patch_from_sbr", return_value=0), \
         mock.patch.object(patch_scores, "_patch_from_nba_cdn", boom), \
         mock.patch.object(patch_scores, "_patch_from_espn", espn):
        total = patch_scores.main(con=con, today=date(2026, 10, 21), us_today=date(2026, 10, 20))
    assert calls == ["cdn", "espn"]
    assert total == 1
    assert con.execute('SELECT Points FROM "2026-27"').fetchone()[0] == 212


def test_future_us_date_is_skipped():
    con = _db_with_opener()
    with mock.patch.object(patch_scores, "_patch_from_nba_live_scoreboard") as live, \
         mock.patch.object(patch_scores, "_patch_from_sbr") as sbr, \
         mock.patch.object(patch_scores, "_patch_from_nba_cdn") as cdn, \
         mock.patch.object(patch_scores, "_patch_from_espn") as espn:
        # 06:00 Taipei 2026-10-20 is 18:00 ET 2026-10-19: the openers have not tipped off.
        total = patch_scores.main(con=con, today=date(2026, 10, 20), us_today=date(2026, 10, 19))
    assert total == 0
    for src in (live, sbr, cdn, espn):
        src.assert_not_called()


def test_cdn_headers_carry_browser_ua_and_referer():
    h = patch_scores.NBA_CDN_HEADERS
    assert "Mozilla" in h["User-Agent"] and h["Referer"].startswith("https://www.nba.com")


def _live_board(game_date, games):
    resp = mock.Mock()
    resp.raise_for_status = lambda: None
    resp.json = lambda: {"scoreboard": {"gameDate": game_date, "games": games}}
    return resp


def _final(game_et, home, away, hs, aws):
    return {"gameStatusText": "Final", "gameEt": game_et,
            "homeTeam": {"teamCity": home[0], "teamName": home[1], "score": hs},
            "awayTeam": {"teamCity": away[0], "teamName": away[1], "score": aws}}


def test_live_scoreboard_final_only_patches_its_own_et_date():
    # 2026-11-11: OKC @ POR played on consecutive nights. The 15:00 Taipei run on
    # 11/11 (07:00Z) saw the 11/10 final on "today's" board and wrote 123-103 onto
    # the unplayed 11/11 row, which the frozen record then graded.
    con = _db_with_opener()
    for d in ("2026-11-10", "2026-11-11"):
        con.execute('INSERT INTO "2026-27" VALUES (?,"Portland Trail Blazers","Oklahoma City Thunder",'
                    "225.5,-2.5,120,-142,0,0,1,1)", (d,))
    board = _live_board("2026-11-10", [
        _final("2026-11-10T23:00:00Z", ("Portland", "Trail Blazers"), ("Oklahoma City", "Thunder"), 123, 103),
    ])
    fake_date = mock.Mock()
    fake_date.today.return_value = date(2026, 11, 11)
    with mock.patch.object(patch_scores.requests, "get", return_value=board), \
         mock.patch.object(patch_scores, "date", fake_date):
        assert patch_scores._patch_from_nba_live_scoreboard(con, "2026-11-11") == 0
        assert patch_scores._patch_from_nba_live_scoreboard(con, "2026-11-10") == 1
    rows = dict(con.execute('SELECT Date, Points FROM "2026-27" WHERE Home="Portland Trail Blazers"').fetchall())
    assert rows == {"2026-11-10": 226, "2026-11-11": 0}
