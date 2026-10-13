import importlib.util
import sqlite3
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest import mock

import pandas as pd

GET_DATA_PATH = Path(__file__).resolve().parents[1] / "src" / "Process-Data" / "Get_Data.py"
SPEC = importlib.util.spec_from_file_location("get_data", GET_DATA_PATH)
get_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(get_data)


class TestGetData(unittest.TestCase):

    def test_iter_dates_inclusive(self):
        dates = list(get_data.iter_dates(date(2025, 1, 1), date(2025, 1, 3)))
        self.assertEqual(dates, [date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3)])

    def test_select_current_season(self):
        config = {
            "get-data": {
                "2025-26": {
                    "start_date": "2025-10-01",
                    "end_date": "2026-06-01",
                    "start_year": "2025",
                    "end_year": "2026"
                }
            }
        }
        season_key, value, start_date, end_date = get_data.select_current_season(
            config, date(2025, 12, 1)
        )
        self.assertEqual(season_key, "2025-26")
        self.assertEqual(value["start_year"], "2025")
        self.assertEqual(start_date, date(2025, 10, 1))
        self.assertEqual(end_date, date(2026, 6, 1))

    def test_get_table_dates_ignores_non_dates(self):
        with sqlite3.connect(":memory:") as con:
            con.execute('CREATE TABLE "2025-01-01" (id INTEGER)')
            con.execute('CREATE TABLE "not_a_date" (id INTEGER)')
            table_dates = get_data.get_table_dates(con)
        self.assertEqual(table_dates, [date(2025, 1, 1)])

    def test_fetch_data_retries_until_non_empty(self):
        df_empty = pd.DataFrame(data={})
        df_ok = pd.DataFrame({"A": [1]})
        with mock.patch.object(get_data, "get_json_data", return_value={"resultSets": []}) as mock_json, \
             mock.patch.object(get_data, "to_data_frame", side_effect=[df_empty, df_ok]), \
             mock.patch.object(get_data.time, "sleep"):
            result = get_data.fetch_data("http://example.com", date(2025, 1, 1), "2025", "2025-26")
        self.assertFalse(result.empty)
        self.assertEqual(mock_json.call_count, 2)

    def test_main_fetches_only_new_dates(self):
        config = {
            "data_url": "http://example.com",
            "get-data": {
                "2025-26": {
                    "start_date": "2025-10-01",
                    "end_date": "2025-10-03",
                    "start_year": "2025",
                    "end_year": "2026"
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            with sqlite3.connect(db_path) as con:
                con.execute('CREATE TABLE "2025-10-01" (id INTEGER)')

            called_dates = []

            def fake_fetch_data(url, date_pointer, start_year, season_key):
                called_dates.append(date_pointer)
                return pd.DataFrame({"A": [1]})

            with mock.patch.object(get_data, "fetch_data", side_effect=fake_fetch_data), \
                 mock.patch.object(get_data.time, "sleep"):
                get_data.main(config=config, db_path=db_path, today=date(2025, 10, 3))

            self.assertEqual(called_dates, [date(2025, 10, 2), date(2025, 10, 3)])

    def _refresh_config(self):
        return {
            "data_url": "http://example.com",
            "get-data": {
                "2026-27": {
                    "start_date": "2026-10-01",
                    "end_date": "2027-06-30",
                    "start_year": "2026",
                    "end_year": "2027"
                }
            }
        }

    def _run_refresh(self, now_utc, refresh_latest=True):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            with sqlite3.connect(db_path) as con:
                con.execute('CREATE TABLE "2026-11-04" (id INTEGER)')
                con.execute('CREATE TABLE "2026-11-05" (id INTEGER)')

            called_dates = []

            def fake_fetch_data(url, date_pointer, start_year, season_key):
                called_dates.append(date_pointer)
                return pd.DataFrame({"A": [1]})

            with mock.patch.object(get_data, "fetch_data", side_effect=fake_fetch_data), \
                 mock.patch.object(get_data.time, "sleep"):
                get_data.main(config=self._refresh_config(), db_path=db_path,
                              today=date(2026, 11, 5), refresh_latest=refresh_latest,
                              now_utc=now_utc)
            logged = []
            if called_dates:
                with sqlite3.connect(db_path) as con:
                    logged = con.execute('SELECT table_name FROM "_fetch_log"').fetchall()
        return called_dates, logged

    def test_refresh_latest_refetches_today_before_noon_et(self):
        # 15:00 Taipei on 11/05 = 02:00 ET on 11/05 (EST): yesterday's games are final.
        called, logged = self._run_refresh(datetime(2026, 11, 5, 7, 0, tzinfo=timezone.utc))
        self.assertEqual(called, [date(2026, 11, 5)])
        self.assertEqual(logged, [("2026-11-05",)])

    def test_refresh_latest_skips_after_noon_et(self):
        # 02:00 Taipei on 11/06 = 13:00 ET on 11/05: a matinee may already be final.
        called, _ = self._run_refresh(datetime(2026, 11, 5, 18, 0, tzinfo=timezone.utc))
        self.assertEqual(called, [])

    def test_without_refresh_latest_today_is_not_refetched(self):
        called, _ = self._run_refresh(datetime(2026, 11, 5, 7, 0, tzinfo=timezone.utc),
                                      refresh_latest=False)
        self.assertEqual(called, [])


class TestPregameRefreshWindow(unittest.TestCase):

    def setUp(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from src.Utils.tools import pregame_refresh_allowed
        self.allowed = pregame_refresh_allowed

    def test_edt_afternoon_taipei_is_allowed(self):
        # 2026-10-21 15:00 Taipei = 03:00 EDT on 10/21
        self.assertTrue(self.allowed(date(2026, 10, 21), datetime(2026, 10, 21, 7, 0, tzinfo=timezone.utc)))

    def test_est_afternoon_taipei_is_allowed(self):
        # 2026-12-25 15:00 Taipei = 02:00 EST on 12/25 (noon tip-offs later that day)
        self.assertTrue(self.allowed("2026-12-25", datetime(2026, 12, 25, 7, 0, tzinfo=timezone.utc)))

    def test_noon_et_closes_window(self):
        self.assertFalse(self.allowed(date(2026, 12, 25), datetime(2026, 12, 25, 17, 0, tzinfo=timezone.utc)))
        self.assertTrue(self.allowed(date(2026, 12, 25), datetime(2026, 12, 25, 16, 59, tzinfo=timezone.utc)))

    def test_previous_et_day_is_allowed(self):
        # morning Taipei run: 06:00 Taipei 10/21 = 18:00 EDT 10/20, table 10/21
        self.assertTrue(self.allowed(date(2026, 10, 21), datetime(2026, 10, 20, 22, 0, tzinfo=timezone.utc)))


if __name__ == "__main__":
    unittest.main()
