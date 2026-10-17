"""get_json_data must never hang: every request carries a timeout, and a timeout
or connection error degrades to the same empty result a bad JSON body gives, so
the fetchers' retry loops (Get_Data / Get_Advanced_Data / Get_Playoffs_Data)
handle it instead of the whole run stalling on one socket."""
import sys
from pathlib import Path
from unittest import mock

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.Utils import tools  # noqa: E402


def test_get_json_data_passes_timeout():
    resp = mock.Mock()
    resp.json.return_value = {"resultSets": [{"headers": ["A"], "rowSet": [[1]]}]}
    with mock.patch.object(tools.requests, "get", return_value=resp) as get:
        out = tools.get_json_data("http://x")
    assert out == [{"headers": ["A"], "rowSet": [[1]]}]
    assert get.call_args.kwargs["timeout"] == tools.REQUEST_TIMEOUT_SECONDS
    assert tools.REQUEST_TIMEOUT_SECONDS <= 60


def test_get_json_data_timeout_returns_empty():
    with mock.patch.object(tools.requests, "get", side_effect=requests.exceptions.ReadTimeout("slow")):
        out = tools.get_json_data("http://x")
    assert out == {}
    # and the downstream frame builder treats it like any other empty answer
    assert tools.to_data_frame(out).empty


def test_get_json_data_connection_error_returns_empty():
    with mock.patch.object(tools.requests, "get", side_effect=requests.exceptions.ConnectionError("reset")):
        assert tools.get_json_data("http://x") == {}


def test_get_todays_games_json_passes_timeout():
    resp = mock.Mock()
    resp.json.return_value = {"gs": {"g": [1, 2]}}
    with mock.patch.object(tools.requests, "get", return_value=resp) as get:
        assert tools.get_todays_games_json("http://x") == [1, 2]
    assert get.call_args.kwargs["timeout"] == tools.REQUEST_TIMEOUT_SECONDS
