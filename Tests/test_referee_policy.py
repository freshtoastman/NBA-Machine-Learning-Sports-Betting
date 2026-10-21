"""[referee-policy] (config.toml, 2026-10-21): with bias_flag_enabled = false the
card keeps the crew names and raw cover rates but every badge input is off and a
reference-only note is attached. Tests load only the helper's source, not the
whole exporter (which imports main.py and the models)."""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SRC = (ROOT / "scripts" / "export_predictions.py").read_text(encoding="utf-8")


def _helper_namespace():
    tree = ast.parse(SRC)
    wanted = {"REFEREE_NOTE_ZH", "_referee_bias_flag_enabled", "build_referee_crew"}
    nodes = [n for n in tree.body
             if (isinstance(n, ast.FunctionDef) and n.name in wanted)
             or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in wanted for t in n.targets))]
    assert len(nodes) == 3
    ns = {"Path": Path, "__file__": str(ROOT / "scripts" / "export_predictions.py")}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "helpers", "exec"), ns)
    return ns


def _crew():
    return [
        {"name": "Simone Jelks", "games": 88, "home_cover_pct": 37.5, "significant": True, "direction": "away", "p_value": 0.025},
        {"name": "Ed Malloy", "games": 112, "home_cover_pct": 57.1, "significant": False, "direction": None},
    ]


def test_config_turns_bias_flag_off_from_2026_10_21():
    import toml
    pol = toml.load(ROOT / "config.toml")["referee-policy"]
    assert pol["bias_flag_enabled"] is False and pol["effective_from"] == "2026-10-21"
    assert _helper_namespace()["_referee_bias_flag_enabled"]() is False


def test_flag_off_keeps_names_and_rates_but_no_badges():
    ns = _helper_namespace()
    crew = _crew()
    out = ns["build_referee_crew"](crew, [crew[0]], bias_flag_enabled=False)
    assert [o["name"] for o in out["officials"]] == ["Simone Jelks", "Ed Malloy"]
    assert out["avg_home_cover_pct"] == 47.3
    assert out["has_significant"] is False and out["significant_officials"] == []
    assert all(o["significant"] is False and o["direction"] is None for o in out["officials"])
    assert out["away_friendly"] is False and out["home_friendly"] is False
    assert out["bias_flag_enabled"] is False and out["note"] == ns["REFEREE_NOTE_ZH"]


def test_flag_on_restores_previous_behaviour():
    ns = _helper_namespace()
    crew = _crew()
    out = ns["build_referee_crew"](crew, [crew[0]], bias_flag_enabled=True)
    assert out["has_significant"] is True and out["significant_officials"][0]["name"] == "Simone Jelks"
    assert out["officials"][0]["direction"] == "away" and out["note"] is None
    low = [{"name": "A", "games": 40, "home_cover_pct": 40.0, "significant": False, "direction": None}]
    assert ns["build_referee_crew"](low, [], bias_flag_enabled=True)["away_friendly"] is True
