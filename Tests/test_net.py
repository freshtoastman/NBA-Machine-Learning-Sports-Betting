"""Outbound requests must not try cdn.nba.com's IPv6 addresses first: on this
network every v6 connect times out (5 × 30 s) before the IPv4 address answers,
which turned a 15 s export into 200+ s on 2026-10-28. Importing src.Utils.net
(pulled in by tools and sbr_scoreboard) must leave urllib3 on AF_INET.

The fresh-process checks run a child interpreter because `from . import net`
inside an already-imported package does not re-run the module."""
import os
import socket
import subprocess
import sys
from pathlib import Path

import urllib3.util.connection as u3c

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _fresh(code: str, **env_extra) -> str:
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    env.pop("NBA_ML_ALLOW_IPV6", None)
    env.update(env_extra)
    out = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env,
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_import_forces_ipv4_in_fresh_process():
    out = _fresh(
        "import socket, src.Utils.net as n, urllib3.util.connection as u;"
        "print(u.HAS_IPV6, u.allowed_gai_family() == socket.AF_INET, n.effective_family() == socket.AF_INET)"
    )
    assert out == "False True True"


def test_pipeline_entry_modules_pull_the_policy_in():
    for mod in ("src.Utils.tools", "src.Utils.sbr_scoreboard", "src.Utils.nba_cdn"):
        out = _fresh(f"import {mod}, urllib3.util.connection as u; print(u.HAS_IPV6)")
        assert out == "False", mod


def test_env_opt_out_keeps_urllib3_detection():
    out = _fresh(
        "import urllib3.util.connection as u; d = u.HAS_IPV6;"
        "import src.Utils.net as n; print(u.HAS_IPV6 == d, n.ipv4_only_enabled())",
        NBA_ML_ALLOW_IPV6="1",
    )
    assert out == "True False"


def test_prefer_ipv4_toggle_in_process():
    from src.Utils import net
    assert net.prefer_ipv4(enabled=False) is False
    assert u3c.HAS_IPV6 == net._HAS_IPV6_DEFAULT
    assert net.prefer_ipv4(enabled=True) is True
    assert u3c.HAS_IPV6 is False
    assert net.effective_family() == socket.AF_INET
