"""Outbound-HTTP network policy: prefer IPv4.

Why (2026-10-28): cdn.nba.com (Akamai) publishes five AAAA records and this
machine's ISP routes IPv6 nowhere useful — every v6 connect sits until the
socket timeout. urllib3 tries the addresses in getaddrinfo order (all five v6
first), so one schedule fetch cost 5 × 30 s before the IPv4 address answered
in 20 ms: `Export_JSON` went from 15 s to 200+ s, `Patch_Scores` to 261 s, and
`Get_Data` to 64 s on the same day. stats.nba.com / ESPN / SBR are IPv4-only,
so forcing AF_INET costs nothing there.

Importing this module (src.Utils.tools and src.Utils.sbr_scoreboard do) flips
urllib3's `HAS_IPV6` for the whole process, which makes `allowed_gai_family()`
return AF_INET and keeps v6 addresses out of every `requests` connection.
Set NBA_ML_ALLOW_IPV6=1 to opt out (e.g. on a network with working v6).
"""
from __future__ import annotations

import os
import socket

import urllib3.util.connection as _u3_connection

# urllib3's own detection result, remembered on the urllib3 module so that a
# re-import of this module (tests, importlib.reload) still sees the real default.
if not hasattr(_u3_connection, "_nba_ml_has_ipv6_default"):
    _u3_connection._nba_ml_has_ipv6_default = _u3_connection.HAS_IPV6
_HAS_IPV6_DEFAULT: bool = _u3_connection._nba_ml_has_ipv6_default


def ipv4_only_enabled() -> bool:
    """Policy switch: on unless NBA_ML_ALLOW_IPV6=1."""
    return os.environ.get("NBA_ML_ALLOW_IPV6", "") != "1"


def prefer_ipv4(enabled: bool | None = None) -> bool:
    """Make urllib3 (and therefore `requests`) resolve IPv4 addresses only.

    Returns True when the override is active. `enabled=False` restores urllib3's
    own detection result.
    """
    if enabled is None:
        enabled = ipv4_only_enabled()
    _u3_connection.HAS_IPV6 = False if enabled else _HAS_IPV6_DEFAULT
    return not _u3_connection.HAS_IPV6


def effective_family() -> int:
    """The address family urllib3 will pass to getaddrinfo right now."""
    return _u3_connection.allowed_gai_family()


prefer_ipv4()

__all__ = ["prefer_ipv4", "ipv4_only_enabled", "effective_family", "socket"]
