"""Production policy read from config.toml.

Two things live here so they are decided in one reviewed place instead of
being implied by file names or hard-coded thresholds:

* ``[production-models]`` — the exact model file the runner loads per kind.
* ``[ats-policy]``        — whether the ATS value flag may be shown at all.

Only the local pipeline reads this (Vercel deploys web/ without config.toml),
so anything the frontend needs is written into the per-game JSON.
"""
import datetime as _dt
from functools import lru_cache
from pathlib import Path

import toml

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "config.toml"

ATS_POLICY_DEFAULTS = {
    # Fail closed: with no [ats-policy] section the value flag stays off.
    "value_flag_enabled": False,
    "effective_from": "2026-10-02",
    "reopen_min_frozen_picks": 100,
    "reopen_ci_lower_pct": 52.4,
}


class PinnedModelMissing(RuntimeError):
    """A model pinned in config.toml is not on disk. Never fall back silently."""


@lru_cache(maxsize=1)
def _config():
    try:
        return toml.load(CONFIG_PATH)
    except FileNotFoundError:
        return {}


def reload():
    _config.cache_clear()


def pinned_model_name(kind):
    """File name pinned for ``kind`` ("ML" / "UO" / "ATS"), or None if not pinned."""
    name = (_config().get("production-models") or {}).get(f"xgb_{kind.lower()}")
    return name or None


def ats_policy():
    policy = dict(ATS_POLICY_DEFAULTS)
    policy.update(_config().get("ats-policy") or {})
    return policy


def _as_date(value):
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    return _dt.date.fromisoformat(str(value)[:10])


def ats_value_flag_active(game_date):
    """True when the ATS value flag may be shown for a game on ``game_date``.

    Games before ``effective_from`` keep whatever the rule produced, so
    re-exporting an old date does not rewrite what was shown at the time.
    """
    policy = ats_policy()
    if policy["value_flag_enabled"]:
        return True
    return _as_date(game_date) < _as_date(policy["effective_from"])


def apply_ats_policy(pred, game_date):
    """Gate ``pred["ats_is_value"]`` by the policy, in place.

    ``ats_shadow_value`` always records what the rule would have flagged, so
    the frozen pre-game record can still be scored against the re-open bar
    while nothing is shown as a recommendation.
    """
    pred["ats_shadow_value"] = bool(pred.get("ats_is_value"))
    if ats_value_flag_active(game_date):
        pred["ats_reference_only"] = False
    else:
        pred["ats_is_value"] = False
        pred["ats_reference_only"] = True
    return pred
