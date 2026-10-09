"""Pre-game prediction freeze for the exported web/data/YYYY-MM-DD.json files.

The daily export rewrites every file inside its look-back window, so until now a
game's prediction fields were recomputed *after* the game with post-game data,
odds and whatever model was pinned that day. The tracker therefore never graded
what a visitor actually saw before tip-off (scripts/audit_live_record.py has to
dig the pre-game state out of git history).

This module keeps a ``pregame`` block per game inside the exported JSON:

* while the game has not started, each export refreshes the block (the lines
  still move and the pre-game features still improve) — the frozen prediction
  is the **last one shown before tip-off**, the same definition the git audit
  uses;
* the moment an export sees the game started (live score or final score) the
  block is locked and never touched again. From then on the visible prediction
  fields of the game are overwritten from the block, and the result fields
  (ml_correct / ats_winner / ou_correct …) are graded against the frozen
  spread, total and moneyline — not against closing numbers or a re-run model;
* a game whose first export already has a result gets ``pregame = None`` and
  ``pregame_missing = True``; the frozen-record tracker skips it instead of
  pretending a pick existed.

Everything here is pure (dict in, dict out) so it can be unit-tested without
the models or the databases.
"""
from __future__ import annotations

import copy
from datetime import datetime
from typing import Iterable

# Prediction fields that must not change once the game has started.
FROZEN_FIELDS: tuple[str, ...] = (
    # moneyline model
    "winner", "home_confidence", "away_confidence",
    "home_team_odds", "away_team_odds", "home_team_ev", "away_team_ev",
    "home_kelly", "away_kelly",
    "value_side", "value_edge", "value_ev", "is_value", "is_golden",
    "ml_shadow_value", "ml_shadow_golden", "ml_value_reference_only",
    "consensus_pick", "is_consensus",
    # totals model
    "ou_pick", "ou_value", "ou_confidence",
    # spread model
    "spread", "ats_model_pick", "ats_model_home_prob", "ats_model_confidence",
    "ats_value_edge", "ats_is_value", "ats_shadow_value", "ats_reference_only",
    "ats_away_quality_filter",
    # playoff strategy signals + the series state they were computed from
    "playoff_ats_picks", "playoff_ats_best_side", "playoff_ats_best_signal",
    "playoff_ats_best_tier", "playoff_ats_consensus", "playoff_ats_has_conflict",
    "playoff_ats_strong_consensus",
    "series_game_num", "series_home_wins", "series_away_wins", "series_round",
    "round_num", "series_status_text", "series_is_elimination",
    # line movement seen before tip-off
    "spread_first", "spread_last", "spread_move", "ou_first", "ou_last", "ou_move",
)

# Closing-number fields that are kept beside the frozen ones for comparison.
_CLOSING_FIELDS: tuple[str, ...] = ("spread", "ou_value", "ats_winner", "ats_cover_margin",
                                    "home_team_odds", "away_team_odds")


def _has_score(g: dict) -> bool:
    hs, as_ = g.get("home_score"), g.get("away_score")
    return hs is not None and as_ is not None


def game_started(g: dict) -> bool:
    """True once the export has seen the game in progress or finished."""
    if _has_score(g):
        return True
    if g.get("live_status"):
        return True
    return bool(g.get("live_is_final"))


def snapshot(g: dict, now_iso: str, revisions: int = 1) -> dict:
    """Build an (unlocked) pregame block from the current prediction fields."""
    block = {"frozen_at": now_iso, "revisions": revisions, "locked": False}
    for f in FROZEN_FIELDS:
        if f in g:
            block[f] = copy.deepcopy(g[f])
    return block


def overlay(g: dict, block: dict) -> dict:
    """Overwrite the prediction fields of ``g`` with the frozen ones, in place.

    The post-game values that would have been shown instead are kept under
    ``closing`` so the difference stays visible (and auditable).
    """
    closing = {}
    for f in _CLOSING_FIELDS:
        if f in g:
            closing[f] = g[f]
    g["closing"] = closing
    for f in FROZEN_FIELDS:
        if f in block:
            g[f] = copy.deepcopy(block[f])
        elif f in g and f not in ("series_game_num", "series_home_wins", "series_away_wins",
                                  "series_round", "round_num", "series_status_text",
                                  "series_is_elimination"):
            # A prediction field the frozen snapshot never had (model did not run
            # pre-game) must not appear after the fact.
            g[f] = None
    return g


def grade(g: dict) -> dict:
    """Re-grade the result fields of ``g`` against its (frozen) lines, in place."""
    if not _has_score(g):
        return g
    hs, as_ = int(g["home_score"]), int(g["away_score"])
    margin = hs - as_
    total = hs + as_
    g["actual_total"] = total
    g["actual_home_win"] = margin > 0
    g["actual_winner"] = "home" if margin > 0 else "away"
    winner = g.get("winner")
    g["ml_correct"] = (winner == g["actual_winner"]) if winner in ("home", "away") else None

    spread = g.get("spread")
    if spread is None:
        g["ats_winner"], g["ats_cover_margin"] = None, None
    else:
        diff = margin - float(spread)
        if abs(diff) < 1e-9:
            g["ats_winner"], g["ats_cover_margin"] = "push", 0.0
        else:
            g["ats_winner"] = "home" if diff > 0 else "away"
            g["ats_cover_margin"] = round(diff, 1)

    ou_value = g.get("ou_value")
    if ou_value in (None, 0):
        g["actual_ou_result"], g["ou_correct"] = None, None
    else:
        ou_value = float(ou_value)
        if total > ou_value:
            g["actual_ou_result"] = "OVER"
        elif total < ou_value:
            g["actual_ou_result"] = "UNDER"
        else:
            g["actual_ou_result"] = "PUSH"
        pick = g.get("ou_pick")
        if g["actual_ou_result"] == "PUSH" or pick not in ("OVER", "UNDER"):
            g["ou_correct"] = None
        else:
            g["ou_correct"] = pick == g["actual_ou_result"]
    return g


def apply_freeze(existing_games: dict | None, fresh_games: dict, now_iso: str | None = None) -> dict:
    """Carry frozen blocks from the previously exported file into the fresh export.

    ``existing_games`` is the ``games`` dict of the file currently on disk (or
    None), ``fresh_games`` the dict just produced by the runner. Returns
    ``fresh_games`` with ``pregame`` blocks attached and locked games overlaid.
    """
    now_iso = now_iso or datetime.now().isoformat(timespec="seconds")
    existing_games = existing_games or {}
    for key, g in fresh_games.items():
        prev = existing_games.get(key) or {}
        prev_block = prev.get("pregame")
        if prev_block and prev_block.get("locked"):
            g["pregame"] = copy.deepcopy(prev_block)
            overlay(g, g["pregame"])
            grade(g)
            g["pregame_missing"] = False
            continue
        if not game_started(g):
            revisions = (prev_block or {}).get("revisions", 0) + 1
            g["pregame"] = snapshot(g, now_iso, revisions)
            g["pregame_missing"] = False
            continue
        # The game has started (or finished) on this export.
        if prev_block:
            block = copy.deepcopy(prev_block)
            block["locked"] = True
            block["locked_at"] = now_iso
            g["pregame"] = block
            overlay(g, block)
            grade(g)
            g["pregame_missing"] = False
        else:
            g["pregame"] = None
            g["pregame_missing"] = True
    return fresh_games


def frozen_games(games: Iterable[tuple[str, dict]]) -> list[tuple[str, dict]]:
    """(key, game) pairs whose prediction is locked pre-game — the only ones a
    tracker may grade."""
    return [(k, g) for k, g in games if (g.get("pregame") or {}).get("locked")]
