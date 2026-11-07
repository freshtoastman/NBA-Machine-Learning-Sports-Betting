"""Flag a spread that moved a lot between the opener and the current line.

Spreads are stored home-positive (``+7.5`` = home favoured by 7.5), so a
positive move means money came in on the home side. The models only see team
statistics; a late scratch moves the line but not the model's probability
(2026-11-05 MIA @ GSW: 7.5 -> 3.5 while the card still said GSW 83.6%). The
alert tells the reader the number on the card predates whatever the market
just learned. It is not a pick: through 2026-11-06 following the late side of
such moves covered 15 of 37 times.
"""
from __future__ import annotations

ALERT_POINTS = 1.5   # same bar as the spread-movement log in docs/preseason_review


def _book(source) -> str | None:
    """'sbr_fanduel_open' -> 'sbr_fanduel' (the opener tag is not a different book)."""
    if not source:
        return None
    return source[:-5] if source.endswith("_open") else source


def _same_book_points(history, last, last_source) -> float | None:
    """How far the current book's own line moved: its earliest row -> ``last``.

    None when the history has no row from that book (or no spread on it).
    """
    book = _book(last_source)
    if not book:
        return None
    for row in history:
        if _book(row.get("source")) != book or row.get("spread") is None:
            continue
        try:
            return round(abs(last - float(row["spread"])), 1)
        except (TypeError, ValueError):
            return None
    return None


def line_move_alert(first, last, model_pick=None, first_source=None, last_source=None,
                    history=None) -> dict | None:
    """Return the alert payload for a game card, or None when the move is small.

    ``model_pick`` is the model's moneyline side ('home' / 'away'); when given,
    ``against_model`` says whether the market moved away from it.
    ``mixed_source`` is set when the opener and the current line come from
    different books: part of the gap may be the books disagreeing rather than
    the market moving (2026-10-25 TOR @ MIN: Fanatics opener TOR -5, FanDuel
    MIN -1.5 — a 6.5-point "flip" that never happened). When it is set and
    ``history`` (the odds_history rows, oldest first) is given, the payload also
    carries ``same_source_points``: the move within the current book's own rows,
    so a real move behind a fallback opener still reads as one (2026-11-06
    TOR @ MIL: DraftKings opener -7.5, FanDuel -7.5 -> -4.5 = 3.0 of its own).
    """
    if first is None or last is None:
        return None
    try:
        first, last = float(first), float(last)
    except (TypeError, ValueError):
        return None
    move = last - first
    flip = first * last < 0
    if abs(move) < ALERT_POINTS and not flip:
        return None
    late_side = "home" if move > 0 else "away"
    alert = {
        "points": round(abs(move), 1),
        "flip": flip,
        "late_side": late_side,
        "against_model": (late_side != model_pick) if model_pick in ("home", "away") else None,
        "mixed_source": bool(_book(first_source) and _book(last_source)
                             and _book(first_source) != _book(last_source)),
    }
    if alert["mixed_source"] and history:
        same = _same_book_points(history, last, last_source)
        if same is not None:
            alert["same_source_points"] = same
    return alert
