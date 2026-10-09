"""Grade the frozen pre-game predictions in web/data/YYYY-MM-DD.json.

Only games whose ``pregame`` block is locked (src/Utils/PregameFreeze.py) are
counted: that block is the last prediction shown before tip-off, and the game's
result fields were graded against the frozen spread / total / moneyline. This
is the record the re-open bars in config.toml ([ats-policy] / [ml-value-policy])
are measured against, so the numbers here are the only "live record" the
frontend may quote.

    python3 scripts/frozen_record.py            # prints + writes web/data/frozen_record.json
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path

from src.Utils import Policy


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * (c - h), 1), round(100 * (c + h), 1)


def rec(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"wins": k, "losses": n - k, "n": n,
            "hit_rate": round(100 * k / n, 1) if n else None, "ci95": [lo, hi]}


def american_profit(odds, won: bool) -> float | None:
    """Flat 1-unit stake profit at American odds; None if the odds are missing."""
    if odds in (None, 0):
        return None
    try:
        odds = float(odds)
    except (TypeError, ValueError):
        return None
    if not won:
        return -1.0
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def roi(returns: list[float]) -> dict:
    n = len(returns)
    if n == 0:
        return {"n": 0, "roi_pct": None, "ci95": [None, None], "wins": 0, "losses": 0}
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n) if n > 1 else None
    lo = round(100 * (mean - 1.96 * se), 1) if se is not None else None
    hi = round(100 * (mean + 1.96 * se), 1) if se is not None else None
    return {"n": n, "roi_pct": round(100 * mean, 1), "ci95": [lo, hi],
            "wins": sum(1 for r in returns if r > 0), "losses": sum(1 for r in returns if r < 0)}


def season_of(iso: str) -> str:
    d = date.fromisoformat(iso[:10])
    y = d.year if d.month >= 10 else d.year - 1
    return f"{y}-{(y + 1) % 100:02d}"


def _is_locked(g: dict) -> bool:
    return bool((g.get("pregame") or {}).get("locked"))


def _graded(g: dict) -> bool:
    return g.get("actual_winner") in ("home", "away")


class _Bucket:
    def __init__(self):
        self.frozen = 0
        self.graded = 0
        self.ml = [0, 0]
        self.ou = [0, 0]
        self.ats = [0, 0]
        self.ats_shadow = [0, 0]
        self.ats_shadow_rows = []
        self.ml_shadow_returns = []
        self.ml_golden_returns = []
        self.playoff_best = [0, 0]
        self.dates = set()

    def add(self, iso: str, g: dict):
        self.frozen += 1
        self.dates.add(iso)
        if not _graded(g):
            return
        self.graded += 1
        aw = g["actual_winner"]
        if g.get("winner") in ("home", "away"):
            self.ml[0] += g["winner"] == aw
            self.ml[1] += 1
        if g.get("ou_pick") in ("OVER", "UNDER") and g.get("actual_ou_result") in ("OVER", "UNDER"):
            self.ou[0] += g["ou_pick"] == g["actual_ou_result"]
            self.ou[1] += 1
        atsw = g.get("ats_winner")
        pick = g.get("ats_model_pick")
        if pick in ("home", "away") and atsw in ("home", "away"):
            hit = pick == atsw
            self.ats[0] += hit
            self.ats[1] += 1
            if g.get("ats_shadow_value"):
                self.ats_shadow[0] += hit
                self.ats_shadow[1] += 1
                self.ats_shadow_rows.append({"date": iso, "game": f"{g.get('away_team')} @ {g.get('home_team')}",
                                             "pick": pick, "spread": g.get("spread"), "hit": hit})
            bs = g.get("playoff_ats_best_side")
            if g.get("is_playoff") and bs in ("home", "away") and g.get("playoff_ats_best_tier") in ("GOLD", "SILVER"):
                self.playoff_best[0] += bs == atsw
                self.playoff_best[1] += 1
        side = g.get("value_side")
        if side in ("home", "away"):
            odds = g.get("home_team_odds") if side == "home" else g.get("away_team_odds")
            profit = american_profit(odds, side == aw)
            if profit is not None:
                if g.get("ml_shadow_value"):
                    self.ml_shadow_returns.append(profit)
                if g.get("ml_shadow_golden"):
                    self.ml_golden_returns.append(profit)

    def summary(self) -> dict:
        return {
            "frozen_games": self.frozen,
            "graded_games": self.graded,
            "game_dates": len(self.dates),
            "ml": rec(*self.ml),
            "ou": rec(*self.ou),
            "ats_model": rec(*self.ats),
            "ats_shadow_value": rec(*self.ats_shadow),
            "ml_shadow_value_roi": roi(self.ml_shadow_returns),
            "ml_shadow_golden_roi": roi(self.ml_golden_returns),
            "playoff_best_gold_silver": rec(*self.playoff_best),
        }


def collect(data_dir: Path, since: str | None = None) -> dict:
    """Walk every daily file and grade the locked pre-game predictions."""
    buckets: dict[str, _Bucket] = defaultdict(_Bucket)
    overall = _Bucket()
    missing = []   # games that started before any pre-game export existed
    unfrozen = 0   # pre-game (not yet started) games
    legacy = 0     # finished games exported before the freeze existed (no block)
    for jf in sorted(data_dir.glob("????-??-??.json")):
        iso = jf.name[:10]
        if since and iso < since:
            continue
        try:
            games = json.loads(jf.read_text(encoding="utf-8")).get("games") or {}
        except Exception:
            continue
        for key, g in games.items():
            if not isinstance(g, dict):
                continue
            if _is_locked(g):
                buckets[season_of(iso)].add(iso, g)
                overall.add(iso, g)
            elif g.get("pregame_missing"):
                missing.append({"date": iso, "game": key})
            elif g.get("pregame"):
                unfrozen += 1
            elif _graded(g):
                legacy += 1
    ats_pol = Policy.ats_policy()
    ml_pol = Policy.ml_value_policy()
    ov = overall.summary()
    shadow = ov["ats_shadow_value"]
    ml_roi = ov["ml_shadow_value_roi"]
    out = {
        "generated_for": "src/Utils/FrozenRecord.py — graded from locked pregame blocks only",
        "since": since,
        "definition_zh": "每場只評分開賽前最後一次輸出並凍結的預測（pregame.locked），結果一律以凍結當下的盤口結算；"
                         "沒有賽前快照的比賽不計。",
        **ov,
        "pending_games": unfrozen,
        "games_without_pregame_snapshot": missing,
        "legacy_graded_games_without_block": legacy,
        "by_season": {s: b.summary() for s, b in sorted(buckets.items())},
        "reopen": {
            "ats": {
                "rule": "ats_shadow_value picks: n >= min_picks AND 95% CI lower bound > ci_lower_pct",
                "min_picks": ats_pol["reopen_min_frozen_picks"],
                "ci_lower_pct": ats_pol["reopen_ci_lower_pct"],
                "n": shadow["n"],
                "ci_lower": shadow["ci95"][0],
                "met": bool(shadow["n"] >= ats_pol["reopen_min_frozen_picks"]
                            and shadow["ci95"][0] is not None
                            and shadow["ci95"][0] > ats_pol["reopen_ci_lower_pct"]),
            },
            "ml_value": {
                "rule": "ml_shadow_value picks settled at the frozen moneyline: n >= min_picks AND ROI 95% CI lower bound > roi_ci_lower_pct",
                "min_picks": ml_pol["reopen_min_frozen_picks"],
                "roi_ci_lower_pct": ml_pol["reopen_roi_ci_lower_pct"],
                "n": ml_roi["n"],
                "roi_ci_lower": ml_roi["ci95"][0],
                "met": bool(ml_roi["n"] >= ml_pol["reopen_min_frozen_picks"]
                            and ml_roi["ci95"][0] is not None
                            and ml_roi["ci95"][0] > ml_pol["reopen_roi_ci_lower_pct"]),
            },
        },
        "ats_shadow_rows": overall.ats_shadow_rows[-50:],
    }
    return out


def write_frozen_record(data_dir: Path, since: str | None = None) -> dict:
    since = since or Policy.ats_policy()["effective_from"]
    out = collect(data_dir, since=str(since)[:10])
    from datetime import datetime
    out["generated_at"] = datetime.now().isoformat(timespec="seconds")
    (data_dir / "frozen_record.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def fmt(r: dict) -> str:
    if not r.get("n"):
        return "0-0"
    return f"{r['wins']}-{r['losses']} ({r['hit_rate']}%, 95% CI {r['ci95'][0]}-{r['ci95'][1]}%)"
