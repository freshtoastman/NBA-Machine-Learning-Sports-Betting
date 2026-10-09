"""Print + write the frozen pre-game record (web/data/frozen_record.json).

Usage:
    PYTHONPATH=. python3 scripts/frozen_record.py [--since 2026-10-02] [--no-write]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.Utils import Policy  # noqa: E402
from src.Utils.FrozenRecord import collect, fmt, write_frozen_record  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "web" / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None, help="first game date to count (default: [ats-policy].effective_from)")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    since = args.since or str(Policy.ats_policy()["effective_from"])[:10]
    if args.no_write:
        out = collect(DATA_DIR, since=since)
    else:
        out = write_frozen_record(DATA_DIR, since=since)
    print(f"frozen games {out['frozen_games']} | graded {out['graded_games']} | pending (pre-game) {out['pending_games']}"
          f" | no snapshot {len(out['games_without_pregame_snapshot'])} | legacy without block {out['legacy_graded_games_without_block']}")
    print("ML winner        :", fmt(out["ml"]))
    print("Over/Under       :", fmt(out["ou"]))
    print("ATS model        :", fmt(out["ats_model"]))
    print("ATS shadow value :", fmt(out["ats_shadow_value"]), "| re-open met:", out["reopen"]["ats"]["met"])
    r = out["ml_shadow_value_roi"]
    print(f"ML shadow diamond: n={r['n']} ROI {r['roi_pct']}% CI {r['ci95']} | re-open met: {out['reopen']['ml_value']['met']}")
    r = out["ml_shadow_golden_roi"]
    print(f"ML shadow golden : n={r['n']} ROI {r['roi_pct']}% CI {r['ci95']}")
    print("Playoff GOLD+SILVER headline:", fmt(out["playoff_best_gold_silver"]))


if __name__ == "__main__":
    main()
