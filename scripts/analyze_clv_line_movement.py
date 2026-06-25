#!/usr/bin/env python3
"""
CLV / line-movement vs ATS-cover analysis.

Hypothesis (sharp-money / "steam"): when the spread moves toward a team between
the opening and closing number, that reflects informed money and the team should
cover more often than chance. We test whether the *magnitude* of the move matters.

Data source: web/data/2026-*.json daily files already carry per-game
`spread_first` / `spread_last` / `spread_move` (home spread; negative = home
favored) plus the graded `ats_winner`. No DB join required.

Output: web/data/clv_analysis.json  (consumed by the off-season "data lab" view)

Re-run next season once the sample grows — the >=2.0 bucket is currently tiny and
must stay observe-only until n is large enough to size on.
"""
import json, glob, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DATA = os.path.join(ROOT, "web", "data")


def load_games():
    games = []
    for f in sorted(glob.glob(os.path.join(WEB_DATA, "2026-*.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not isinstance(d, dict) or "games" not in d:
            continue
        for _, g in d["games"].items():
            if not isinstance(g, dict):
                continue
            if not g.get("is_historical"):
                continue
            if g.get("ats_winner") in (None, "push"):
                continue
            move = g.get("spread_move")
            if move is None:
                continue
            games.append({
                "date": d.get("date"),
                "home": g.get("home_team"),
                "away": g.get("away_team"),
                "move": move,
                "spread_first": g.get("spread_first"),
                "spread_last": g.get("spread_last"),
                "ats": g.get("ats_winner"),
                "cover_margin": g.get("ats_cover_margin"),
                "is_playoff": bool(g.get("is_playoff")),
            })
    return games


def bucket(games, lo, hi=None):
    """Games whose |move| is in [lo, hi)."""
    out = []
    for g in games:
        m = abs(g["move"])
        if m >= lo and (hi is None or m < hi):
            out.append(g)
    return out


def clv_hitrate(rows):
    """How often the side the line moved TOWARD ended up covering."""
    hit = tot = 0
    for g in rows:
        toward = "home" if g["move"] < 0 else "away"
        tot += 1
        if g["ats"] == toward:
            hit += 1
    pct = (hit / tot * 100) if tot else 0.0
    return {"n": tot, "hit": hit, "pct": round(pct, 1)}


def main():
    games = load_games()
    nomove = bucket(games, 0.0, 0.5)
    home_cover_nomove = sum(1 for g in nomove if g["ats"] == "home")

    result = {
        "generated_from": "web/data/2026-*.json (graded games w/ spread_move)",
        "total_graded": len(games),
        "note": "Late-2025-26-season + playoff sample only; line-movement tracking "
                "began ~April. Large-move buckets are tiny — OBSERVE-ONLY until "
                "cross-season sample grows. Small moves are noise.",
        "buckets": [],
        "baseline_no_move": {
            "n": len(nomove),
            "home_cover": home_cover_nomove,
            "home_cover_pct": round(home_cover_nomove / len(nomove) * 100, 1) if nomove else 0.0,
        },
    }
    for lo in (0.5, 1.0, 1.5, 2.0):
        rows = bucket(games, lo)
        r = clv_hitrate(rows)
        r["threshold"] = lo
        result["buckets"].append(r)

    # headline: the cleanest actionable read
    big = clv_hitrate(bucket(games, 2.0))
    result["headline"] = {
        "claim": "Large spread moves (>=2.0 pts) followed the cover direction "
                 f"{big['pct']}% ({big['hit']}/{big['n']}); small moves were noise.",
        "status": "observe-only",
        "min_n_to_size": 30,
    }

    out = os.path.join(WEB_DATA, "clv_analysis.json")
    json.dump(result, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"wrote {out}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
