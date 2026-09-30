"""Audit the TRUE live (pre-game) record from git history of web/data/YYYY-MM-DD.json.

Daily export files get rewritten after games finish (re-exports, score patches, signal
pruning), so the current files and the tracker do not show what a visitor actually saw
before tip-off. This walks every committed version of every daily file and, per game,
keeps the LATEST version in which the game had no final score yet — the last pre-game
state — then grades those picks against the final result.

Usage:
    python3 scripts/audit_live_record.py [--since 2026-04-01] [--json out.json]
"""
import argparse
import collections
import json
import math
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout


def has_result(g):
    return g.get("home_score") not in (None, 0) and g.get("actual_winner") is not None


def wilson(k, n, z=1.96):
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * (c - h), 1), round(100 * (c + h), 1)


def rec(k, n):
    lo, hi = wilson(k, n)
    return {"wins": k, "losses": n - k, "n": n,
            "hit_rate": round(100 * k / n, 1) if n else None, "ci95": [lo, hi]}


def fmt(r):
    if not r["n"]:
        return "0-0"
    return f"{r['wins']}-{r['losses']} ({r['hit_rate']}%, 95% CI {r['ci95'][0]:.0f}-{r['ci95'][1]:.0f}%)"


def collect(since):
    paths = sorted({l for l in git("log", "--all", "--name-only", "--format=", "--", "web/data/").split()
                    if l.startswith("web/data/20") and l.endswith(".json") and l.split("/")[-1][:10] >= since})
    rows = []
    for path in paths:
        commits = [l.split() for l in git("log", "--format=%H %cI", "--", path).splitlines()][::-1]
        versions = []
        for h, ts in commits:
            txt = git("show", f"{h}:{path}")
            try:
                versions.append((h, ts, json.loads(txt)))
            except json.JSONDecodeError:
                continue
        if not versions:
            continue
        for key, fg in versions[-1][2].get("games", {}).items():
            pre = [(h, ts, v["games"][key]) for h, ts, v in versions
                   if key in v.get("games", {}) and not has_result(v["games"][key])]
            rows.append({"date": path.split("/")[-1][:10], "key": key, "final": fg,
                         "pre": pre[-1][2] if pre else None})
    return rows


def grade(rows):
    graded = [r for r in rows if r["pre"] is not None and r["final"].get("actual_winner")]
    out = {"games_in_files": len(rows), "graded_with_pregame_snapshot": len(graded),
           "dates_without_pregame_snapshot": sorted({r["date"] for r in rows if r["pre"] is None})}

    ml = [r for r in graded if r["pre"].get("winner") in ("home", "away")]
    out["ml"] = rec(sum(r["pre"]["winner"] == r["final"]["actual_winner"] for r in ml), len(ml))
    ou = [r for r in graded if r["pre"].get("ou_pick") in ("OVER", "UNDER")
          and r["final"].get("actual_ou_result") in ("OVER", "UNDER")]
    out["ou"] = rec(sum(r["pre"]["ou_pick"] == r["final"]["actual_ou_result"] for r in ou), len(ou))
    ats = [r for r in graded if r["pre"].get("ats_model_pick") in ("home", "away")
           and r["final"].get("ats_winner") in ("home", "away")]
    out["ats_raw"] = rec(sum(r["pre"]["ats_model_pick"] == r["final"]["ats_winner"] for r in ats), len(ats))
    val = [r for r in ats if r["pre"].get("ats_is_value")]
    out["ats_value"] = rec(sum(r["pre"]["ats_model_pick"] == r["final"]["ats_winner"] for r in val), len(val))
    out["ats_edge_changed_after_game"] = sum(
        r["pre"].get("ats_value_edge") != r["final"].get("ats_value_edge") for r in graded)

    tier = collections.defaultdict(lambda: [0, 0])
    best = collections.defaultdict(lambda: [0, 0])
    per_signal = collections.defaultdict(lambda: [0, 0])
    removed = []
    for r in graded:
        aw = r["final"].get("ats_winner")
        if aw not in ("home", "away"):
            continue
        final_names = {s.get("signal") for s in (r["final"].get("playoff_ats_picks") or [])}
        for s in r["pre"].get("playoff_ats_picks") or []:
            if s.get("side") not in ("home", "away"):
                continue
            hit = s["side"] == aw
            for bucket in (tier[s.get("tier")], per_signal[f"[{s.get('tier')}] {s.get('signal')}"]):
                bucket[0] += hit
                bucket[1] += 1
            if s.get("signal") not in final_names:
                removed.append({"date": r["date"], "game": r["key"], "tier": s.get("tier"),
                                "signal": s.get("signal"), "hit": hit})
        bs = r["pre"].get("playoff_ats_best_side")
        if bs in ("home", "away"):
            b = best[r["pre"].get("playoff_ats_best_tier")]
            b[0] += bs == aw
            b[1] += 1
    out["signals_by_tier"] = {t: rec(*v) for t, v in tier.items()}
    gs = [v for t, v in tier.items() if t in ("GOLD", "SILVER")]
    out["signals_gold_silver"] = rec(sum(v[0] for v in gs), sum(v[1] for v in gs))
    out["best_signal_by_tier"] = {t: rec(*v) for t, v in best.items()}
    gb = [v for t, v in best.items() if t in ("GOLD", "SILVER")]
    out["best_signal_gold_silver"] = rec(sum(v[0] for v in gb), sum(v[1] for v in gb))
    out["per_signal"] = {k: rec(*v) for k, v in sorted(per_signal.items(), key=lambda x: -x[1][1])}
    out["signals_removed_after_game"] = {
        "n": len(removed), "hits": sum(x["hit"] for x in removed),
        "misses": sum(not x["hit"] for x in removed), "items": removed}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2000-01-01")
    ap.add_argument("--json", help="write the summary here")
    args = ap.parse_args()
    s = grade(collect(args.since))
    print(f"games in files {s['games_in_files']} | graded with pre-game snapshot {s['graded_with_pregame_snapshot']}")
    print("no pre-game snapshot on:", ", ".join(s["dates_without_pregame_snapshot"]) or "-")
    print("ML winner       :", fmt(s["ml"]))
    print("Over/Under      :", fmt(s["ou"]))
    print("ATS model raw   :", fmt(s["ats_raw"]))
    print("ATS value picks :", fmt(s["ats_value"]))
    print("ATS edge changed after the game:", s["ats_edge_changed_after_game"], "games")
    for t, r in sorted(s["signals_by_tier"].items(), key=lambda x: str(x[0])):
        print(f"signals {t:<7}:", fmt(r))
    print("signals GOLD+SILVER (every row)     :", fmt(s["signals_gold_silver"]))
    print("headline signal GOLD+SILVER per game:", fmt(s["best_signal_gold_silver"]))
    rm = s["signals_removed_after_game"]
    print(f"signals shown pre-game but gone from final file: {rm['n']} (hits {rm['hits']}, misses {rm['misses']})")
    if args.json:
        Path(args.json).write_text(json.dumps(s, ensure_ascii=False, indent=1))
        print("wrote", args.json)


if __name__ == "__main__":
    main()
