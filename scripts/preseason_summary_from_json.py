#!/usr/bin/env python3
"""Recover the `--summary` input of build_preseason_preview.py from its own output.

The team-summary JSON (last-season records / ATS / O-U / NBA.com advanced ratings)
was produced once in September 2026 and lived only in a session scratchpad, so a
later run that just wants to fold in new injury/roster notes had no `--summary`
file to pass. Every field build() reads is written verbatim into
web/data/preseason_2026-27.json, so invert that mapping:

  python scripts/preseason_summary_from_json.py web/data/preseason_2026-27.json summary.json
  python scripts/build_preseason_preview.py --summary summary.json --curated scripts/preseason_curated_2026-27.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_preseason_preview import PLAYOFF_ZH  # noqa: E402

ZH_TO_EN = {v: k for k, v in PLAYOFF_ZH.items()}


def summary_from_output(data: dict) -> dict:
    teams = {}
    for team, t in data["teams"].items():
        ls = t["last_season"]
        teams[team] = {
            "regular_season": {
                "record": ls["record"],
                "home_record": ls["home_record"],
                "away_record": ls["away_record"],
                "point_diff_per_game": ls["point_diff"],
                "ats": {"overall": ls["ats"], "home": ls["ats_home"], "away": ls["ats_away"]},
                "over_under": {"record": ls["ou"], "over_pct": ls["over_pct"]},
            },
            "advanced_regular_season_nba_com": {
                "NET_RATING": ls["net_rating"], "OFF_RATING": ls["off_rating"],
                "DEF_RATING": ls["def_rating"], "NET_RATING_RANK": ls["net_rank"],
            },
            "playoff_round_reached": ZH_TO_EN.get(ls["playoff_result"], ls["playoff_result"]),
            "playoffs": ({"record": ls["playoff_record"], "ats": {"overall": ls["playoff_ats"]}}
                         if ls.get("playoff_record") else None),
        }
    L = data["league_last_season"]
    league = {"regular_season": {
        "home_cover_rate_excl_push": L["home_cover_pct"],
        "favorite_cover_rate_excl_push": L["favorite_cover_pct"],
        "home_win_pct": L["home_win_pct"],
        "over_rate_excl_push": L["over_pct"],
        "avg_home_spread": L["avg_home_spread"],
        "games": L["games"],
    }}
    return {"teams": teams, "league": league}


def main():
    src, dst = sys.argv[1], sys.argv[2]
    data = json.loads(Path(src).read_text(encoding="utf-8"))
    Path(dst).write_text(json.dumps(summary_from_output(data), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(data['teams'])} teams -> {dst}")


if __name__ == "__main__":
    main()
