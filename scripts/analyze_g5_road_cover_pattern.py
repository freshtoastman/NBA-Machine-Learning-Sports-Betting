"""Resolve the Finals G5 signal conflict with conditional series-pattern data.

Question: in best-of-7 playoff series where the ROAD team covered most/all of
games 1-4, what does the G5 home team do ATS?  The unconditional "G5 home
covers 85% (CF+Finals)" GOLD signal assumes home-court reassertion — but if
road sides have covered 4-0, the market may be systematically mispricing the
home edge in this series.

Reconstructs series game-by-game from OddsData odds tables (same convention as
backtest_ats.py: Home_Cover_Margin = Win_Margin - Spread, positive => home
covered).
"""
import sqlite3
import pandas as pd

DB = "Data/OddsData.sqlite"

PLAYOFF_RANGES = {
    "2012-13": ("2013-04-20", "2013-06-20"),
    "2013-14": ("2014-04-19", "2014-06-15"),
    "2014-15": ("2015-04-18", "2015-06-16"),
    "2015-16": ("2016-04-16", "2016-06-19"),
    "2016-17": ("2017-04-15", "2017-06-12"),
    "2017-18": ("2018-04-14", "2018-06-08"),
    "2018-19": ("2019-04-13", "2019-06-13"),
    "2019-20": ("2020-08-15", "2020-10-11"),
    "2020-21": ("2021-05-18", "2021-07-20"),
    "2021-22": ("2022-04-11", "2022-06-16"),
    "2022-23": ("2023-04-11", "2023-06-12"),
    "2023-24": ("2024-04-16", "2024-06-17"),
    "2024-25": ("2025-04-15", "2025-06-22"),
    "2025-26": ("2026-04-14", "2026-06-22"),
}


def _table_exists(con, name):
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _freshest_odds_table(con, season_key):
    for t in (f"odds_{season_key}_new", f"odds_{season_key}",
              f"{season_key}_new", season_key):
        if _table_exists(con, t):
            return t
    return None


def load_playoff_series():
    con = sqlite3.connect(DB)
    all_series = []
    for season, (start, end) in PLAYOFF_RANGES.items():
        tbl = _freshest_odds_table(con, season)
        if tbl is None:
            continue
        df = pd.read_sql(
            f'SELECT Date, Home, Away, Spread, Win_Margin, Points FROM "{tbl}"',
            con,
        )
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        for col in ("Spread", "Win_Margin"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Date", "Spread", "Win_Margin"])
        po = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
        po["pair"] = po.apply(
            lambda r: tuple(sorted([r["Home"], r["Away"]])), axis=1
        )
        for pair, grp in po.groupby("pair"):
            grp = grp.sort_values("Date").reset_index(drop=True)
            if len(grp) < 4:  # play-in or data gaps; need a real series
                continue
            games = []
            for i, r in grp.iterrows():
                margin = r["Win_Margin"] - r["Spread"]
                games.append({
                    "season": season,
                    "game_num": i + 1,
                    "home": r["Home"],
                    "away": r["Away"],
                    "spread": r["Spread"],
                    "win_margin": r["Win_Margin"],
                    "home_covered": margin > 0,
                    "away_covered": margin < 0,
                    "push": margin == 0,
                    "home_won": r["Win_Margin"] > 0,
                })
            all_series.append({"season": season, "pair": pair, "games": games})
    con.close()
    return all_series


def main():
    series_list = load_playoff_series()
    print(f"Reconstructed {len(series_list)} playoff series "
          f"({sum(len(s['games']) for s in series_list)} games)\n")

    buckets = {
        "road 4-0 ATS in G1-4": [],
        "road 3+ of 4 ATS in G1-4": [],
        "road <=2 of 4 ATS in G1-4": [],
    }
    # Secondary cut: G5 where the AWAY team leads 3-1 (road closeout chance)
    away_lead_31 = []

    for s in series_list:
        games = {g["game_num"]: g for g in s["games"]}
        if 5 not in games:
            continue
        first4 = [games[i] for i in (1, 2, 3, 4) if i in games]
        if len(first4) < 4:
            continue
        road_covers = sum(1 for g in first4 if g["away_covered"])
        g5 = games[5]
        if g5["push"]:
            continue
        rec = {
            "season": s["season"],
            "pair": s["pair"],
            "g5_home": g5["home"],
            "g5_spread": g5["spread"],
            "g5_home_covered": g5["home_covered"],
            "g5_home_won": g5["home_won"],
            "road_covers_g1_4": road_covers,
        }
        if road_covers == 4:
            buckets["road 4-0 ATS in G1-4"].append(rec)
        if road_covers >= 3:
            buckets["road 3+ of 4 ATS in G1-4"].append(rec)
        else:
            buckets["road <=2 of 4 ATS in G1-4"].append(rec)

        # away leads 3-1 entering G5?
        home_wins = sum(1 for g in first4 if g["home_won"]
                        and g["home"] == g5["home"]) + \
                    sum(1 for g in first4 if not g["home_won"]
                        and g["away"] == g5["home"])
        if home_wins == 1:  # G5 home team has 1 win => trails 1-3
            away_lead_31.append(rec)

    for label, recs in buckets.items():
        n = len(recs)
        if not n:
            print(f"{label}: n=0")
            continue
        cov = sum(1 for r in recs if r["g5_home_covered"])
        won = sum(1 for r in recs if r["g5_home_won"])
        print(f"{label}: n={n}  G5 home ATS {cov}/{n} = {cov/n:.1%}  "
              f"(SU {won}/{n} = {won/n:.1%})")

    n = len(away_lead_31)
    if n:
        cov = sum(1 for r in away_lead_31 if r["g5_home_covered"])
        won = sum(1 for r in away_lead_31 if r["g5_home_won"])
        print(f"\nG5 home trails 1-3 (facing elimination at home): "
              f"n={n}  home ATS {cov}/{n} = {cov/n:.1%}  SU {won}/{n} = {won/n:.1%}")

    # Joint condition matching the current Finals exactly:
    joint = [r for r in away_lead_31 if r["road_covers_g1_4"] >= 3]
    if joint:
        cov = sum(1 for r in joint if r["g5_home_covered"])
        print(f"\nJOINT (home trails 1-3 AND road covered 3+ of G1-4): "
              f"n={len(joint)}  home ATS {cov}/{len(joint)} = {cov/len(joint):.1%}")
        for r in joint:
            print(f"  {r['season']} {r['pair'][0]} vs {r['pair'][1]} — "
                  f"G5 home {r['g5_home']} spread {r['g5_spread']:+.1f} "
                  f"covered={r['g5_home_covered']}")

    print("\nDetail: road 4-0 ATS series")
    for r in buckets["road 4-0 ATS in G1-4"]:
        print(f"  {r['season']} {r['pair'][0]} vs {r['pair'][1]} — "
              f"G5 home {r['g5_home']} spread {r['g5_spread']:+.1f} "
              f"covered={r['g5_home_covered']} won={r['g5_home_won']}")


if __name__ == "__main__":
    main()
