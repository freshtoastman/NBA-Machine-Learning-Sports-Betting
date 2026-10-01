"""Playoff home-cover rates by round and game number, with spreads as stored vs signed.

Checks the round/game base rates that PlayoffATSStrategy's GOLD/SILVER signals quote
(R1 G6 away 80.4%, R2 G2 home 87.5%, R2 G5 home 83.7%, CF G1 home 83.3%, ...). Series
game numbers are the pre-game state from the dataset; rounds come from series_state_<season>.

Usage:
    PYTHONPATH=. python3 scripts/audit_playoff_base_rates.py
"""
import math
import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import src.Utils.OddsTables as OT  # noqa: E402



def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


with sqlite3.connect("Data/dataset.sqlite") as con:
    ds = pd.read_sql_query('select Date, TEAM_NAME Home, "TEAM_NAME.1" Away, series_game_num g, series_lead_for_home lead, is_elimination_game elim '
                           'from "dataset_2012-26" where is_playoff = 1 and Score > 0', con)
ds["Date"] = ds["Date"].astype(str).str[:10]
ds["season"] = ds.Date.apply(lambda d: f"{int(d[:4])-1}-{d[2:4]}")
oc = sqlite3.connect("Data/OddsData.sqlite")
rounds = []
for s in sorted(ds.season.unique()):
    try:
        r = pd.read_sql_query(f'select round_num, high_seed, low_seed from "series_state_{s}"', oc)
    except Exception:
        continue
    for _, x in r.iterrows():
        rounds.append({"season": s, "pair": frozenset((x.high_seed, x.low_seed)), "round": int(x.round_num), "high": x.high_seed})
rounds = pd.DataFrame(rounds)
ds["pair"] = [frozenset(p) for p in zip(ds.Home, ds.Away)]
ds = ds.merge(rounds, on=["season", "pair"], how="left")

seasons = [f"{y}-{(y+1)%100:02d}" for y in range(2012, 2026)]
signed = OT.load_seasons_odds(oc, seasons)[["Date", "Home", "Away", "Spread", "Win_Margin"]]
orig = OT._sign_spreads
OT._sign_spreads = lambda df: df
unsigned = OT.load_seasons_odds(oc, seasons)[["Date", "Home", "Away", "Spread"]].rename(columns={"Spread": "Spread_raw"})
OT._sign_spreads = orig
m = ds.merge(signed, on=["Date", "Home", "Away"]).merge(unsigned, on=["Date", "Home", "Away"])
m = m[m.Spread.notna()]
print("playoff games with spread:", len(m), "| with round:", int(m["round"].notna().sum()))

def rate(x, col):
    v = x[x.Win_Margin != x[col]]
    k = int((v.Win_Margin - v[col] > 0).sum())
    return k, len(v)

def show(label, x, side="home"):
    out = []
    for col in ("Spread_raw", "Spread"):
        k, n = rate(x, col)
        if side == "away": k = n - k
        lo, hi = wilson(k, n)
        out.append(f"{k:3d}/{n:<3d} {100*k/n if n else float('nan'):5.1f}% [{lo:.0f}-{hi:.0f}]")
    print(f"  {label:<34} as stored {out[0]} | signed {out[1]}")

print("ALL playoff games"); show("home covers", m)
print("By round and game number (side with the claimed edge):")
show("R1 G6 away covers", m[(m["round"] == 1) & (m.g == 6)], "away")
show("R2 G1 home covers", m[(m["round"] == 2) & (m.g == 1)])
show("R2 G2 home covers", m[(m["round"] == 2) & (m.g == 2)])
show("R2 G5 home covers", m[(m["round"] == 2) & (m.g == 5)])
show("Finals away covers", m[m["round"] == 4], "away")
show("Elimination games, home covers", m[m.elim == 1])
old = m[m.Date < "2022-10-01"]; new = m[m.Date >= "2022-10-01"]
print("Home covers, all playoff games by era:")
show("2013-2022 (legacy tables)", old); show("2023-2026 (signed at source)", new)
print("Every (round, game) cell, home cover, signed:")
for r in (1, 2, 3, 4):
    for g in range(1, 8):
        x = m[(m["round"] == r) & (m.g == g)]
        if len(x) >= 8:
            show(f"R{r} G{g}", x)

# The quoted rates ("home is the favourite in all 48 R2 G2 games") were measured on the
# unsigned spreads; straight-up home wins show where numbers of that size come from.
print("Straight-up home wins in the cells quoted as GOLD cover rates (seasons through 2024-25):")
upto = m[m.Date < "2025-10-01"]
for label, r, g in (("R2 G2", 2, 2), ("R2 G5", 2, 5), ("CF G1", 3, 1), ("CF G2", 3, 2), ("CF G5", 3, 5)):
    x = upto[(upto["round"] == r) & (upto.g == g)]
    k, n = int((x.Win_Margin > 0).sum()), len(x)
    ck, cn = rate(x, "Spread")
    print(f"  {label}: home wins {k}/{n} {100 * k / n:5.1f}% | home covers (signed) {ck}/{cn} {100 * ck / cn:5.1f}%")
