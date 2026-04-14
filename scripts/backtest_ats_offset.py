"""ATS "buy point" offset cover-rate table.

For every closed game in OddsData.sqlite (across multiple seasons with
consistent spread conventions), compute home_cover_margin = Win_Margin -
Spread. Then report the probability that each side covers if the bettor
gets N extra points above the closing line.

This table is the source of truth for the in-app "買點計算器" widget in
web/templates/index.html — if you re-run and the numbers shift, update
the `BUY_POINT_TABLE` constant in that template.

Usage:
    PYTHONPATH=. python scripts/backtest_ats_offset.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ODDS_DB = PROJECT_ROOT / "Data" / "OddsData.sqlite"

# Seasons with the same Spread sign convention
# (Spread > 0 ⇒ home favored, away gets +|Spread|).
# Older `_new` tables (pre-2022) use the opposite convention and are excluded
# so the baseline home_cover rate lands near 50% as a sanity check.
SEASONS = ["odds_2022-23_new", "odds_2023-24_new", "odds_2025-26"]


def load_games() -> pd.DataFrame:
    with sqlite3.connect(ODDS_DB) as con:
        frames = []
        for t in SEASONS:
            try:
                d = pd.read_sql_query(f'SELECT Spread, Win_Margin FROM "{t}"', con)
            except Exception:
                print(f"  skip {t} (not present)")
                continue
            d["Spread"] = pd.to_numeric(d["Spread"], errors="coerce")
            d["Win_Margin"] = pd.to_numeric(d["Win_Margin"], errors="coerce")
            d = d.dropna()
            d = d[d["Spread"] != 0]
            print(f"  {t}: {len(d)} rows")
            frames.append(d)
    return pd.concat(frames, ignore_index=True)


def main():
    df = load_games()
    df["hcm"] = df["Win_Margin"] - df["Spread"]
    n = len(df)
    hc = (df["hcm"] > 0).mean()
    ac = (df["hcm"] < 0).mean()
    pu = (df["hcm"] == 0).mean()
    print(f"\nTotal: {n} games")
    print(f"Baseline: home cover {hc*100:.2f}%  away cover {ac*100:.2f}%  push {pu*100:.2f}%")
    if abs(hc - 0.5) > 0.05:
        print(f"  ⚠ unusual baseline skew — check spread sign convention")
    print()
    print(f"{'offset':<10}{'best cov%':<12}{'edge@-110':<14}{'edge@-105':<14}")
    print("-" * 50)
    rows = []
    for off in np.arange(0.0, 6.1, 0.5):
        p_fav = float((df["hcm"] > -off).mean())
        p_dog = float((df["hcm"] < off).mean())
        best = max(p_fav, p_dog)
        e110 = (best - 0.5238) * 100
        e105 = (best - 0.5122) * 100
        rows.append((off, best, e110, e105))
        print(f"+{off:<8.1f}{best*100:<12.2f}{e110:<+14.2f}{e105:<+14.2f}")

    print("\nJS lookup table snippet (paste into index.html):")
    print("    const BUY_POINT_TABLE = [")
    for off, cov, *_ in rows:
        print(f"        [{off:.1f}, {cov:.3f}],")
    print("    ];")


if __name__ == "__main__":
    main()
