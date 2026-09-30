"""Audit same-day stat leakage and its effect on the production ML/ATS models.

1. Inclusion test: for teams that played on day D, does the stats table named D already
   count that game (GP == games through D)? Backfilled tables do, because Get_Data fetches
   leaguedashteamstats with DateTo=D after D is over.
2. Re-scoring: rebuild the seasons twice with Create_Games — LEAKY (table for D, what
   dataset.sqlite uses) and CLEAN (latest table strictly before D, same season, i.e. what
   a live pre-game run can see) — and score both with the production models, thresholds
   and away-quality filter on the identical set of games.

Nothing is written under Data/; rebuilt datasets go to --work-dir.

Usage:
    PYTHONPATH=. python3 scripts/audit_feature_leak.py [--seasons 2024-25 2025-26] [--work-dir DIR]
"""
import argparse
import datetime as dt
import importlib.util
import math
import os
import random
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import toml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
ODDS_DB = REPO / "Data" / "OddsData.sqlite"
STEMS_NOT_IN_MODEL = {"game_num_season", "month_sin", "month_cos", "form_ats_pct_home_10",
                      "form_ats_pct_away_10", "form_pts_for_5", "form_pts_against_5", "form_pts_diff_10"}


def most_complete_odds_table(con, season_key):
    cands = [f"odds_{season_key}_new", f"odds_{season_key}", f"{season_key}_new", season_key]
    ex = [t for t in cands if con.execute(
        "select 1 from sqlite_master where type='table' and name=?", (t,)).fetchone()]
    if not ex:
        return None
    return max(ex, key=lambda t: con.execute(f'select count(*) from "{t}"').fetchone()[0])


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def fmt(hits, n):
    if n == 0:
        return "—"
    lo, hi = wilson(hits, n)
    return f"{hits}-{n - hits} {100 * hits / n:5.1f}% [95% CI {lo:.0f}-{hi:.0f}]"


def season_bounds(cfg, seasons):
    out = {}
    for s in seasons:
        y, m, d = map(int, cfg["create-games"][s]["start_date"].split("-"))
        out[s] = (f"{y:04d}-{m:02d}-{d:02d}", f"{y + 1:04d}-10-01")
    return out


def inclusion_test(seasons, bounds):
    print("== 1. Same-day inclusion test (teams that played on the table's date) ==")
    with sqlite3.connect(ODDS_DB) as oc:
        for s in seasons:
            o = pd.read_sql_query(f'select Date, Home, Away from "{most_complete_odds_table(oc, s)}"', oc)
            o["Date"] = o["Date"].astype(str).str[:10]
            lo, hi = bounds[s]
            o = o[(o.Date >= lo) & (o.Date < hi)].drop_duplicates()
            for db in ("TeamData.sqlite", "AdvancedTeamData.sqlite"):
                con = sqlite3.connect(REPO / "Data" / db)
                names = [r[0] for r in con.execute(
                    "select name from sqlite_master where type='table' and name >= ? and name < ? "
                    "and name glob '????-??-??'", (lo, hi))]
                if not names:
                    print(f"  {s} {db}: no tables")
                    continue
                random.seed(7)
                incl = excl = other = 0
                for d in random.sample(names, min(40, len(names))):
                    df = pd.read_sql_query(f'select TEAM_NAME, GP from "{d}"', con)
                    playing = set(o[o.Date == d].Home) | set(o[o.Date == d].Away)
                    for _, r in df.iterrows():
                        if r.TEAM_NAME not in playing:
                            continue
                        m = (o.Home == r.TEAM_NAME) | (o.Away == r.TEAM_NAME)
                        if r.GP == int((o.Date <= d)[m].sum()):
                            incl += 1
                        elif r.GP == int((o.Date < d)[m].sum()):
                            excl += 1
                        else:
                            other += 1
                print(f"  {s} {db:<24} includes same-day game: {incl:4d} | excludes: {excl:4d} | other: {other}")


def rebuild(seasons, bounds, cfg, work_db):
    spec = importlib.util.spec_from_file_location("cg", REPO / "src/Process-Data/Create_Games.py")
    cg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cg)
    leaky_team, leaky_adv = cg.fetch_team_table, cg.fetch_advanced_table

    def prior(con, date_str):
        start = next((lo for lo, hi in bounds.values() if lo <= date_str < hi), None)
        if start is None:
            return None
        row = con.execute("select name from sqlite_master where type='table' and name < ? and name >= ? "
                          "and name glob '????-??-??' order by name desc limit 1", (date_str, start)).fetchone()
        return row[0] if row else None

    def clean_team(con, date_str):
        t = prior(con, date_str)
        return pd.read_sql_query(f'select * from "{t}"', con) if t else None

    def clean_adv(con, date_str):
        t = prior(con, date_str) if con is not None else None
        return leaky_adv(con, t) if t else None

    class _Toml:
        @staticmethod
        def load(_):
            c = dict(cfg)
            c["create-games"] = {k: v for k, v in cfg["create-games"].items() if k in seasons}
            return c

    cg.toml = _Toml
    cg.select_odds_table = most_complete_odds_table
    cg.OUTPUT_DB_PATH = work_db
    for mode, team_fn, adv_fn in (("leaky", leaky_team, leaky_adv), ("clean", clean_team, clean_adv)):
        cg.fetch_team_table, cg.fetch_advanced_table = team_fn, adv_fn
        cg.OUTPUT_TABLE = f"ds_{mode}"
        print(f"  building {mode} dataset ...", flush=True)
        cg.main()


def score(seasons, work_db):
    import xgboost as xgb
    import src.Utils.AdvancedFeatures as AF
    from src.Predict.XGBoost_Runner import _build_frame_ats, _load_calibrator, _select_model_path
    from src.Utils import SeasonStats as SS

    AF._freshest_odds_table = most_complete_odds_table
    AF.reset_cache()
    ml_path, ats_path = _select_model_path("ML"), _select_model_path("ATS")
    print("  models:", ml_path.name, "|", ats_path.name)
    b_ml, b_ats = xgb.Booster(), xgb.Booster()
    b_ml.load_model(str(ml_path))
    b_ats.load_model(str(ats_path))
    cal_ml = _load_calibrator(ml_path)
    with sqlite3.connect(ODDS_DB) as oc:
        odds = pd.concat([pd.read_sql_query(
            f'select Date, Home, Away, ML_Home, ML_Away, Spread, Win_Margin from "{most_complete_odds_table(oc, s)}"', oc)
            for s in seasons])
    odds["Date"] = odds["Date"].astype(str).str[:10]
    odds = odds.drop_duplicates(["Date", "Home", "Away"])
    frames = {}
    for mode in ("leaky", "clean"):
        with sqlite3.connect(work_db) as con:
            df = pd.read_sql_query(f'select * from "ds_{mode}"', con)
        df = df[df["Score"].fillna(0) > 0].reset_index(drop=True)
        df["Date"] = df["Date"].astype(str).str[:10]
        df = df.merge(odds, how="left", left_on=["Date", "TEAM_NAME", "TEAM_NAME.1"],
                      right_on=["Date", "Home", "Away"])
        f = df.drop(columns=SS.DROP_FEATURE_COLS, errors="ignore").drop(
            columns=["Home", "Away", "ML_Home", "ML_Away", "Spread", "Win_Margin"], errors="ignore")
        f_ml = f.drop(columns=["OU"]) if "OU" in f.columns else f
        X_ml = f_ml.astype(float).to_numpy()[:, :int(b_ml.num_features())]
        p_ml = np.asarray(b_ml.predict(xgb.DMatrix(X_ml)))
        if cal_ml is not None:
            p_ml = np.asarray(cal_ml.predict_proba(p_ml))
        helper = AF.merge_into(df[["TEAM_NAME", "TEAM_NAME.1", "Date"]].copy(), "TEAM_NAME", "TEAM_NAME.1", "Date")
        adv = [c for c in helper.columns if c.startswith(("H_", "A_", "D_"))]
        adv = [c for c in adv if c[2:] not in STEMS_NOT_IN_MODEL] + [c for c in adv if c[2:] in STEMS_NOT_IN_MODEL]
        spreads = np.where(pd.isna(df["Spread"]), 0.0, df["Spread"].astype(float))
        Xa = _build_frame_ats(f_ml, spreads, advanced=helper[adv].fillna(0.0).reset_index(drop=True))
        Xa = Xa.astype(float).to_numpy()[:, :int(b_ats.num_features())]
        p_ats = np.asarray(b_ats.predict(xgb.DMatrix(Xa)))
        r = pd.DataFrame({"Date": df["Date"], "Home": df["TEAM_NAME"], "Away": df["TEAM_NAME.1"],
                          "Spread": df["Spread"], "Win_Margin": df["Win_Margin"],
                          "home_win": df["Home-Team-Win"].astype(int), "p_ml": p_ml[:, 1], "p_ats": p_ats[:, 1],
                          "net_h": df.get("ADV_NET_RATING"), "net_a": df.get("ADV_NET_RATING.1"),
                          "po": df.get("is_playoff")})
        r = r[r.Spread.notna() & r.Win_Margin.notna()]
        r = r[(r.Win_Margin - r.Spread) != 0].copy()
        r["home_cov"] = ((r.Win_Margin - r.Spread) > 0).astype(int)
        r["pick_home"] = (r.p_ats >= 0.5).astype(int)
        r["edge"] = (r.p_ats - 0.5).abs() * 100
        r["ats_hit"] = (r.pick_home == r.home_cov).astype(int)
        r["ml_hit"] = ((r.p_ml >= 0.5).astype(int) == r.home_win).astype(int)
        r["octfeb"] = r.Date.str[5:7].astype(int).isin([10, 11, 12, 1, 2])
        r["value"] = r.edge >= np.where(r.pick_home == 1, 8.0, 9.0)
        r["qfilt"] = ((r.pick_home == 0) & (r.net_h < -3) & (r.Spread.abs() <= 8)
                      & ((r.net_h - r.net_a) > -7)).fillna(False)
        frames[mode] = r
    key = ["Date", "Home", "Away"]
    common = frames["leaky"][key].merge(frames["clean"][key], on=key)
    return {m: f.merge(common, on=key) for m, f in frames.items()}


def report(frames, bounds):
    print("\n== 2. Same models / thresholds / games — LEAKY vs CLEAN features ==")
    for s, (lo, hi) in bounds.items():
        print(f"-- {s} --")
        for mode in ("leaky", "clean"):
            x = frames[mode][(frames[mode].Date >= lo) & (frames[mode].Date < hi)]
            of = x[x.octfeb]
            v = of[of.value]
            fl = v[~v.qfilt]
            print(f"  {mode.upper():<6} ML {fmt(int(x.ml_hit.sum()), len(x))} | ATS raw {fmt(int(x.ats_hit.sum()), len(x))}")
            print(f"         Oct-Feb value H8/A9 {fmt(int(v.ats_hit.sum()), len(v))} | +away filter {fmt(int(fl.ats_hit.sum()), len(fl))}")
            for side, name in ((1, "home"), (0, "away")):
                z = v[v.pick_home == side]
                print(f"           {name} value picks {fmt(int(z.ats_hit.sum()), len(z))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=["2024-25", "2025-26"])
    ap.add_argument("--work-dir", default=os.path.join(tempfile.gettempdir(), "nba_leak_audit"))
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()
    os.chdir(REPO)
    cfg = toml.load(REPO / "config.toml")
    bounds = season_bounds(cfg, args.seasons)
    inclusion_test(args.seasons, bounds)
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    work_db = Path(args.work_dir) / "leak_audit.sqlite"
    if not args.skip_build:
        rebuild(args.seasons, bounds, cfg, work_db)
    report(score(args.seasons, work_db), bounds)


if __name__ == "__main__":
    main()
