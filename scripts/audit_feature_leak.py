"""Audit same-day stat leakage and its effect on the production ML/ATS models.

0. Dataset test: does any row of the production dataset carry a GP that already counts
   the game the row describes? Must be zero since Create_Games picks pre-game snapshots.
1. Inclusion test: for teams that played on day D, does the stats table named D already
   count that game (GP == games through D)? Backfilled tables do, because Get_Data fetches
   leaguedashteamstats with DateTo=D after D is over.
2. Re-scoring: rebuild the seasons twice with Create_Games — LEAKY (SNAPSHOT_MODE
   "same_day": the table named D, what dataset.sqlite used until 2026-10-01) and CLEAN
   (SNAPSHOT_MODE "pregame", the production default) — and score both with the production
   models, thresholds and away-quality filter on the identical set of games.

Nothing is written under Data/; rebuilt datasets go to --work-dir.

Usage:
    PYTHONPATH=. python3 scripts/audit_feature_leak.py [--seasons 2024-25 2025-26] [--work-dir DIR]
"""
import bisect
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


def season_odds(con, season_key):
    from src.Utils.OddsTables import load_season_odds
    return load_season_odds(con, season_key)


def _shift(date_str, days):
    return (dt.datetime.strptime(date_str, "%Y-%m-%d") + dt.timedelta(days=days)).strftime("%Y-%m-%d")


def dataset_test(seasons, cfg, dataset_db, table="dataset_2012-26"):
    """Count dataset team-rows whose GP already includes the row's own game.

    The games a team had completed before date D are derived from a LATER anchor: the
    first stats table named >= D+2 on whose date and eve the team was idle, minus the
    games it played from D up to that anchor. This is independent of how Create_Games
    counts (it anchors backwards), so the two cannot share a blind spot.
    """
    from src.Utils.Dictionaries import team_index_current

    print(f"== 0. Dataset test ({Path(dataset_db).name}): rows whose GP counts their own game ==")
    cup_finals = {"2023-12-09", "2024-12-17", "2025-12-16"}
    tc = sqlite3.connect(REPO / "Data" / "TeamData.sqlite")
    with sqlite3.connect(dataset_db) as con:
        rows = pd.read_sql_query(f'select Date, GP, "GP.1" from "{table}"', con)
        names = pd.read_sql_query(f'select Home, Away from "{table}_snapshots"', con)
    rows["Home"], rows["Away"] = names["Home"], names["Away"]
    total_leaks = 0
    with sqlite3.connect(ODDS_DB) as oc:
        for s in seasons:
            start = dt.datetime.strptime(cfg["create-games"][s]["start_date"], "%Y-%m-%d").strftime("%Y-%m-%d")
            po = cfg.get("get-playoffs", {}).get(s, {}).get("start_date", "9999-12-31")
            po = dt.datetime.strptime(po, "%Y-%m-%d").strftime("%Y-%m-%d")
            odds = season_odds(oc, s)
            odds = odds[odds.Date < po]
            scheduled, played = {}, {}
            for d, h, a, pts in zip(odds.Date, odds.Home, odds.Away, odds.Points):
                for t in (h, a):
                    scheduled.setdefault(t, set()).add(d)
                    if pts > 0 and d not in cup_finals:
                        played.setdefault(t, []).append(d)
            for v in played.values():
                v.sort()
            tables = sorted(r[0] for r in tc.execute(
                "select name from sqlite_master where type='table' and name glob '????-??-??' "
                "and name >= ? and name < ?", (start, _shift(po, 3))))
            gp_cache = {}

            def gp_of(tbl, team):
                if tbl not in gp_cache:
                    gp_cache[tbl] = pd.read_sql_query(f'select GP from "{tbl}"', tc)["GP"].tolist()
                col = gp_cache[tbl]
                return col[team_index_current[team]] if len(col) == 30 and team in team_index_current else None

            x = rows[(rows.Date >= start) & (rows.Date < po)]
            n = leak = stale = unknown = 0
            for d, h, a, g, g1 in zip(x.Date, x.Home, x.Away, x.GP, x["GP.1"]):
                for t, row_gp in ((h, g), (a, g1)):
                    n += 1
                    before = None
                    i = bisect.bisect_left(tables, _shift(d, 2))
                    for anchor in tables[i:i + 30]:
                        if anchor in scheduled.get(t, ()) or _shift(anchor, -1) in scheduled.get(t, ()):
                            continue
                        shown = gp_of(anchor, t)
                        if shown is None:
                            continue
                        pl = played.get(t, [])
                        before = shown - (bisect.bisect_right(pl, _shift(anchor, -2)) - bisect.bisect_left(pl, d))
                        break
                    if before is None:
                        unknown += 1
                    elif row_gp > before:
                        leak += 1
                    elif row_gp < before:
                        stale += 1
            total_leaks += leak
            print(f"  {s}: team-rows {n:5d} | counts own game: {leak:4d} | older than pre-game: {stale:4d} "
                  f"| no later anchor: {unknown}")
    tc.close()
    return total_leaks


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
            o = season_odds(oc, s)[["Date", "Home", "Away"]]
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

    class _Toml:
        @staticmethod
        def load(_):
            c = dict(cfg)
            c["create-games"] = {k: v for k, v in cfg["create-games"].items() if k in seasons}
            return c

    cg.toml = _Toml
    cg.OUTPUT_DB_PATH = work_db
    for mode, snapshot_mode in (("leaky", "same_day"), ("clean", "pregame")):
        cg.SNAPSHOT_MODE = snapshot_mode
        cg.OUTPUT_TABLE = f"ds_{mode}"
        print(f"  building {mode} dataset ...", flush=True)
        cg.main()


def score(seasons, work_db):
    import xgboost as xgb
    import src.Utils.AdvancedFeatures as AF
    from src.Predict.XGBoost_Runner import _build_frame_ats, _load_calibrator, _select_model_path
    from src.Utils import SeasonStats as SS

    AF.reset_cache()
    ml_path, ats_path = _select_model_path("ML"), _select_model_path("ATS")
    print("  models:", ml_path.name, "|", ats_path.name)
    b_ml, b_ats = xgb.Booster(), xgb.Booster()
    b_ml.load_model(str(ml_path))
    b_ats.load_model(str(ats_path))
    cal_ml = _load_calibrator(ml_path)
    with sqlite3.connect(ODDS_DB) as oc:
        odds = pd.concat([
            season_odds(oc, s)[["Date", "Home", "Away", "ML_Home", "ML_Away", "Spread", "Win_Margin"]]
            for s in seasons])
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
        adv = AF.ats_model_columns(helper.columns)
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
    ap.add_argument("--dataset-only", action="store_true",
                    help="run only part 0 and exit non-zero if any row counts its own game")
    args = ap.parse_args()
    os.chdir(REPO)
    cfg = toml.load(REPO / "config.toml")
    bounds = season_bounds(cfg, args.seasons)
    leaks = dataset_test(args.seasons, cfg, REPO / "Data" / "dataset.sqlite")
    if args.dataset_only:
        sys.exit(1 if leaks else 0)
    inclusion_test(args.seasons, bounds)
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    work_db = Path(args.work_dir) / "leak_audit.sqlite"
    if not args.skip_build:
        rebuild(args.seasons, bounds, cfg, work_db)
    report(score(args.seasons, work_db), bounds)


if __name__ == "__main__":
    main()
