import argparse
import os
from datetime import datetime, timedelta

import pandas as pd
from colorama import Fore, Style

# Lazy tensorflow import — only needed for NN predictions, not XGBoost or export.
tf = None
def _get_tf():
    global tf
    if tf is None:
        import tensorflow as _tf
        tf = _tf
    return tf

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

from src.DataProviders.SbrOddsProvider import SbrOddsProvider
from src.Predict import NN_Runner, XGBoost_Runner
from src.Utils.Dictionaries import team_index_current
from src.Utils.tools import (
    create_todays_games_from_odds,
    current_nba_season,
    current_nba_season_start_year,
    get_json_data,
    to_data_frame,
    get_todays_games_json,
    create_todays_games,
)

_SEASON_START_YEAR = current_nba_season_start_year()
_SEASON = current_nba_season()

TODAYS_GAMES_URL = f"https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/{_SEASON_START_YEAR}/scores/00_todays_scores.json"
DATA_URL = f"https://stats.nba.com/stats/leaguedashteamstats?Conference=&DateFrom=&DateTo=&Division=&GameScope=&GameSegment=&Height=&ISTRound=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season={_SEASON}&SeasonSegment=&SeasonType=Regular%20Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision="
ADVANCED_DATA_URL = DATA_URL.replace("MeasureType=Base", "MeasureType=Advanced")
SCHEDULE_PATH = os.path.join(PROJECT_ROOT, "Data", f"nba-{_SEASON_START_YEAR}-UTC.csv")

# Advanced stat columns to merge (same as Create_Games).
_ADVANCED_COLS = [
    "TEAM_ID",
    "OFF_RATING", "DEF_RATING", "NET_RATING",
    "PACE", "TS_PCT", "EFG_PCT",
    "OREB_PCT", "DREB_PCT", "REB_PCT",
    "TM_TOV_PCT", "AST_PCT", "AST_TO", "AST_RATIO",
    "PIE",
]


def create_todays_games_data(games, df, odds, schedule_df, today, interactive=True, adv_df=None):
    """Build today's matchup features.

    When `odds` is provided, prefill OU + money lines from sportsbook data.
    When `odds` is None and `interactive=True` (CLI mode), prompt the user.
    When `odds` is None and `interactive=False` (Flask/web mode), fall back to
    None values so the model still produces predictions without stdin.
    """
    match_data = []
    todays_games_uo = []
    home_team_odds = []
    away_team_odds = []

    for game in games:
        home_team, away_team = game
        if home_team not in team_index_current or away_team not in team_index_current:
            continue
        if odds:
            game_key = f"{home_team}:{away_team}"
            game_odds = odds.get(game_key)
            if game_odds is None:
                todays_games_uo.append(None)
                home_team_odds.append(None)
                away_team_odds.append(None)
            else:
                todays_games_uo.append(game_odds.get('under_over_odds'))
                home_team_odds.append(game_odds.get(home_team, {}).get('money_line_odds'))
                away_team_odds.append(game_odds.get(away_team, {}).get('money_line_odds'))
        elif interactive:
            todays_games_uo.append(input(home_team + ' vs ' + away_team + ': '))
            home_team_odds.append(input(home_team + ' odds: '))
            away_team_odds.append(input(away_team + ' odds: '))
        else:
            # Non-interactive (Flask): no live odds, fall through with None.
            todays_games_uo.append(None)
            home_team_odds.append(None)
            away_team_odds.append(None)

        # calculate days rest for both teams
        home_games = schedule_df[
            (schedule_df['Home Team'] == home_team) | (schedule_df['Away Team'] == home_team)
        ]
        away_games = schedule_df[
            (schedule_df['Home Team'] == away_team) | (schedule_df['Away Team'] == away_team)
        ]
        previous_home_games = home_games.loc[
            home_games['Date'] <= today
        ].sort_values('Date', ascending=False).head(1)['Date']
        previous_away_games = away_games.loc[
            away_games['Date'] <= today
        ].sort_values('Date', ascending=False).head(1)['Date']
        if len(previous_home_games) > 0:
            last_home_date = previous_home_games.iloc[0]
            home_days_off = timedelta(days=1) + today - last_home_date
        else:
            home_days_off = timedelta(days=7)
        if len(previous_away_games) > 0:
            last_away_date = previous_away_games.iloc[0]
            away_days_off = timedelta(days=1) + today - last_away_date
        else:
            away_days_off = timedelta(days=7)
        home_team_series = df.iloc[team_index_current.get(home_team)]
        away_team_series = df.iloc[team_index_current.get(away_team)]
        stats = pd.concat([home_team_series, away_team_series])
        stats['Days-Rest-Home'] = home_days_off.days
        stats['Days-Rest-Away'] = away_days_off.days

        # Merge advanced efficiency stats if available.
        if adv_df is not None and len(adv_df) == 30:
            home_idx = team_index_current.get(home_team)
            away_idx = team_index_current.get(away_team)
            if home_idx is not None and away_idx is not None:
                adv_cols = [c for c in adv_df.columns if c != "TEAM_ID"]
                home_adv = adv_df.iloc[home_idx]
                away_adv = adv_df.iloc[away_idx]
                for col in adv_cols:
                    stats[f"ADV_{col}"] = home_adv[col]
                    stats[f"ADV_{col}.1"] = away_adv[col]

        match_data.append(stats)

    games_data_frame = pd.concat(match_data, ignore_index=True, axis=1)
    games_data_frame = games_data_frame.T

    frame_ml = games_data_frame.drop(columns=['TEAM_ID', 'TEAM_NAME'])
    data = frame_ml.values
    data = data.astype(float)

    return data, todays_games_uo, frame_ml, home_team_odds, away_team_odds


def _ats_value_threshold(game_date, is_away_pick: bool = False) -> float:
    """Return the minimum edge% to flag a bet as ATS value on a given date.

    OOS analysis with MCW=26 model (2024-25 + 2025-26, both seasons):
      Oct-Feb home picks:  ≥8%  → 94.1% (2024-25: 9/10, 2025-26: 7/7)
      Oct-Feb away picks:  ≥9%  → 82.1% (2024-25: 16/19, 2025-26: 7/9)
      Combined (H≥8%, A≥9% + quality filter): 86.7% (39/45 across both OOS)
      Per-season: 2024-25 86.2% (25/29), 2025-26 87.5% (14/16)
      March (late regular):  100% — 0/3 hit rate, negative EV
      April 1-13:            100% — garbage time / resting
      Playoffs (Apr 14+):    10%  — thin OOS sample, elevated bar
    """
    import datetime as _dt
    if isinstance(game_date, _dt.datetime):
        game_date = game_date.date()
    elif isinstance(game_date, str):
        game_date = _dt.date.fromisoformat(game_date)
    month = game_date.month
    day   = game_date.day
    # Late regular season (March and first ~2 weeks of April) has negative EV
    # due to resting/tanking — suppress by setting threshold to 100%.
    if month == 3:
        return 100.0  # Suppress — 0/3 hit rate in March 2025-26
    if month == 4 and day < 14:
        return 100.0  # Suppress — end-of-regular-season garbage time
    if month == 4:
        return 10.0   # Playoffs/play-in: slightly higher bar (thin OOS sample)
    # Oct, Nov, Dec, Jan, Feb — standard regular season.
    # Away picks use a 1pp higher threshold (9% vs 8%) since away picks at 8-9%
    # edge are only break-even. MCW=26 OOS: 86.7% combined (39/45, both seasons ≥86%).
    return 9.0 if is_away_pick else 8.0


def load_schedule():
    return pd.read_csv(SCHEDULE_PATH, parse_dates=['Date'], date_format='%d/%m/%Y %H:%M')


def resolve_games(odds, sportsbook):
    if odds:
        games = create_todays_games_from_odds(odds)
        if len(games) == 0:
            print("No games found.")
            return None, None
        game_key = f"{games[0][0]}:{games[0][1]}"
        if game_key not in odds:
            print(game_key)
            print(
                Fore.RED,
                "--------------Games list not up to date for todays games!!! Scraping disabled until list is updated.--------------",
            )
            print(Style.RESET_ALL)
            return games, None
        print(f"------------------{sportsbook} odds data------------------")
        for game_key in odds.keys():
            home_team, away_team = game_key.split(":")
            print(
                f"{away_team} ({odds[game_key][away_team]['money_line_odds']}) @ "
                f"{home_team} ({odds[game_key][home_team]['money_line_odds']})"
            )
        return games, odds

    games_json = get_todays_games_json(TODAYS_GAMES_URL)
    return create_todays_games(games_json), None


def run_models(data, normalized_data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args):
    if args.xgb:
        print("---------------XGBoost Model Predictions---------------")
        XGBoost_Runner.xgb_runner(
            data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc
        )
        print("-------------------------------------------------------")
    if args.nn:
        print("------------Neural Network Model Predictions-----------")
        NN_Runner.nn_runner(
            normalized_data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc
        )
        print("-------------------------------------------------------")


def _load_today_odds_fallback(games, today):
    """Read today's odds from local OddsData.sqlite, keyed by (home, away).

    Returns a dict {(home, away): {ML_Home, ML_Away, OU, Spread}} for any games
    whose row exists in the season table. Used when sbrscrape misses.
    """
    season_table = _season_table_for_date(today.date())
    out = {}
    try:
        with sqlite3.connect(_ODDS_DB) as con:
            rows = con.execute(
                f'SELECT Home, Away, ML_Home, ML_Away, OU, Spread FROM "{season_table}" '
                f'WHERE Date = ?',
                (today.date().isoformat(),),
            ).fetchall()
        for home, away, ml_h, ml_a, ou, spread in rows:
            out[(home, away)] = {
                "ML_Home": ml_h,
                "ML_Away": ml_a,
                "OU": ou,
                "Spread": spread,
            }
    except Exception:
        pass
    return out


def predict_today_xgb(sportsbook):
    """Programmatic XGBoost prediction for today's NBA games.

    Returns a list of structured prediction dicts (see XGBoost_Runner.xgb_predict).
    Used by Flask app and any other consumer that needs structured output.
    """
    odds = SbrOddsProvider(sportsbook=sportsbook).get_odds() if sportsbook else None
    games, odds = resolve_games(odds, sportsbook)
    if games is None:
        return []

    stats_json = get_json_data(DATA_URL)
    df = to_data_frame(stats_json)

    # Fetch advanced efficiency stats (OffRtg, DefRtg, Pace, TS%, etc.)
    adv_df = None
    try:
        adv_json = get_json_data(ADVANCED_DATA_URL)
        adv_raw = to_data_frame(adv_json)
        if not adv_raw.empty:
            available = [c for c in _ADVANCED_COLS if c in adv_raw.columns]
            adv_df = adv_raw[available] if available else None
    except Exception:
        adv_df = None

    schedule_df = load_schedule()
    today = datetime.today()
    data, todays_games_uo, frame_ml, home_team_odds, away_team_odds = create_todays_games_data(
        games, df, odds, schedule_df, today, interactive=False, adv_df=adv_df
    )

    # Fallback to local OddsData.sqlite for any missing OU / ML / Spread values.
    fallback = _load_today_odds_fallback(games, today)
    valid_games = [(h, a) for h, a in games if h in team_index_current and a in team_index_current]
    spreads = []
    for idx, (home_team, away_team) in enumerate(valid_games):
        fb = fallback.get((home_team, away_team), {})
        if todays_games_uo[idx] in (None, "") and fb.get("OU"):
            todays_games_uo[idx] = fb["OU"]
        if home_team_odds[idx] in (None, "") and fb.get("ML_Home"):
            home_team_odds[idx] = fb["ML_Home"]
        if away_team_odds[idx] in (None, "") and fb.get("ML_Away"):
            away_team_odds[idx] = fb["ML_Away"]

        spread_val = None
        if odds:
            game_odds = odds.get(f"{home_team}:{away_team}", {})
            raw_spread = game_odds.get("home_spread")
            # SBR convention: home_spread positive = home is underdog (receives pts).
            # Our convention (matching OddsData): positive = home is favorite (gives pts).
            if raw_spread is not None:
                spread_val = -float(raw_spread)
        if spread_val is None and fb.get("Spread") not in (None, ""):
            spread_val = float(fb["Spread"])
        spreads.append(spread_val)
    predictions = XGBoost_Runner.xgb_predict(
        data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds
    )
    # ATS model batch prediction for today's games.
    safe_spreads = [float(sp) if sp not in (None, "") else 0.0 for sp in spreads]
    advanced_df = None
    try:
        from src.Utils.AdvancedFeatures import merge_into as _merge_adv
        helper = pd.DataFrame({
            "TEAM_NAME": [g[0] for g in games],
            "TEAM_NAME.1": [g[1] for g in games],
            "Date": [today.date().isoformat()] * len(games),
        })
        helper = _merge_adv(helper, "TEAM_NAME", "TEAM_NAME.1", "Date")
        adv_cols = [c for c in helper.columns if c.startswith(("H_", "A_", "D_"))]
        if adv_cols:
            # Keep legacy columns first (preserves column positions for the
            # 175-feature model); new/experimental features go at the end so
            # _align_features() truncation drops them until retrained.
            # IMPORTANT: any feature added to AdvancedFeatures AFTER the 175-
            # feature model was trained must be listed here so it lands in
            # adv_new (appended last) rather than adv_old (inline), which would
            # shift A_/D_ feature positions and silently break the model.
            _NEW_ADV_STEMS = {
                "game_num_season", "month_sin", "month_cos",   # temporal
                "form_ats_pct_home_10", "form_ats_pct_away_10",  # home/away ATS split
                "form_pts_for_5", "form_pts_against_5", "form_pts_diff_10",  # off/def split
            }
            adv_old = [c for c in adv_cols if c[2:] not in _NEW_ADV_STEMS]
            adv_new = [c for c in adv_cols if c[2:] in _NEW_ADV_STEMS]
            advanced_df = helper[adv_old + adv_new].fillna(0.0)
    except Exception:
        advanced_df = None
    ats_probs = XGBoost_Runner.predict_ats_probs(frame_ml, safe_spreads, advanced=advanced_df)

    # Fetch injury reports for all teams via AI web search.
    injury_cache = {}
    today_iso = today.date().isoformat()
    try:
        from src.Utils.InjuryReport import get_game_injuries
        for home_team, away_team in valid_games:
            injury_data = get_game_injuries(home_team, away_team, today_iso)
            injury_cache[(home_team, away_team)] = injury_data
    except Exception:
        pass

    for idx, (pred, sp) in enumerate(zip(predictions, spreads)):
        pred["spread"] = float(sp) if sp not in (None, "") else None

        # Attach injury report.
        inj = injury_cache.get((pred["home_team"], pred["away_team"]))
        pred["injuries"] = inj if inj else None

        pred["home_profile"] = team_profile_for_date(pred["home_team"], today_iso)
        pred["away_profile"] = team_profile_for_date(pred["away_team"], today_iso)
        pred["ats_winner"] = None
        pred["ats_cover_margin"] = None

        # ATS model prediction.
        if ats_probs is not None and sp not in (None, ""):
            p_home_cover = float(ats_probs[idx, 1])
            pred["ats_model_pick"] = "home" if p_home_cover >= 0.5 else "away"
            pred["ats_model_home_prob"] = round(p_home_cover * 100, 1)
            pred["ats_model_confidence"] = round(max(p_home_cover, 1 - p_home_cover) * 100, 1)
            edge_pp = abs(p_home_cover - 0.5) * 100
            pred["ats_value_edge"] = round(edge_pp, 1)
            # Asymmetric threshold: home picks ≥8%, away picks ≥9%.
            # MCW=26 OOS: 86.7% combined (39/45, both seasons ≥86%).
            _is_away = p_home_cover < 0.5
            pred["ats_is_value"] = edge_pp >= _ats_value_threshold(today, is_away_pick=_is_away)
            # Away-pick quality filter: suppress away bets where the home team
            # is bad (NET<-3) but not hugely outclassed (D_NET>-7) at a
            # small spread (≤8).
            if pred["ats_is_value"] and _is_away:
                try:
                    _home_net = float(frame_ml.iloc[idx].get("ADV_NET_RATING", float("nan")))
                    _away_net = float(frame_ml.iloc[idx].get("ADV_NET_RATING.1", float("nan")))
                    _abs_sp = abs(float(sp))
                    if (not (pd.isna(_home_net) or pd.isna(_away_net)) and
                            _home_net < -3 and _abs_sp <= 8 and (_home_net - _away_net) > -7):
                        pred["ats_is_value"] = False
                        pred["ats_away_quality_filter"] = True
                except Exception:
                    pass
        else:
            pred["ats_model_pick"] = None
            pred["ats_model_home_prob"] = None
            pred["ats_model_confidence"] = None
            pred["ats_value_edge"] = None
            pred["ats_is_value"] = False

        # Value pick detection (ML moneyline).
        model_home_prob = pred["home_confidence"] / 100.0
        pred.update(evaluate_value(
            model_home_prob, pred.get("home_team_odds"), pred.get("away_team_odds"),
            spread=pred.get("spread"),
        ))

        # Consensus.
        ml_value_side = pred.get("value_side") if pred.get("is_value") else None
        ats_pick = pred.get("ats_model_pick")
        if ml_value_side and ats_pick and ml_value_side == ats_pick:
            pred["consensus_pick"] = ml_value_side
            pred["is_consensus"] = True
        else:
            pred["consensus_pick"] = None
            pred["is_consensus"] = False

        # Playoff series context (only meaningful in playoff window).
        try:
            from src.Utils.PlayoffContext import get_series_state, is_playoff_date
            if is_playoff_date(today_iso):
                state = get_series_state(pred["home_team"], pred["away_team"], today_iso)
                if state:
                    pred["is_playoff"] = True
                    pred["series_round"] = state["round_label"]
                    pred["round_num"] = state.get("round_num")
                    pred["series_game_num"] = state["series_game_num"]
                    pred["series_home_wins"] = state["home_wins"]
                    pred["series_away_wins"] = state["away_wins"]
                    pred["series_status_text"] = state["status_text"]
                    pred["series_is_elimination"] = state["is_elimination"]
                else:
                    pred["is_playoff"] = True
                    pred["series_round"] = None
                    pred["round_num"] = None
                    pred["series_game_num"] = None
                    pred["series_status_text"] = "尚無系列賽資料"
                    pred["series_is_elimination"] = False
            else:
                pred["is_playoff"] = False
        except Exception:
            pred["is_playoff"] = False

        # Playoff ATS strategy picks.
        try:
            if pred.get("is_playoff"):
                series_st = None
                try:
                    series_st = get_series_state(pred["home_team"], pred["away_team"], today_iso)
                except Exception:
                    pass
                team_form = _build_team_form(pred["home_team"], pred["away_team"], today_iso)
                ats_picks = evaluate_playoff_ats(pred, series_st, team_form)
                pred["playoff_ats_picks"] = picks_to_dict(ats_picks)
                bp = best_pick(ats_picks)
                pred["playoff_ats_best_side"] = bp.side if bp else None
                pred["playoff_ats_best_signal"] = bp.signal_name if bp else None
                pred["playoff_ats_best_tier"] = bp.tier if bp else None
                pred["playoff_ats_consensus"] = consensus_side(ats_picks)
                pred["playoff_ats_has_conflict"] = has_conflict(ats_picks)
                pred["playoff_ats_strong_consensus"] = strong_consensus_side(ats_picks)
            else:
                pred["playoff_ats_picks"] = []
                pred["playoff_ats_best_side"] = None
                pred["playoff_ats_best_signal"] = None
                pred["playoff_ats_best_tier"] = None
                pred["playoff_ats_consensus"] = None
                pred["playoff_ats_has_conflict"] = False
                pred["playoff_ats_strong_consensus"] = None
        except Exception:
            pred["playoff_ats_picks"] = []
            pred["playoff_ats_best_side"] = None
            pred["playoff_ats_best_signal"] = None
            pred["playoff_ats_best_tier"] = None
            pred["playoff_ats_consensus"] = None
            pred["playoff_ats_has_conflict"] = False
            pred["playoff_ats_strong_consensus"] = None
    return predictions


# Historical (past dates) — read features + actual outcomes from local SQLite,
# join money lines from OddsData.sqlite, and compare predictions to actuals.
import sqlite3  # noqa: E402

from src.Utils.TeamProfile import team_profile_for_date, grade_spread  # noqa: E402
from src.Utils.ValueFinder import evaluate_value  # noqa: E402
from src.Utils.PlayoffATSStrategy import evaluate_playoff_ats, best_pick, consensus_side, picks_to_dict, has_conflict, strong_consensus_side  # noqa: E402

def _build_team_form(home_team, away_team, date_str, pre_game=False):
    """Build team_form dict for PlayoffATSStrategy from AdvancedFeatures.

    pre_game=True uses strictly-before comparison to get entering form for
    historical games where the target date's result is already in the DB.
    """
    try:
        from src.Utils.AdvancedFeatures import build_feature_table
        table = build_feature_table()
        if table.empty:
            return None
        import pandas as pd
        target = pd.Timestamp(date_str)
        if pre_game:
            h_rows = table[(table["Team"] == home_team) & (table["Date"] < target)].tail(1)
            a_rows = table[(table["Team"] == away_team) & (table["Date"] < target)].tail(1)
        else:
            h_rows = table[(table["Team"] == home_team) & (table["Date"] <= target)].tail(1)
            a_rows = table[(table["Team"] == away_team) & (table["Date"] <= target)].tail(1)
        form = {}
        if not h_rows.empty:
            form["home_ats_l10"] = h_rows.iloc[0].get("form_ats_pct_10")
            form["home_w20"] = h_rows.iloc[0].get("form_w_pct_10")
        if not a_rows.empty:
            form["away_ats_l10"] = a_rows.iloc[0].get("form_ats_pct_10")
            form["away_w20"] = a_rows.iloc[0].get("form_w_pct_10")
        return form if form else None
    except Exception:
        return None


_DATASET_DB = os.path.join(PROJECT_ROOT, "Data", "dataset.sqlite")
_ODDS_DB = os.path.join(PROJECT_ROOT, "Data", "OddsData.sqlite")
_DATASET_TABLE = "dataset_2012-26"
_DROP_FEATURE_COLS = [
    "index", "Score", "Home-Team-Win", "TEAM_NAME", "Date",
    "index.1", "TEAM_NAME.1", "Date.1", "OU-Cover",
]


def _season_table_for_date(date_obj):
    """Return the OddsData season table key for a date (NBA season starts Oct)."""
    if date_obj.month >= 10:
        start = date_obj.year
    else:
        start = date_obj.year - 1
    return f"{start}-{(start + 1) % 100:02d}"


def predict_historical_xgb(target_date):
    """Predict + grade NBA games for a past date using locally cached data.

    Returns a list of dicts shaped like xgb_predict() output, plus extra fields:
        actual_home_win (bool|None), actual_winner ('home'|'away'|None),
        actual_total (float|None), actual_ou_result ('OVER'|'UNDER'|'PUSH'|None),
        ml_correct (bool|None), ou_correct (bool|None).
    """
    import pandas as pd

    from datetime import date as _date_cls
    _today = _date_cls.today()

    with sqlite3.connect(_DATASET_DB) as con:
        df = pd.read_sql_query(
            f'SELECT * FROM "{_DATASET_TABLE}" WHERE Date = ?',
            con,
            params=(target_date.isoformat(),),
        )

    # For future/today/recent dates (upcoming or just-played games not yet in dataset),
    # return game stubs from OddsData so playoff signals + injury reports still show.
    if df.empty and target_date >= _today - pd.Timedelta(days=2):
        season_table = _season_table_for_date(target_date)
        stubs = []
        try:
            with sqlite3.connect(_ODDS_DB) as _con:
                _odds = pd.read_sql_query(
                    f'SELECT * FROM "{season_table}" WHERE Date = ?',
                    _con,
                    params=(target_date.isoformat(),),
                )
            for _, row in _odds.iterrows():
                home_team = row["Home"]
                away_team = row["Away"]
                _pts = row.get("Points")
                _wm = row.get("Win_Margin")
                _played = pd.notna(_pts) and _pts and _pts > 0 and _wm is not None
                if _played:
                    _hs_score = int((_pts + _wm) / 2)
                    _as_score = int((_pts - _wm) / 2)
                else:
                    _hs_score = _as_score = None
                _spread = float(row["Spread"]) if pd.notna(row.get("Spread")) else None
                if _played and _spread is not None:
                    _cover_diff = _wm - _spread
                    if abs(_cover_diff) < 0.001:
                        _ats_w = "push"
                    elif _cover_diff > 0:
                        _ats_w = "home"
                    else:
                        _ats_w = "away"
                else:
                    _ats_w = None
                stub = {
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_score": _hs_score,
                    "away_score": _as_score,
                    "spread": _spread,
                    "Spread": _spread,
                    "OU": float(row["OU"]) if pd.notna(row.get("OU")) else None,
                    "ML_Home": row.get("ML_Home"),
                    "ML_Away": row.get("ML_Away"),
                    "Days_Rest_Home": row.get("Days_Rest_Home"),
                    "Days_Rest_Away": row.get("Days_Rest_Away"),
                    "home_confidence": None,
                    "away_confidence": None,
                    "value_side": None,
                    "is_value": False,
                    "ats_model_pick": None,
                    "ats_model_home_prob": None,
                    "ats_model_confidence": None,
                    "ats_value_edge": None,
                    "ats_is_value": False,
                    "is_historical": True,
                    "actual_home_win": bool(_wm > 0) if _played else None,
                    "actual_winner": ("home" if _wm > 0 else "away") if _played else None,
                    "actual_total": int(_pts) if _played else None,
                    "actual_ou_result": ("OVER" if _pts > float(row["OU"]) else "UNDER" if _pts < float(row["OU"]) else "PUSH") if _played and pd.notna(row.get("OU")) and row["OU"] else None,
                    "ml_correct": None,
                    "ou_correct": None,
                    "ats_winner": _ats_w,
                    "is_upcoming": not _played,
                    "is_playoff": False,
                    "playoff_ats_picks": [],
                    "playoff_ats_best_side": None,
                    "playoff_ats_best_signal": None,
                    "playoff_ats_best_tier": None,
                    "playoff_ats_consensus": None,
                    "playoff_ats_has_conflict": False,
                    "playoff_ats_strong_consensus": None,
                }
                # Apply playoff signals for upcoming games.
                # NOTE: do NOT re-import evaluate_playoff_ats etc. here — doing so
                # would make Python treat them as local vars for the whole function,
                # causing UnboundLocalError in the historical branch.
                try:
                    from src.Utils.PlayoffContext import is_playoff_date, get_series_state
                    target_iso = target_date.isoformat()
                    if is_playoff_date(target_iso):
                        state = get_series_state(home_team, away_team, target_iso, as_of_date=target_date)
                        if state:
                            stub["is_playoff"] = True
                            stub["series_round"] = state["round_label"]
                            stub["round_num"] = state.get("round_num")
                            stub["series_game_num"] = state["series_game_num"]
                            stub["series_home_wins"] = state["home_wins"]
                            stub["series_away_wins"] = state["away_wins"]
                            stub["series_status_text"] = state["status_text"]
                            stub["series_is_elimination"] = state["is_elimination"]
                        else:
                            stub["is_playoff"] = True
                        team_form = _build_team_form(home_team, away_team, target_iso, pre_game=False)
                        ats_picks = evaluate_playoff_ats(stub, state, team_form)
                        stub["playoff_ats_picks"] = picks_to_dict(ats_picks)
                        bp = best_pick(ats_picks)
                        stub["playoff_ats_best_side"] = bp.side if bp else None
                        stub["playoff_ats_best_signal"] = bp.signal_name if bp else None
                        stub["playoff_ats_best_tier"] = bp.tier if bp else None
                        stub["playoff_ats_consensus"] = consensus_side(ats_picks)
                        stub["playoff_ats_has_conflict"] = has_conflict(ats_picks)
                        stub["playoff_ats_strong_consensus"] = strong_consensus_side(ats_picks)
                except Exception:
                    pass
                stubs.append(stub)
        except Exception:
            pass
        return stubs

    if df.empty:
        return []

    # Try to attach money lines from OddsData season table.
    season_table = _season_table_for_date(target_date)
    odds_lookup = {}
    try:
        with sqlite3.connect(_ODDS_DB) as con:
            odds_df = pd.read_sql_query(
                f'SELECT Date, Home, Away, ML_Home, ML_Away, OU, Spread, Points, Win_Margin '
                f'FROM "{season_table}" WHERE Date = ?',
                con,
                params=(target_date.isoformat(),),
            )
        for _, row in odds_df.iterrows():
            key = (row["Home"], row["Away"])
            odds_lookup[key] = row
    except Exception:
        pass

    # Build the same arrays xgb_predict expects.
    games = []
    home_odds_list = []
    away_odds_list = []
    todays_uo = []
    actual_rows = []  # parallel to games
    for _, row in df.iterrows():
        home_team = row["TEAM_NAME"]
        away_team = row["TEAM_NAME.1"]
        games.append([home_team, away_team])
        odds_row = odds_lookup.get((home_team, away_team))
        if odds_row is not None:
            home_odds_list.append(odds_row.get("ML_Home"))
            away_odds_list.append(odds_row.get("ML_Away"))
            todays_uo.append(odds_row.get("OU") if pd.notna(odds_row.get("OU")) else row.get("OU"))
        else:
            home_odds_list.append(None)
            away_odds_list.append(None)
            todays_uo.append(row.get("OU"))
        actual_rows.append({
            "home_win": int(row["Home-Team-Win"]) if pd.notna(row.get("Home-Team-Win")) else None,
            "ou_cover": int(row["OU-Cover"]) if pd.notna(row.get("OU-Cover")) else None,
            "total_points": odds_row.get("Points") if odds_row is not None and pd.notna(odds_row.get("Points")) else None,
            "ou_value": todays_uo[-1],
        })

    # Build the feature frame keeping OU at its trained position.
    # ML model drops OU; UO model expects it. xgb_predict will rebuild
    # frame_uo with the right column layout via _build_frame_uo, so we keep
    # OU here so the column position matches what the trainer saw.
    frame_ml = df.drop(columns=_DROP_FEATURE_COLS, errors="ignore").copy()
    if "OU" in frame_ml.columns:
        ml_data = frame_ml.drop(columns=["OU"]).astype(float).to_numpy()
    else:
        ml_data = frame_ml.astype(float).to_numpy()
    data = ml_data

    predictions = XGBoost_Runner.xgb_predict(
        data, todays_uo, frame_ml, games, home_odds_list, away_odds_list
    )

    # Run the dedicated ATS model in a single batch.
    spreads_list = [
        odds_lookup.get((g[0], g[1]), {}).get("Spread") if odds_lookup.get((g[0], g[1])) is not None else None
        for g in games
    ]
    safe_spreads = [float(s) if s is not None and pd.notna(s) else 0.0 for s in spreads_list]

    # Inject AdvancedFeatures rows so the ATS model gets rolling form +
    # schedule density just like at training time.
    advanced_df = None
    try:
        from src.Utils.AdvancedFeatures import merge_into as _merge_advanced
        helper = pd.DataFrame({
            "TEAM_NAME": [g[0] for g in games],
            "TEAM_NAME.1": [g[1] for g in games],
            "Date": [target_date.isoformat()] * len(games),
        })
        helper = _merge_advanced(helper, "TEAM_NAME", "TEAM_NAME.1", "Date")
        advanced_cols = [c for c in helper.columns if c.startswith(("H_", "A_", "D_"))]
        if advanced_cols:
            # Legacy columns first so the 175-feature model's column positions
            # are preserved; new/experimental features appended at the end where
            # _align_features() truncation drops them until a new model is trained.
            # Any feature added after the 175-feature model was trained MUST be
            # listed here — omitting it shifts A_/D_ positions and breaks the model.
            _NEW_ADV_STEMS = {
                "game_num_season", "month_sin", "month_cos",   # temporal
                "form_ats_pct_home_10", "form_ats_pct_away_10",  # home/away ATS split
                "form_pts_for_5", "form_pts_against_5", "form_pts_diff_10",  # off/def split
            }
            adv_old = [c for c in advanced_cols if c[2:] not in _NEW_ADV_STEMS]
            adv_new = [c for c in advanced_cols if c[2:] in _NEW_ADV_STEMS]
            advanced_df = helper[adv_old + adv_new].fillna(0.0)
    except Exception:
        advanced_df = None
    ats_probs = XGBoost_Runner.predict_ats_probs(frame_ml, safe_spreads, advanced=advanced_df)

    # Annotate with actual outcomes + correctness + per-team scores.
    # Treat Points==0 as "not yet played" (Get_Odds_Data writes 0 placeholders
    # before tip-off when scraped early in the day).
    for idx, pred in enumerate(predictions):
        pred["is_historical"] = True
        home_team = pred["home_team"]
        away_team = pred["away_team"]
        odds_row = odds_lookup.get((home_team, away_team))

        # ATS model prediction (P that home team covers the spread).
        if ats_probs is not None and spreads_list[idx] is not None:
            p_home_cover = float(ats_probs[idx, 1])
            pred["ats_model_pick"] = "home" if p_home_cover >= 0.5 else "away"
            pred["ats_model_home_prob"] = round(p_home_cover * 100, 1)
            pred["ats_model_confidence"] = round(max(p_home_cover, 1 - p_home_cover) * 100, 1)
            # ATS edge: market implied is ~50% (vig-removed at -110 ≈ 0.5).
            # OOS analysis (2025-26 full season, 1214 games):
            #   ≥8% → 64.3% (56 games); ≥9% → 67.4% (43 games).
            # Regular season (Oct-Mar): ≥8% → 70.8%, ≥9% → 75.7%.
            # Late regular season (Mar / early Apr): suppressed — negative EV.
            # Playoffs (Apr 14+): ≥10%.
            edge_pp = abs(p_home_cover - 0.5) * 100
            pred["ats_value_edge"] = round(edge_pp, 1)
            _is_away = p_home_cover < 0.5
            pred["ats_is_value"] = edge_pp >= _ats_value_threshold(target_date, is_away_pick=_is_away)
            # Away-pick quality filter (same as today path — see above).
            if pred["ats_is_value"] and _is_away and spreads_list[idx] is not None:
                try:
                    _home_net = float(frame_ml.iloc[idx].get("ADV_NET_RATING", float("nan")))
                    _away_net = float(frame_ml.iloc[idx].get("ADV_NET_RATING.1", float("nan")))
                    _abs_sp = abs(float(spreads_list[idx]))
                    if (not (pd.isna(_home_net) or pd.isna(_away_net)) and
                            _home_net < -3 and _abs_sp <= 8 and (_home_net - _away_net) > -7):
                        pred["ats_is_value"] = False
                        pred["ats_away_quality_filter"] = True
                except Exception:
                    pass
        else:
            pred["ats_model_pick"] = None
            pred["ats_model_home_prob"] = None
            pred["ats_model_confidence"] = None
            pred["ats_value_edge"] = None
            pred["ats_is_value"] = False

        # CONSENSUS: ML moneyline value pick + ATS model agreement.
        # ML diamond skews to underdogs (avg award), ATS model skews to home —
        # the overlap is a stricter filter that catches "model thinks underdog
        # wins outright AND covers". When both fire, both bet sides win.
        ml_value_side = pred.get("value_side") if pred.get("is_value") else None
        ats_pick = pred.get("ats_model_pick")
        if ml_value_side and ats_pick and ml_value_side == ats_pick:
            pred["consensus_pick"] = ml_value_side
            pred["is_consensus"] = True
        else:
            pred["consensus_pick"] = None
            pred["is_consensus"] = False

        played = False
        total_points = None
        win_margin = None
        if odds_row is not None:
            pts = odds_row.get("Points")
            mar = odds_row.get("Win_Margin")
            if pd.notna(pts) and float(pts) > 0:
                played = True
                total_points = float(pts)
                win_margin = float(mar) if pd.notna(mar) else 0.0

        # Spread: positive = home favored by N, negative = home underdog.
        if odds_row is not None and pd.notna(odds_row.get("Spread")):
            pred["spread"] = float(odds_row["Spread"])
        else:
            pred["spread"] = None

        # Historical splits (weekday + opponent strength + ATS).
        target_iso = target_date.isoformat()
        pred["home_profile"] = team_profile_for_date(home_team, target_iso)
        pred["away_profile"] = team_profile_for_date(away_team, target_iso)

        # Value detection: model vs market.
        model_home_prob = pred["home_confidence"] / 100.0
        home_ml = odds_row.get("ML_Home") if odds_row is not None else None
        away_ml = odds_row.get("ML_Away") if odds_row is not None else None
        pred.update(evaluate_value(model_home_prob, home_ml, away_ml, spread=pred.get("spread")))

        if played:
            pred["home_score"] = int((total_points + win_margin) / 2)
            pred["away_score"] = int((total_points - win_margin) / 2)
            pred["actual_total"] = int(total_points)
            home_won = win_margin > 0
            pred["actual_home_win"] = home_won
            pred["actual_winner"] = "home" if home_won else "away"
            pred["ml_correct"] = pred["winner"] == pred["actual_winner"]

            # ATS: which side covered the spread?
            cover_side, cover_margin = grade_spread(pred["spread"], win_margin)
            pred["ats_winner"] = cover_side  # 'home' | 'away' | 'push' | None
            pred["ats_cover_margin"] = round(cover_margin, 1) if cover_margin is not None else None

            ou_value = odds_row.get("OU") if odds_row is not None else None
            if pd.notna(ou_value) and ou_value:
                if total_points > float(ou_value):
                    pred["actual_ou_result"] = "OVER"
                elif total_points < float(ou_value):
                    pred["actual_ou_result"] = "UNDER"
                else:
                    pred["actual_ou_result"] = "PUSH"
                if pred["actual_ou_result"] == "PUSH":
                    pred["ou_correct"] = None
                else:
                    pred["ou_correct"] = pred["ou_pick"] == pred["actual_ou_result"]
            else:
                pred["actual_ou_result"] = None
                pred["ou_correct"] = None
        else:
            pred["home_score"] = None
            pred["away_score"] = None
            pred["actual_total"] = None
            pred["actual_home_win"] = None
            pred["actual_winner"] = None
            pred["ml_correct"] = None
            pred["actual_ou_result"] = None
            pred["ou_correct"] = None
            pred["ats_winner"] = None
            pred["ats_cover_margin"] = None

        # Inject playoff series state for historical games too.
        # Always pass as_of_date=target_date so series_game_num counts only games
        # scheduled BEFORE target_date (avoids G2 signal firing on historical G1
        # games, and correctly labels future G3/G4 games that are in the dataset).
        _game_played = pred.get("home_score") is not None
        _as_of = target_date
        try:
            from src.Utils.PlayoffContext import is_playoff_date, get_series_state
            target_iso = target_date.isoformat()
            if is_playoff_date(target_iso):
                state = get_series_state(home_team, away_team, target_iso, as_of_date=_as_of)
                if state:
                    pred["is_playoff"] = True
                    pred["series_round"] = state["round_label"]
                    pred["round_num"] = state.get("round_num")
                    pred["series_game_num"] = state["series_game_num"]
                    pred["series_home_wins"] = state["home_wins"]
                    pred["series_away_wins"] = state["away_wins"]
                    pred["series_status_text"] = state["status_text"]
                    pred["series_is_elimination"] = state["is_elimination"]
                else:
                    pred["is_playoff"] = True
            else:
                pred["is_playoff"] = False
        except Exception:
            pred["is_playoff"] = False

        # Playoff ATS strategy picks.
        try:
            if pred.get("is_playoff"):
                series_st = None
                try:
                    series_st = get_series_state(home_team, away_team, target_date.isoformat(), as_of_date=_as_of)
                except Exception:
                    pass
                team_form = _build_team_form(home_team, away_team, target_date.isoformat(), pre_game=_game_played)
                ats_picks = evaluate_playoff_ats(pred, series_st, team_form)
                pred["playoff_ats_picks"] = picks_to_dict(ats_picks)
                bp = best_pick(ats_picks)
                pred["playoff_ats_best_side"] = bp.side if bp else None
                pred["playoff_ats_best_signal"] = bp.signal_name if bp else None
                pred["playoff_ats_best_tier"] = bp.tier if bp else None
                pred["playoff_ats_consensus"] = consensus_side(ats_picks)
                pred["playoff_ats_has_conflict"] = has_conflict(ats_picks)
                pred["playoff_ats_strong_consensus"] = strong_consensus_side(ats_picks)
            else:
                pred["playoff_ats_picks"] = []
                pred["playoff_ats_best_side"] = None
                pred["playoff_ats_best_signal"] = None
                pred["playoff_ats_best_tier"] = None
                pred["playoff_ats_consensus"] = None
                pred["playoff_ats_has_conflict"] = False
                pred["playoff_ats_strong_consensus"] = None
        except Exception:
            pred["playoff_ats_picks"] = []
            pred["playoff_ats_best_side"] = None
            pred["playoff_ats_best_signal"] = None
            pred["playoff_ats_best_tier"] = None
            pred["playoff_ats_consensus"] = None
            pred["playoff_ats_has_conflict"] = False
            pred["playoff_ats_strong_consensus"] = None

    # Append stubs for games in odds table but missing from the ML dataset.
    if True:
        predicted_teams = {(p["home_team"], p["away_team"]) for p in predictions}
        season_table = _season_table_for_date(target_date)
        try:
            with sqlite3.connect(_ODDS_DB) as _con:
                _odds = pd.read_sql_query(
                    f'SELECT * FROM "{season_table}" WHERE Date = ?',
                    _con,
                    params=(target_date.isoformat(),),
                )
            for _, row in _odds.iterrows():
                home_team = row["Home"]
                away_team = row["Away"]
                if (home_team, away_team) in predicted_teams:
                    continue
                _pts = row.get("Points")
                _wm = row.get("Win_Margin")
                _sp = float(row["Spread"]) if pd.notna(row.get("Spread")) else None
                _ou_v = float(row["OU"]) if pd.notna(row.get("OU")) else None
                _played = _pts is not None and pd.notna(_pts) and _pts > 0
                if _played:
                    _hs = round((_pts + _wm) / 2)
                    _as = round((_pts - _wm) / 2)
                    _ats_w = _ats_m = None
                    if _sp is not None:
                        _diff = _wm - _sp
                        if _diff > 0:
                            _ats_w, _ats_m = "home", round(_diff, 1)
                        elif _diff < 0:
                            _ats_w, _ats_m = "away", round(_diff, 1)
                        else:
                            _ats_w, _ats_m = "push", 0.0
                else:
                    _hs = _as = None
                    _ats_w = _ats_m = None
                stub = {
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_score": _hs,
                    "away_score": _as,
                    "spread": _sp,
                    "Spread": _sp,
                    "OU": _ou_v,
                    "ML_Home": row.get("ML_Home"),
                    "ML_Away": row.get("ML_Away"),
                    "Days_Rest_Home": row.get("Days_Rest_Home"),
                    "Days_Rest_Away": row.get("Days_Rest_Away"),
                    "home_confidence": None,
                    "away_confidence": None,
                    "value_side": None,
                    "is_value": False,
                    "ats_model_pick": None,
                    "ats_model_home_prob": None,
                    "ats_model_confidence": None,
                    "ats_value_edge": None,
                    "ats_is_value": False,
                    "is_historical": True,
                    "actual_home_win": _hs > _as if _played else None,
                    "actual_winner": (home_team if _wm > 0 else away_team) if _played else None,
                    "actual_total": int(_pts) if _played else None,
                    "actual_ou_result": ("OVER" if _pts > _ou_v else "UNDER") if (_played and _ou_v) else None,
                    "ml_correct": None,
                    "ou_correct": None,
                    "ats_winner": _ats_w,
                    "ats_cover_margin": _ats_m,
                    "is_upcoming": not _played,
                    "is_playoff": False,
                    "playoff_ats_picks": [],
                    "playoff_ats_best_side": None,
                    "playoff_ats_best_signal": None,
                    "playoff_ats_best_tier": None,
                    "playoff_ats_consensus": None,
                    "playoff_ats_has_conflict": False,
                    "playoff_ats_strong_consensus": None,
                }
                try:
                    from src.Utils.PlayoffContext import is_playoff_date, get_series_state
                    target_iso = target_date.isoformat()
                    if is_playoff_date(target_iso):
                        state = get_series_state(home_team, away_team, target_iso, as_of_date=target_date)
                        if state:
                            stub["is_playoff"] = True
                            stub["series_round"] = state["round_label"]
                            stub["round_num"] = state.get("round_num")
                            stub["series_game_num"] = state["series_game_num"]
                            stub["series_home_wins"] = state["home_wins"]
                            stub["series_away_wins"] = state["away_wins"]
                            stub["series_status_text"] = state["status_text"]
                            stub["series_is_elimination"] = state["is_elimination"]
                        else:
                            stub["is_playoff"] = True
                        team_form = _build_team_form(home_team, away_team, target_iso, pre_game=False)
                        ats_picks = evaluate_playoff_ats(stub, state, team_form)
                        stub["playoff_ats_picks"] = picks_to_dict(ats_picks)
                        bp = best_pick(ats_picks)
                        stub["playoff_ats_best_side"] = bp.side if bp else None
                        stub["playoff_ats_best_signal"] = bp.signal_name if bp else None
                        stub["playoff_ats_best_tier"] = bp.tier if bp else None
                        stub["playoff_ats_consensus"] = consensus_side(ats_picks)
                        stub["playoff_ats_has_conflict"] = has_conflict(ats_picks)
                        stub["playoff_ats_strong_consensus"] = strong_consensus_side(ats_picks)
                except Exception:
                    pass
                predictions.append(stub)
        except Exception:
            pass

    return predictions


def main(args):
    odds = None
    if args.odds:
        odds = SbrOddsProvider(sportsbook=args.odds).get_odds()
    games, odds = resolve_games(odds, args.odds)
    if games is None:
        return

    stats_json = get_json_data(DATA_URL)
    df = to_data_frame(stats_json)

    # Fetch advanced efficiency stats.
    adv_df = None
    try:
        adv_json = get_json_data(ADVANCED_DATA_URL)
        adv_raw = to_data_frame(adv_json)
        if not adv_raw.empty:
            available = [c for c in _ADVANCED_COLS if c in adv_raw.columns]
            adv_df = adv_raw[available] if available else None
    except Exception:
        adv_df = None

    schedule_df = load_schedule()
    today = datetime.today()
    data, todays_games_uo, frame_ml, home_team_odds, away_team_odds = create_todays_games_data(
        games, df, odds, schedule_df, today, adv_df=adv_df
    )

    if args.A:
        args.xgb = True
        args.nn = True

    normalized_data = _get_tf().keras.utils.normalize(data, axis=1) if args.nn else None
    run_models(
        data,
        normalized_data,
        todays_games_uo,
        frame_ml,
        games,
        home_team_odds,
        away_team_odds,
        args,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model to Run')
    parser.add_argument('-xgb', action='store_true', help='Run with XGBoost Model')
    parser.add_argument('-nn', action='store_true', help='Run with Neural Network Model')
    parser.add_argument('-A', action='store_true', help='Run all Models')
    parser.add_argument('-odds', help='Sportsbook to fetch from. (fanduel, draftkings, betmgm, pointsbet, caesars, wynn, bet_rivers_ny')
    parser.add_argument('-kc', action='store_true', help='Calculates percentage of bankroll to bet based on model edge')
    args = parser.parse_args()
    main(args)
