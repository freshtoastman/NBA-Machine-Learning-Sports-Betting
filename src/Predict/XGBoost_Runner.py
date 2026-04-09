import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from colorama import Fore, Style, init, deinit
from src.Utils import Expected_Value
from src.Utils import Kelly_Criterion as kc


init()

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "Models" / "XGBoost_Models"
ACCURACY_PATTERN = re.compile(r"XGBoost_(\d+(?:\.\d+)?)%_")

xgb_ml = None
xgb_uo = None
xgb_ml_calibrator = None
xgb_uo_calibrator = None


def _select_model_path(kind):
    candidates = list(MODEL_DIR.glob(f"*{kind}*.json"))
    if not candidates:
        raise FileNotFoundError(f"No XGBoost {kind} model found in {MODEL_DIR}")

    def score(path):
        match = ACCURACY_PATTERN.search(path.name)
        accuracy = float(match.group(1)) if match else 0.0
        return (path.stat().st_mtime, accuracy)

    return max(candidates, key=score)


def _load_calibrator(model_path):
    calibration_path = model_path.with_name(f"{model_path.stem}_calibration.pkl")
    if not calibration_path.exists():
        return None
    try:
        return joblib.load(calibration_path)
    except Exception:
        return None


def _load_models():
    global xgb_ml, xgb_uo, xgb_ml_calibrator, xgb_uo_calibrator
    if xgb_ml is None:
        ml_path = _select_model_path("ML")
        xgb_ml = xgb.Booster()
        xgb_ml.load_model(str(ml_path))
        xgb_ml_calibrator = _load_calibrator(ml_path)
    if xgb_uo is None:
        uo_path = _select_model_path("UO")
        xgb_uo = xgb.Booster()
        xgb_uo.load_model(str(uo_path))
        xgb_uo_calibrator = _load_calibrator(uo_path)


def _predict_probs(model, data, calibrator=None):
    if calibrator is not None:
        return calibrator.predict_proba(data)
    return model.predict(xgb.DMatrix(data))


def _format_game_line(home_team, away_team, winner_is_home, winner_confidence, under_over, ou_value, ou_confidence):
    winner_team = home_team if winner_is_home else away_team
    loser_team = away_team if winner_is_home else home_team
    winner_color = Fore.GREEN if winner_is_home else Fore.RED
    loser_color = Fore.RED if winner_is_home else Fore.GREEN
    ou_label = "UNDER" if under_over == 0 else "OVER"
    ou_color = Fore.MAGENTA if under_over == 0 else Fore.BLUE
    return (
        f"{winner_color}{winner_team}{Style.RESET_ALL}"
        f"{Fore.CYAN} ({winner_confidence}%)"
        f"{Style.RESET_ALL} vs {loser_color}{loser_team}{Style.RESET_ALL}: "
        f"{ou_color}{ou_label} {Style.RESET_ALL}{ou_value}"
        f"{Style.RESET_ALL}{Fore.CYAN} ({ou_confidence}%)"
        f"{Style.RESET_ALL}"
    )


def _print_expected_value(
    games,
    ml_predictions_array,
    home_team_odds,
    away_team_odds,
    kelly_criterion,
):
    if kelly_criterion:
        print("------------Expected Value & Kelly Criterion-----------")
    else:
        print("---------------------Expected Value--------------------")
    for idx, game in enumerate(games):
        home_team, away_team = game
        ev_home = ev_away = 0
        if home_team_odds[idx] and away_team_odds[idx]:
            ev_home = float(
                Expected_Value.expected_value(
                    ml_predictions_array[idx][1],
                    int(home_team_odds[idx]),
                )
            )
            ev_away = float(
                Expected_Value.expected_value(
                    ml_predictions_array[idx][0],
                    int(away_team_odds[idx]),
                )
            )
        expected_value_colors = {
            "home_color": Fore.GREEN if ev_home > 0 else Fore.RED,
            "away_color": Fore.GREEN if ev_away > 0 else Fore.RED,
        }
        bankroll_descriptor = " Fraction of Bankroll: "
        if home_team_odds[idx] and away_team_odds[idx]:
            bankroll_fraction_home = bankroll_descriptor + str(
                kc.calculate_kelly_criterion(home_team_odds[idx], ml_predictions_array[idx][1])
            ) + "%"
            bankroll_fraction_away = bankroll_descriptor + str(
                kc.calculate_kelly_criterion(away_team_odds[idx], ml_predictions_array[idx][0])
            ) + "%"
        else:
            bankroll_fraction_home = bankroll_descriptor + "n/a"
            bankroll_fraction_away = bankroll_descriptor + "n/a"

        print(
            home_team
            + " EV: "
            + expected_value_colors["home_color"]
            + str(ev_home)
            + Style.RESET_ALL
            + (bankroll_fraction_home if kelly_criterion else "")
        )
        print(
            away_team
            + " EV: "
            + expected_value_colors["away_color"]
            + str(ev_away)
            + Style.RESET_ALL
            + (bankroll_fraction_away if kelly_criterion else "")
        )


def xgb_predict(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds):
    """Run XGBoost prediction and return structured results (no printing).

    Returns a list of dicts, one per game, with keys:
        away_team, home_team,
        away_confidence, home_confidence,
        ou_pick ('OVER'|'UNDER'), ou_value, ou_confidence,
        away_team_ev, home_team_ev,
        away_team_odds, home_team_odds.
    """
    _load_models()

    frame_uo = frame_ml.copy()
    frame_uo["OU"] = np.asarray(todays_games_uo, dtype=float)

    ml_predictions_array = _predict_probs(xgb_ml, data, xgb_ml_calibrator)
    ou_predictions_array = _predict_probs(
        xgb_uo, frame_uo.values.astype(float), xgb_uo_calibrator
    )

    results = []
    for idx, game in enumerate(games):
        home_team, away_team = game
        ml_probs = ml_predictions_array[idx]
        ou_probs = ou_predictions_array[idx]
        winner = int(np.argmax(ml_probs))
        under_over = int(np.argmax(ou_probs))

        home_odds = home_team_odds[idx]
        away_odds = away_team_odds[idx]
        ev_home = ev_away = 0.0
        if home_odds and away_odds:
            ev_home = float(Expected_Value.expected_value(ml_probs[1], int(home_odds)))
            ev_away = float(Expected_Value.expected_value(ml_probs[0], int(away_odds)))

        results.append({
            "home_team": home_team,
            "away_team": away_team,
            "home_confidence": round(float(ml_probs[1]) * 100, 1),
            "away_confidence": round(float(ml_probs[0]) * 100, 1),
            "winner": "home" if winner == 1 else "away",
            "ou_pick": "OVER" if under_over == 1 else "UNDER",
            "ou_value": todays_games_uo[idx],
            "ou_confidence": round(float(ou_probs[under_over]) * 100, 1),
            "home_team_ev": ev_home,
            "away_team_ev": ev_away,
            "home_team_odds": int(home_odds) if home_odds else None,
            "away_team_odds": int(away_odds) if away_odds else None,
            "home_kelly": kc.calculate_kelly_criterion(home_odds, ml_probs[1]) if (home_odds and away_odds) else None,
            "away_kelly": kc.calculate_kelly_criterion(away_odds, ml_probs[0]) if (home_odds and away_odds) else None,
        })

    return results


def xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, kelly_criterion):
    _load_models()

    frame_uo = frame_ml.copy()
    frame_uo["OU"] = np.asarray(todays_games_uo, dtype=float)

    try:
        ml_predictions_array = _predict_probs(xgb_ml, data, xgb_ml_calibrator)
        ou_predictions_array = _predict_probs(
            xgb_uo,
            frame_uo.values.astype(float),
            xgb_uo_calibrator,
        )

        for idx, game in enumerate(games):
            home_team, away_team = game
            winner = int(np.argmax(ml_predictions_array[idx]))
            under_over = int(np.argmax(ou_predictions_array[idx]))
            winner_confidence = round(ml_predictions_array[idx][winner] * 100, 1)
            ou_confidence = round(ou_predictions_array[idx][under_over] * 100, 1)

            print(
                _format_game_line(
                    home_team,
                    away_team,
                    winner_is_home=(winner == 1),
                    winner_confidence=winner_confidence,
                    under_over=under_over,
                    ou_value=todays_games_uo[idx],
                    ou_confidence=ou_confidence,
                )
            )

        _print_expected_value(
            games,
            ml_predictions_array,
            home_team_odds,
            away_team_odds,
            kelly_criterion,
        )
    finally:
        deinit()
