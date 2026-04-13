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
xgb_ats = None
xgb_ml_calibrator = None
xgb_uo_calibrator = None
xgb_ats_calibrator = None


def reset_model_cache():
    """Force the runner to re-select and reload models on the next call.

    Call this after retraining so Flask (or any long-running process) picks
    up the freshly saved model instead of the stale in-memory booster.
    """
    global xgb_ml, xgb_uo, xgb_ats
    global xgb_ml_calibrator, xgb_uo_calibrator, xgb_ats_calibrator
    xgb_ml = None
    xgb_uo = None
    xgb_ats = None
    xgb_ml_calibrator = None
    xgb_uo_calibrator = None
    xgb_ats_calibrator = None


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
    global xgb_ml, xgb_uo, xgb_ats
    global xgb_ml_calibrator, xgb_uo_calibrator, xgb_ats_calibrator
    if xgb_ml is None:
        ml_path = _select_model_path("ML")
        xgb_ml = xgb.Booster()
        xgb_ml.load_model(str(ml_path))
        xgb_ml_calibrator = _load_calibrator(ml_path)
    if xgb_uo is None:
        uo_path = _select_model_path("UO")
        xgb_uo = xgb.Booster()
        xgb_uo.load_model(str(uo_path))
        # Sigmoid calibration squashes UO probabilities so much that argmax
        # always picks the same class. Use raw booster output instead.
        xgb_uo_calibrator = None
    if xgb_ats is None:
        try:
            ats_path = _select_model_path("ATS")
            xgb_ats = xgb.Booster()
            xgb_ats.load_model(str(ats_path))
            # Sigmoid calibration squashes the (already weak) ATS signal to
            # near-constant 0.49. Use raw booster output to preserve range.
            xgb_ats_calibrator = None
        except FileNotFoundError:
            xgb_ats = False  # Sentinel: no model available; don't try again.
            xgb_ats_calibrator = None


def _build_frame_ats(frame_ml, spreads, advanced=None):
    """Build feature frame for the ATS model.

    The ATS trainer drops OU and appends Spread + advanced rolling features.
    `advanced` is an optional DataFrame indexed parallel to frame_ml with
    H_/A_/D_ columns from src.Utils.AdvancedFeatures. When provided, those
    columns are appended in the same order the trainer produced them.
    """
    frame_ats = frame_ml.copy()
    if "OU" in frame_ats.columns:
        frame_ats = frame_ats.drop(columns=["OU"])
    spread_array = np.asarray(spreads, dtype=float)
    if "Spread" in frame_ats.columns:
        frame_ats["Spread"] = spread_array
    else:
        frame_ats["Spread"] = spread_array
    if advanced is not None and len(advanced) == len(frame_ats):
        for col in advanced.columns:
            frame_ats[col] = advanced[col].values
    return frame_ats


def predict_ats_probs(frame_ml, spreads, advanced=None):
    """Return P(home_covers) for each game, or None if model unavailable.

    `advanced`: optional DataFrame of H_/A_/D_ rolling features parallel to
    frame_ml. If the loaded ATS model expects them, pass them; otherwise
    omit and the helper falls back to the basic feature layout.
    """
    _load_models()
    if xgb_ats is False or xgb_ats is None:
        return None
    frame_ats = _build_frame_ats(frame_ml, spreads, advanced=advanced)
    data = frame_ats.values.astype(float)
    return _predict_probs(xgb_ats, data, xgb_ats_calibrator)


def _align_features(model, data):
    """Backward-compat shim: if the dataset added new feature columns at the
    end (e.g. is_playoff + series cols) but the loaded model was trained
    against an older feature count, slice off the trailing extras so the
    prediction still works. The retraining step removes the need for this.
    """
    try:
        expected = int(getattr(model, "num_features", lambda: data.shape[1])())
    except Exception:
        expected = data.shape[1]
    if data.shape[1] > expected:
        return data[:, :expected]
    return data


def _predict_probs(model, data, calibrator=None):
    aligned = _align_features(model, data)
    raw = np.asarray(model.predict(xgb.DMatrix(aligned)))
    if calibrator is not None:
        return np.asarray(calibrator.predict_proba(raw))
    return raw


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


def _build_frame_uo(frame_ml, todays_games_uo):
    """Build the frame the UO model expects with OU in the trained position.

    The UO model was trained with OU located right before Days-Rest-Home in
    the dataset. xgb_predict / nn_runner historically appended OU to the end,
    which silently misaligned every column after position ~104 and made the
    model read garbage. We insert OU at the correct slot here.
    """
    frame_uo = frame_ml.copy()
    ou_array = np.asarray(todays_games_uo, dtype=float)
    if "OU" in frame_uo.columns:
        frame_uo["OU"] = ou_array
        return frame_uo
    if "Days-Rest-Home" in frame_uo.columns:
        insert_pos = frame_uo.columns.get_loc("Days-Rest-Home")
    else:
        insert_pos = len(frame_uo.columns)
    frame_uo.insert(insert_pos, "OU", ou_array)
    return frame_uo


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

    frame_uo = _build_frame_uo(frame_ml, todays_games_uo)

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

    frame_uo = _build_frame_uo(frame_ml, todays_games_uo)

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
