"""IsotonicCalibrator: step-point interpolation and the [2%, 98%] probability clip."""
import numpy as np

from src.Utils.Calibration import IsotonicCalibrator


def _calibrator():
    # Shape of the pinned 2026-10-07 ML calibrator: bottom step 0.0, top step 1.0.
    return IsotonicCalibrator(x=[0.2, 0.2, 0.5, 0.5, 0.84, 0.84],
                              y=[0.0, 0.0, 0.6, 0.6, 1.0, 1.0])


def test_top_step_is_capped_below_certainty():
    probs = _calibrator().predict_proba(np.array([[0.1, 0.9]]))
    assert probs[0, 1] == 0.98
    assert round(probs[0, 1] * 100, 1) == 98.0  # what the card shows, never 100.0


def test_bottom_step_is_floored_above_zero():
    probs = _calibrator().predict_proba(np.array([[0.9, 0.1]]))
    assert probs[0, 1] == 0.02
    assert probs[0, 0] == 0.98


def test_interior_steps_are_unchanged_and_rows_sum_to_one():
    probs = _calibrator().predict_proba(np.array([[0.5, 0.5], [0.33, 0.67]]))
    assert probs[0, 1] == 0.6
    assert 0.6 < probs[1, 1] < 1.0
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_one_dimensional_input_is_treated_as_home_probability():
    probs = _calibrator().predict_proba(np.array([0.9]))
    assert probs.shape == (1, 2)
    assert probs[0, 1] == 0.98
