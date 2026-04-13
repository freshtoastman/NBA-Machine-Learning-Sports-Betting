"""Isotonic calibrator for XGBoost home-win probabilities.

Pickled instances of this class sit next to the booster (same stem +
`_calibration.pkl`) and are loaded by `_load_calibrator` in XGBoost_Runner.

Contract: `predict_proba(raw_probs)` where `raw_probs` is the (n, 2) matrix
returned by `booster.predict(DMatrix)`. Returns a (n, 2) matrix with the
home-win probability (column 1) remapped through the isotonic fit and the
away probability set to `1 - p_home`.
"""
from __future__ import annotations

import numpy as np


class IsotonicCalibrator:
    def __init__(self, iso):
        self.iso = iso

    def predict_proba(self, raw_probs):
        raw = np.asarray(raw_probs, dtype=float)
        if raw.ndim == 1:
            p_home = raw
        else:
            p_home = raw[:, 1]
        cal = np.clip(self.iso.transform(p_home), 1e-6, 1 - 1e-6)
        return np.column_stack([1.0 - cal, cal])
