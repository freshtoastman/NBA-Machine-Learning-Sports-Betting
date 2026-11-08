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
    """Pickles made before 2026-10-07 carry the sklearn IsotonicRegression itself
    (``iso``); newer ones store its step points as plain arrays so the file does
    not depend on the scikit-learn version that wrote it (the pipeline's .venv
    runs an older sklearn than the conda python that trains)."""

    # Isotonic regression maps its extreme bins to the bins' empirical rate, which
    # is exactly 0.0 / 1.0 when every training game in the top (bottom) bin was
    # won (lost) -- the 2026-10-07 ML calibrator's top step (raw >= 0.842) is
    # y = 1.0, so on 2026-11-08 BOS @ MIL rendered as "100.0%" and fed a p = 1
    # edge into the shadow value / EV columns. Those bins hold a few dozen games,
    # so the honest estimate is "very likely", not certain: clip to [2%, 98%].
    PROB_CLIP = 0.02

    def __init__(self, iso=None, x=None, y=None):
        self.iso = iso
        self.x = None if x is None else np.asarray(x, dtype=float)
        self.y = None if y is None else np.asarray(y, dtype=float)

    @classmethod
    def from_isotonic(cls, iso):
        return cls(x=iso.X_thresholds_, y=iso.y_thresholds_)

    def transform(self, p_home):
        p = np.asarray(p_home, dtype=float)
        if getattr(self, "x", None) is not None:
            # Same as IsotonicRegression(out_of_bounds="clip").transform: linear
            # interpolation between the fitted step points, clipped at both ends.
            return np.interp(p, self.x, self.y)
        return self.iso.transform(p)

    def predict_proba(self, raw_probs):
        raw = np.asarray(raw_probs, dtype=float)
        if raw.ndim == 1:
            p_home = raw
        else:
            p_home = raw[:, 1]
        cal = np.clip(self.transform(p_home), self.PROB_CLIP, 1 - self.PROB_CLIP)
        return np.column_stack([1.0 - cal, cal])
