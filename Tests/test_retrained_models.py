"""The 2026-10-07 retrained boosters: pins, feature widths and the
sklearn-independent isotonic calibrator."""
import json
import unittest
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from src.Predict import XGBoost_Runner
from src.Utils import Policy
from src.Utils.Calibration import IsotonicCalibrator

REPO = Path(__file__).resolve().parents[1]


class TestIsotonicCalibrator(unittest.TestCase):
    def test_step_points_match_sklearn_transform(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0, 1, 400)
        y = (rng.uniform(0, 1, 400) < p).astype(int)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        cal = IsotonicCalibrator.from_isotonic(iso)
        q = np.linspace(-0.2, 1.2, 500)
        np.testing.assert_allclose(cal.transform(q), iso.transform(q), atol=1e-12)
        self.assertIsNone(cal.iso)

    def test_predict_proba_shape_and_complement(self):
        cal = IsotonicCalibrator(x=[0.0, 0.5, 1.0], y=[0.1, 0.5, 0.9])
        out = cal.predict_proba(np.array([[0.75, 0.25], [0.0, 1.0]]))
        self.assertEqual(out.shape, (2, 2))
        np.testing.assert_allclose(out.sum(axis=1), 1.0)
        np.testing.assert_allclose(out[:, 1], [0.3, 0.9 - 0.0], atol=1e-9)

    def test_legacy_pickle_path_still_works(self):
        iso = IsotonicRegression(out_of_bounds="clip").fit([0.0, 0.5, 1.0], [0.0, 0.4, 1.0])
        legacy = IsotonicCalibrator(iso)
        np.testing.assert_allclose(legacy.transform([0.25]), iso.transform([0.25]))


class TestPinnedRetrainedModels(unittest.TestCase):
    def test_pins_are_the_clean_retrain(self):
        self.assertIn("_ML_clean_", Policy.pinned_model_name("ML"))
        self.assertIn("_ATS_clean_", Policy.pinned_model_name("ATS"))

    def test_feature_widths_match_runner_layout(self):
        ml = xgb.Booster()
        ml.load_model(str(XGBoost_Runner._select_model_path("ML")))
        ats = xgb.Booster()
        ats.load_model(str(XGBoost_Runner._select_model_path("ATS")))
        self.assertEqual(ml.num_features(), 138)   # dataset columns minus the 10 non-feature columns
        self.assertEqual(ats.num_features(), 175)  # 138 + Spread + 36 rolling-form features

    def test_ml_calibrator_loads_as_step_points(self):
        cal = XGBoost_Runner._load_calibrator(XGBoost_Runner._select_model_path("ML"))
        self.assertIsInstance(cal, IsotonicCalibrator)
        self.assertIsNotNone(cal.x)
        self.assertTrue(np.all(np.diff(cal.x) >= 0))
        self.assertTrue(np.all(np.diff(cal.y) >= 0))

    def test_provenance_file_names_the_pins(self):
        prov = json.loads((REPO / "web" / "data" / "production_models.json").read_text(encoding="utf-8"))
        self.assertEqual(prov["models"]["ml"]["file"], Policy.pinned_model_name("ML"))
        self.assertEqual(prov["models"]["ats"]["file"], Policy.pinned_model_name("ATS"))
        # The % in the file name is the pooled walk-forward hit rate, not an in-sample score.
        for kind in ("ml", "ats"):
            m = prov["models"][kind]
            self.assertIn(f"{m['out_of_sample']['pooled']['pct']:.1f}%", m["file"])


if __name__ == "__main__":
    unittest.main()
