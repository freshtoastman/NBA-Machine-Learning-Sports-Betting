import datetime
import unittest
from pathlib import Path
from unittest import mock

from src.Predict import XGBoost_Runner
from src.Utils import Policy


class PolicyTestCase(unittest.TestCase):
    def setUp(self):
        Policy.reload()
        self.addCleanup(Policy.reload)

    def _with_config(self, config):
        patcher = mock.patch.object(Policy, "_config", return_value=config)
        patcher.start()
        self.addCleanup(patcher.stop)


class TestAtsPolicy(PolicyTestCase):
    def test_value_flag_off_from_effective_date(self):
        self._with_config({"ats-policy": {"value_flag_enabled": False, "effective_from": "2026-10-02"}})
        pred = {"ats_is_value": True}
        Policy.apply_ats_policy(pred, datetime.date(2026, 10, 20))
        self.assertFalse(pred["ats_is_value"])
        self.assertTrue(pred["ats_shadow_value"])
        self.assertTrue(pred["ats_reference_only"])

    def test_dates_before_effective_date_are_not_rewritten(self):
        self._with_config({"ats-policy": {"value_flag_enabled": False, "effective_from": "2026-10-02"}})
        pred = {"ats_is_value": True}
        Policy.apply_ats_policy(pred, datetime.datetime(2026, 1, 10))
        self.assertTrue(pred["ats_is_value"])
        self.assertFalse(pred["ats_reference_only"])

    def test_enabled_policy_keeps_flag(self):
        self._with_config({"ats-policy": {"value_flag_enabled": True, "effective_from": "2026-10-02"}})
        pred = {"ats_is_value": True}
        Policy.apply_ats_policy(pred, "2026-11-01")
        self.assertTrue(pred["ats_is_value"])
        self.assertFalse(pred["ats_reference_only"])

    def test_missing_section_fails_closed(self):
        self._with_config({})
        pred = {"ats_is_value": True}
        Policy.apply_ats_policy(pred, "2026-11-01")
        self.assertFalse(pred["ats_is_value"])
        self.assertTrue(pred["ats_shadow_value"])

    def test_repo_config_has_flag_off(self):
        policy = Policy.ats_policy()
        self.assertFalse(policy["value_flag_enabled"])
        self.assertEqual(policy["effective_from"], "2026-10-02")


class TestMlValuePolicy(PolicyTestCase):
    CONFIG = {"ml-value-policy": {"value_flag_enabled": False, "effective_from": "2026-10-04"}}

    def test_flags_off_from_effective_date(self):
        self._with_config(self.CONFIG)
        pred = {"is_value": True, "is_golden": True, "value_side": "home", "value_edge": 6.1}
        Policy.apply_ml_value_policy(pred, datetime.date(2026, 10, 20))
        self.assertFalse(pred["is_value"])
        self.assertFalse(pred["is_golden"])
        self.assertTrue(pred["ml_shadow_value"])
        self.assertTrue(pred["ml_shadow_golden"])
        self.assertTrue(pred["ml_value_reference_only"])
        # Kept so the frozen record can be settled at the moneyline.
        self.assertEqual(pred["value_side"], "home")

    def test_dates_before_effective_date_are_not_rewritten(self):
        self._with_config(self.CONFIG)
        pred = {"is_value": True, "is_golden": False}
        Policy.apply_ml_value_policy(pred, "2026-04-10")
        self.assertTrue(pred["is_value"])
        self.assertFalse(pred["ml_value_reference_only"])
        self.assertFalse(pred["ml_shadow_golden"])

    def test_enabled_policy_keeps_flags(self):
        self._with_config({"ml-value-policy": {"value_flag_enabled": True, "effective_from": "2026-10-04"}})
        pred = {"is_value": True, "is_golden": True}
        Policy.apply_ml_value_policy(pred, "2026-11-01")
        self.assertTrue(pred["is_value"])
        self.assertTrue(pred["is_golden"])

    def test_missing_section_fails_closed(self):
        self._with_config({})
        pred = {"is_value": True, "is_golden": True}
        Policy.apply_ml_value_policy(pred, "2026-11-01")
        self.assertFalse(pred["is_value"])
        self.assertTrue(pred["ml_shadow_value"])

    def test_repo_config_has_flag_off(self):
        policy = Policy.ml_value_policy()
        self.assertFalse(policy["value_flag_enabled"])
        self.assertEqual(policy["effective_from"], "2026-10-04")


class TestPinnedModels(PolicyTestCase):
    def test_repo_pins_exist_on_disk(self):
        for kind in ("ML", "UO", "ATS"):
            name = Policy.pinned_model_name(kind)
            self.assertIsNotNone(name, kind)
            self.assertIn(f"_{kind}_", name)
            self.assertEqual(XGBoost_Runner._select_model_path(kind).name, name)

    def test_pin_wins_over_filename_accuracy(self):
        # 51.1% ATS file exists next to the 51.8% one; the pin must pick it anyway.
        lower = "XGBoost_51.1%_ATS_md4_eta0p033_sub0p657_col0p919_mcw16_g0p721_nb1544.json"
        self.assertTrue((XGBoost_Runner.MODEL_DIR / lower).exists())
        with mock.patch.object(XGBoost_Runner, "pinned_model_name", return_value=lower):
            self.assertEqual(XGBoost_Runner._select_model_path("ATS").name, lower)

    def test_missing_pin_raises_instead_of_falling_back(self):
        with mock.patch.object(XGBoost_Runner, "pinned_model_name", return_value="XGBoost_99.9%_ATS_nope.json"):
            with self.assertRaises(Policy.PinnedModelMissing):
                XGBoost_Runner._select_model_path("ATS")
        # FileNotFoundError is swallowed by the ATS loader; the pin error must not be.
        self.assertFalse(issubclass(Policy.PinnedModelMissing, FileNotFoundError))

    def test_no_pin_keeps_legacy_selection(self):
        with mock.patch.object(XGBoost_Runner, "pinned_model_name", return_value=None):
            path = XGBoost_Runner._select_model_path("ATS")
        self.assertIsInstance(path, Path)
        self.assertIn("51.8%", path.name)


if __name__ == "__main__":
    unittest.main()
