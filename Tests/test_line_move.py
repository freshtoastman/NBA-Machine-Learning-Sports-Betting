import unittest

from src.Utils.LineMove import line_move_alert


class TestLineMoveAlert(unittest.TestCase):
    def test_small_move_is_not_flagged(self):
        self.assertIsNone(line_move_alert(6.5, 7.5, "home"))
        self.assertIsNone(line_move_alert(-3.5, -3.5, "away"))

    def test_missing_line_is_not_flagged(self):
        self.assertIsNone(line_move_alert(None, 3.5))
        self.assertIsNone(line_move_alert(3.5, None))

    def test_big_move_away_from_the_model_pick(self):
        # 2026-11-05 MIA @ GSW: 7.5 -> 3.5 while the model still said GSW 83.6%.
        alert = line_move_alert(7.5, 3.5, "home")
        self.assertEqual(alert, {"points": 4.0, "flip": False, "late_side": "away",
                                 "against_model": True, "mixed_source": False})

    def test_big_move_toward_the_model_pick(self):
        alert = line_move_alert(-9.5, -7.5, "home")
        self.assertEqual(alert["late_side"], "home")
        self.assertFalse(alert["against_model"])

    def test_flip_is_flagged_even_when_short(self):
        # +0.5 -> -0.5 is one point but the favourite changed.
        alert = line_move_alert(0.5, -0.5, "away")
        self.assertTrue(alert["flip"])
        self.assertEqual(alert["points"], 1.0)
        self.assertFalse(alert["against_model"])

    def test_odds_only_stub_has_no_model_side(self):
        # 2026-11-06 PHI @ CLE: CLE -1.5 at the opener, PHI -2.5 later.
        alert = line_move_alert(1.5, -2.5, None)
        self.assertTrue(alert["flip"])
        self.assertEqual(alert["late_side"], "away")
        self.assertIsNone(alert["against_model"])

    def test_opener_tag_is_the_same_book(self):
        alert = line_move_alert(7.5, 3.5, "home", "sbr_fanduel_open", "sbr_fanduel")
        self.assertFalse(alert["mixed_source"])

    def test_opener_from_another_book_is_marked(self):
        # 2026-10-25 TOR @ MIN: Fanatics opener TOR -5, FanDuel MIN -1.5.
        alert = line_move_alert(-5.0, 1.5, "home", "sbr_fanatics_open", "sbr_fanduel")
        self.assertTrue(alert["flip"])
        self.assertTrue(alert["mixed_source"])
        self.assertNotIn("same_source_points", alert)

    def test_same_book_move_behind_a_fallback_opener(self):
        # 2026-11-06 TOR @ MIL: DraftKings opener -7.5, FanDuel -7.5 -> -7.5 -> -4.5.
        history = [{"spread": -7.5, "source": "sbr_draftkings_open"},
                   {"spread": -7.5, "source": "sbr_fanduel"},
                   {"spread": -7.5, "source": "sbr_fanduel"},
                   {"spread": -4.5, "source": "sbr_fanduel"}]
        alert = line_move_alert(-7.5, -4.5, "away", "sbr_draftkings_open", "sbr_fanduel", history)
        self.assertTrue(alert["mixed_source"])
        self.assertEqual(alert["same_source_points"], 3.0)

    def test_same_book_points_show_a_book_difference(self):
        # Fanatics-style artifact: the current book never moved at all.
        history = [{"spread": -5.0, "source": "sbr_fanatics_open"},
                   {"spread": 1.5, "source": "sbr_fanduel"}]
        alert = line_move_alert(-5.0, 1.5, "home", "sbr_fanatics_open", "sbr_fanduel", history)
        self.assertEqual(alert["same_source_points"], 0.0)

    def test_same_book_points_only_when_sources_differ(self):
        history = [{"spread": 7.5, "source": "sbr_fanduel_open"},
                   {"spread": 3.5, "source": "sbr_fanduel"}]
        alert = line_move_alert(7.5, 3.5, "home", "sbr_fanduel_open", "sbr_fanduel", history)
        self.assertNotIn("same_source_points", alert)


if __name__ == "__main__":
    unittest.main()
