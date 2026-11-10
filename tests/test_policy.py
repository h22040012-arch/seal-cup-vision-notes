import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from seal_vision.policy import Detection, decide, match_result


def row(label="front", score=0.9):
    return Detection(label, score, (0, 0, 100, 100))


class PolicyTests(unittest.TestCase):
    def test_front_and_back(self):
        self.assertTrue(decide([row()], threshold=0.65).accepted)
        self.assertFalse(decide([row("back")], threshold=0.65).accepted)

    def test_empty_and_below_threshold(self):
        self.assertEqual(decide([], threshold=0.65).reason, "no_detection")
        self.assertEqual(decide([row(score=0.64)], threshold=0.65).reason, "below_threshold")

    def test_boundary_and_tie_order(self):
        self.assertTrue(decide([row(score=0.65)], threshold=0.65).accepted)
        self.assertTrue(decide([row(), row("back")], threshold=0.65).accepted)
        self.assertFalse(decide([row("back"), row()], threshold=0.65).accepted)

    def test_multiple_candidates_choose_highest(self):
        self.assertTrue(decide([row("back", 0.7), row("front", 0.9)], threshold=0.65).accepted)

    def test_invalid_output_rejects_entire_request(self):
        for bad in [row(score=float("nan")), row(score=1.1), row("other"),
                    Detection("front", 0.9, (2, 2, 1, 1))]:
            with self.subTest(bad=bad):
                self.assertEqual(decide([row(), bad], threshold=0.65).reason, "invalid_model_output")

    def test_invalid_threshold(self):
        for value in [-1, 2, float("inf"), float("nan"), True]:
            with self.assertRaises(ValueError):
                decide([], threshold=value)

    def test_previous_ok_cannot_match_current_frame(self):
        ok = decide([row()], threshold=0.65)
        self.assertFalse(match_result("new", "new", "old", "old", ok).accepted)
        self.assertFalse(match_result("a", "new", "a", "old", ok).accepted)
        self.assertFalse(match_result("", "", "", "", ok).accepted)
        self.assertTrue(match_result("a", "b", "a", "b", ok).accepted)


if __name__ == "__main__":
    unittest.main()
