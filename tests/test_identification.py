import unittest

import numpy as np

from rearranger.identification import get_all_transition_points


class TransitionPointTest(unittest.TestCase):
    def test_short_diagonal_does_not_hide_later_valid_diagonal(self):
        patterns = np.zeros((20, 20))
        current_boundaries = (0, 10)
        candidate_boundaries = (15, 19)

        patterns[9, 14] = 0.2
        patterns[11, 16] = 0.7
        patterns[12, 17] = 0.8

        points = get_all_transition_points(
            current_boundaries=current_boundaries,
            candidate_boundaries=candidate_boundaries,
            radius=2,
            patterns=patterns,
        )

        self.assertEqual(points, [(2, 0.8)])


if __name__ == "__main__":
    unittest.main()
