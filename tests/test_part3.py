"""Small hand-checkable tests for the Part 3 re-ranker and metrics."""

import unittest

import numpy as np
import pandas as pd

from src import part3_diversity as div


class Part3Tests(unittest.TestCase):
    def setUp(self):
        movies = pd.DataFrame({'genres': ['Action', 'Action|Sci-Fi', 'Comedy', 'Drama|Romance']})
        self.genres, self.names = div.genre_matrix(movies)
        self.distance = div.jaccard_distance(self.genres)

    def test_jaccard_distance_by_hand(self):
        # Action vs Action|Sci-Fi share 1 of 2 genres -> distance 0.5.
        self.assertAlmostEqual(self.distance[0, 1], 0.5)
        self.assertAlmostEqual(self.distance[0, 2], 1.0)
        self.assertAlmostEqual(self.distance[2, 2], 0.0)

    def test_lambda_one_keeps_base_order(self):
        candidates = [1, 0, 2, 3]
        self.assertEqual(div.mmr_rerank(candidates, [1, .9, .5, .4], self.distance, 1.0, n=4), candidates)

    def test_low_lambda_skips_near_duplicate(self):
        # Item 0 is very relevant but shares Action with item 1, so MMR prefers Comedy.
        picks = div.mmr_rerank([1, 0, 2, 3], [1, .95, .5, .4], self.distance, 0.3, n=2)
        self.assertEqual(picks, [1, 2])

    def test_ild(self):
        self.assertAlmostEqual(div.intra_list_diversity([0, 1, 2], self.distance), (0.5 + 1 + 1) / 3)

    def test_gini_extremes(self):
        self.assertAlmostEqual(div.gini([5, 5, 5, 5]), 0.0)
        self.assertAlmostEqual(div.gini([0, 0, 0, 8]), 0.75)  # (n - 1) / n for one item

    def test_calibration_zero_when_list_matches_history(self):
        p = div.genre_distribution([0, 2], self.genres)
        self.assertAlmostEqual(div.calibration_error(p, p), 0.0)
        self.assertGreater(div.calibration_error(p, div.genre_distribution([3], self.genres)), 0)

    def test_head_items(self):
        head = div.head_items(np.array([50, 30, 10, 10]))
        self.assertEqual(head.tolist(), [True, False, False, False])


if __name__ == '__main__':
    unittest.main()
