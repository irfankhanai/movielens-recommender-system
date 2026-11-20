"""Small correctness checks, not extra model experiments."""

import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score, precision_score, recall_score

from src.models import ModelData, UserKNN, ContentProfile, Hybrid, ItemKNN
from src.part2_evaluation import per_user, mean_ci, paired_test
from src.baselines import recommend_knn


class Part2Tests(unittest.TestCase):
    def setUp(self):
        self.movies = pd.DataFrame({'movie_id': [11, 22, 33, 44, 55],
                                    'title': ['A (1990)', 'B (1991)', 'C (1980)', 'D (1981)', 'E (2000)'],
                                    'genres': ['Drama', 'Drama', 'Comedy', 'Comedy', 'Drama|Comedy']})
        self.users = pd.DataFrame({'user_id': [1, 2, 3, 4]})
        self.train = pd.DataFrame([(1, 11, 5), (1, 33, 1), (2, 11, 4), (2, 22, 5),
                                   (3, 33, 5), (3, 44, 4)], columns=['user_id', 'movie_id', 'rating'])
        self.data = ModelData(self.train, self.movies, self.users)

    def test_content_hand_calculation(self):
        model = ContentProfile(self.data, year_weight=0.25)
        scores = model.scores(1)
        self.assertAlmostEqual(scores[0], 1)
        self.assertAlmostEqual(scores[1], 1)
        self.assertAlmostEqual(scores[2], 0)
        self.assertAlmostEqual(scores[4], 0.75/np.sqrt(2))

    def test_neighbors_exclude_self(self):
        scores = UserKNN(self.data, k=1).scores(1)
        np.testing.assert_allclose(scores, [1/np.sqrt(2), 1/np.sqrt(2), 0, 0, 0])

    def test_seen_cold_and_unique(self):
        for model in [UserKNN(self.data), ContentProfile(self.data), Hybrid(self.data), ItemKNN(self.data, k=2)]:
            recs = model.recommend(1)
            self.assertFalse({11, 33} & set(recs))  # Low-rated movies are also seen.
            self.assertEqual(len(recs), len(set(recs)))
            self.assertEqual(model.recommend(4), self.data.rank(4, self.data.popular))
            self.assertEqual(model.recommend(999), self.data.rank(999, self.data.popular))

    def test_original_seen_fixed_during_ablation(self):
        reduced = ModelData(self.train.iloc[2:], self.movies, self.users, seen_train=self.train)
        self.assertFalse({11, 33} & set(ContentProfile(reduced).recommend(1)))

    def test_metrics_against_sklearn_and_hand_values(self):
        truth = np.zeros(12)
        truth[[0, 2, 10]] = 1
        heldout = pd.DataFrame({'user_id': [1, 1, 1], 'movie_id': [0, 2, 10], 'rating': [4, 5, 4]})
        result = per_user(lambda _: list(range(10)), heldout).iloc[0]
        predicted = np.array([1]*10+[0]*2)
        self.assertAlmostEqual(result['Precision@10'], precision_score(truth, predicted))
        self.assertAlmostEqual(result['Recall@10'], recall_score(truth, predicted))
        self.assertAlmostEqual(result['NDCG@10'], ndcg_score(truth[None], np.arange(12, 0, -1)[None], k=10))
        self.assertEqual(mean_ci(np.zeros(4)), (0, 0))

    def test_item_baseline_matches_part1(self):
        from sklearn.metrics.pairwise import cosine_similarity
        sim = cosine_similarity(self.data.x.T)
        np.fill_diagonal(sim, 0)
        neighbors = np.argsort(sim, axis=1)[:, -2:]
        original = recommend_knn(1, self.train[self.train.rating >= 4], self.data.items,
                                 self.data.item_index, neighbors, sim, self.train, self.movies, n=3)
        self.assertEqual(ItemKNN(self.data, k=2).recommend(1), original)

    def test_paired_test_null_and_clear_difference(self):
        self.assertEqual(paired_test(np.zeros(20), repeats=201), 1)
        self.assertLess(paired_test(np.ones(20), repeats=1000), 0.01)


if __name__ == '__main__':
    unittest.main()
