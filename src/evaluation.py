"""Precision@10, Recall@10, NDCG@10, and catalogue coverage."""
import numpy as np


def evaluate(recommend_fn, dataset, n_movies):
    """
    Evaluate the recommendation function, and return the precision, recall, ndcg and catalogue coverage.
    :param recommend_fn: Recommendation function
    :param dataset: The relevant dataset (train, test or val)
    :param n_movies: Total number of movies
    :return: Precision, recall, ndcg and catalogue coverage.
    """
    relevant = (
        dataset[dataset["rating"] >= 4]
        .groupby("user_id")["movie_id"]
        .apply(set)
        .to_dict()
    )

    precision, recall, ndcg = [], [], []
    recommended_movies = set()

    for user, actual in relevant.items():
        predicted = recommend_fn(user)[:10]
        hits = [int(m in actual) for m in predicted]

        precision.append(sum(hits) / 10)
        recall.append(sum(hits) / len(actual))

        dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
        idcg = sum(1 / np.log2(i + 2)
                   for i in range(min(10, len(actual))))
        ndcg.append(dcg / idcg)

        recommended_movies.update(predicted)

    return [
        np.mean(precision),
        np.mean(recall),
        np.mean(ndcg),
        len(recommended_movies) / n_movies
    ]

