"""Precision@10, Recall@10, NDCG@10, and catalogue coverage."""
import numpy as np
import logging


def evaluate(recommend_fn, dataset, n_movies, eval_users=None, show_completion=False):
    """
    Evaluate the recommendation function, and return the precision, recall, ndcg and catalogue coverage.
    :param recommend_fn: Recommendation function
    :param dataset: The relevant dataset (train, test or val)
    :param n_movies: Total number of movies
    :param eval_users: Optional set of user ids to evaluate. If None, every user with a relevant rating is evaluated
    :param show_completion: Show completion percentage
    :return: Precision, recall, ndcg and catalogue coverage.
    """
    relevant = (
        dataset[dataset["rating"] >= 4]
        .groupby("user_id")["movie_id"]
        .apply(set)
        .to_dict()
    )

    # Needed when we want to filter out users with no history in the relevant split:
    if eval_users is not None:
        relevant = {user: actual for user, actual in relevant.items() if user in eval_users}

    precision, recall, ndcg = [], [], []
    recommended_movies = set()

    count = 0
    half_point = False
    for user, actual in relevant.items():
        count += 1
        if show_completion:
            if count >= len(relevant) // 2 and not half_point:
                logging.info(f"Function evaluated {count / len(relevant) * 100:.2f}%")
                half_point = True
            elif count == len(relevant):
                logging.info(f"Function finished evaluating")

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

