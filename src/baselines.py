"""Most-popular variants and item-based kNN baselines (Part 1)."""
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def recommend_popular(user_id, train, ranking, n=10):
    """
    Recommend top n popular movies for the given user, based on the provided ranking.
    :param user_id: The user's id to recommend movies
    :param train: The training dataset
    :param ranking: The ranking of the full movie catalogue, either based on all time popular or only positive ratings
    :param n: Number of items to recommend
    :return: Top n recommended movies for the user_id
    """
    seen = set(train.loc[train["user_id"] == user_id, "movie_id"])
    return [m for m in ranking if m not in seen][:n]


def recommend_knn(user_id, positive, movie_ids, movie_idx, neighbors, similarity, train, movies, n=10):
    """
    Recommend top n movies, based on the K nearest neighbors algorithm
    :param user_id: The user's id to recommend movies
    :param positive: Ratings that only have a positive rating (>=4)
    :param movie_ids: Unique movie ids
    :param movie_idx: Movie indices
    :param neighbors: K nearest neighbors based on the similarity matrix
    :param similarity: The similarity matrix
    :param train: The training dataset
    :param movies: The movie catalogue
    :param n: Number of items to recommend
    :return:
    """
    liked = positive[positive["user_id"] == user_id]["movie_id"]
    seen = train[train["user_id"] == user_id]["movie_id"]

    scores = np.zeros(len(movie_ids))

    for movie in liked:
        i = movie_idx[movie]
        nearby = neighbors[i]
        scores[nearby] += similarity[i, nearby]

    scores[[movie_idx[m] for m in seen]] = -np.inf

    if np.max(scores) <= 0:
        positive_counts = train[train["rating"] >= 4]["movie_id"].value_counts()

        all_movies = sorted(movies["movie_id"].unique())

        # Rank the full movie catalogue
        popular_positive = sorted(
            all_movies, key=lambda m: (-positive_counts.get(m, 0), m)
        )

        return recommend_popular(user_id, train, popular_positive, n)

    return movie_ids[np.argsort(-scores)[:n]].tolist()