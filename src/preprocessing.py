"""Load MovieLens data and create a temporal split."""

import pandas as pd
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MOVIES_PATH = DATA_DIR / "movies.dat"
USERS_PATH = DATA_DIR / "users.dat"
RATINGS_PATH = DATA_DIR / "ratings.dat"


def get_movies():
    """
    :return: Returns the movies dataset
    """
    return pd.read_csv(
        MOVIES_PATH,
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )


def get_ratings():
    """
    :return: Returns the ratings dataset
    """
    return pd.read_csv(
        RATINGS_PATH,
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )


def get_users():
    """
    :return: Returns the users dataset
    """
    return pd.read_csv(
        USERS_PATH,
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        dtype={"zip_code": "string"},
    )


def get_data():
    """
    Returns the datasets, and prints their shape
    :return: Movies, Ratings, Users datasets
    """
    return get_movies(), get_ratings(), get_users()


def split_data(ratings):
    """
    Splits the dataset into train, validation and test sets, with 80/10/10  global temporal split respectively.
    :param ratings: Rating dataset
    :return: The train, validation and test sets
    """
    logging.info(f"--------- Logging the training, validation and test splits: ---------")
    # Sort ratings by time:
    df = ratings.sort_values(
        ["timestamp", "user_id", "movie_id"]
    ).copy()

    # Find timestamps by which x% of all timestamps have occurred:
    timestamp_val = int(df["timestamp"].quantile(0.80))
    timestamp_test = int(df["timestamp"].quantile(0.90))

    logging.info(f"Validation split happens at: {pd.to_datetime(timestamp_val, unit="s", utc=True)}")
    logging.info(f"Test split happens at: {pd.to_datetime(timestamp_test, unit="s", utc=True)}")

    # Global Temporal 80/10/10 split
    train = df[df["timestamp"] < timestamp_val].copy()
    val = df[(df["timestamp"] >= timestamp_val) & (df["timestamp"] < timestamp_test)].copy()
    test = df[df["timestamp"] >= timestamp_test].copy()

    logging.info(f"Training ratings: {len(train)}")
    logging.info(f"Validation ratings: {len(val)}")
    logging.info(f"Testing ratings: {len(test)}")

    n_users_train = train["user_id"].nunique()
    n_users_val = val["user_id"].nunique()
    n_users_test = test["user_id"].nunique()

    logging.info(f"Users in training set: {n_users_train}")
    logging.info(f"Users in validation set: {n_users_val}")
    logging.info(f"Users in test set: {n_users_test}")

    return train, val, test


def get_cosine_similarity(train, movies, users):
    """
    Get the cosine similarity between users and movies
    :param train: Training set
    :param movies: Movies dataset
    :param users: Users dataset
    :return: The cosine similarity between users and movies and the user-movie interaction matrix X
    """
    # Treat ratings >= 4 as positive feedback
    positive = train[train["rating"] >= 4]

    movie_ids = movies["movie_id"].unique()
    movie_idx = {m: i for i, m in enumerate(movie_ids)}

    # User-movie interaction matrix
    X = csr_matrix(
        (np.ones(len(positive)),
         (positive["user_id"] - 1, positive["movie_id"].map(movie_idx))),
        shape=(users["user_id"].max(), len(movie_ids))
    )

    # Calculate movie-to-movie similarity
    similarity = cosine_similarity(X.T)
    np.fill_diagonal(similarity, 0)

    return similarity, X


def get_jaccard_similarity(X):
    """
    Get the jaccard similarity from the user-movie interaction matrix
    :param X: User movie interaction matrix
    :return: Jaccard similarity
    """
    co = (X.T @ X).toarray()
    counts = np.asarray(X.sum(axis=0)).ravel()
    union = counts[:, None] + counts[None, :] - co

    jaccard = np.divide(co, union, out=np.zeros_like(co),
                        where=union > 0)
    np.fill_diagonal(jaccard, 0)

    return jaccard
