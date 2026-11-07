"""Load MovieLens data and create a temporal split."""

import pandas as pd
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

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
    Splits the dataset into train, validation and test sets, with 80/10/10 split respectively.
    :param ratings: Rating dataset
    :return: The train, validation and test sets
    """
    # Sort each user's ratings by time
    df = ratings.sort_values(
        ["user_id", "timestamp", "movie_id"]
    ).copy()

    # Calculate each rating's position within its user
    pos = df.groupby("user_id").cumcount()
    total = df.groupby("user_id")["rating"].transform("size")

    # Temporal 80/10/10 split per user
    train = df[pos < (total * 0.8).astype(int)].copy()
    val = df[(pos >= (total * 0.8).astype(int)) &
             (pos < (total * 0.9).astype(int))].copy()
    test = df[pos >= (total * 0.9).astype(int)].copy()

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
