"""Load MovieLens data and create a temporal split."""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MOVIES_PATH = DATA_DIR / "movies.dat"
USERS_PATH = DATA_DIR / "users.dat"
RATINGS_PATH = DATA_DIR / "ratings.dat"


def get_movies():
    return pd.read_csv(
        MOVIES_PATH,
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )


def get_ratings():
    return pd.read_csv(
        RATINGS_PATH,
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )


def get_users():
    return pd.read_csv(
        USERS_PATH,
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        dtype={"zip_code": "string"},
    )


def get_data():
    return get_movies(), get_ratings(), get_users()
