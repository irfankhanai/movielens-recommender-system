"""Explore data"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from pathlib import Path


def save_plot(name, output_dir=None, show=True):
    """Save the current figure and optionally display it, then release it."""
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(output_dir) / f"{name}.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()



def print_statistics(movies, ratings, users):
    """
    Calculate and print the sparsity/density of the user-movie matrix
    :param movies: Movies dataset
    :param ratings: Rating dataset
    :param users: Users dataset
    :return:
    """
    logging.info(f"--------- Logging statistics about the dataset: ---------")

    n_users = len(users)
    n_movies = movies["movie_id"].nunique()
    n_ratings = len(ratings)
    n_movies_with_ratings = ratings["movie_id"].nunique()

    density = (n_ratings / (n_users * n_movies)) * 100
    density_adjusted = (n_ratings / (n_users * n_movies_with_ratings)) * 100

    logging.info(f"Total Users: {n_users}")
    logging.info(f"Total Movies: {n_movies}")
    logging.info(f"Total Movies with ratings: {n_movies_with_ratings}")
    logging.info(f"Total Ratings: {n_ratings}")
    logging.info(f"Dataset Density: {density:.2f}%")
    logging.info(f"Dataset Density (adjusted): {density_adjusted:.2f}%")


def show_rating_distribution(ratings, output_dir=None, show=True):
    """
    Calculate and show the rating distribution
    :param ratings: Rating dataset
    :return:
    """
    logging.info(f"--------- Logging the rating distribution: ---------")

    # Count how frequent each rating (1-5) appears in the dataset
    rating = ratings["rating"]
    rating_counts = rating.value_counts().sort_index()

    # Gather overall stats on ratings
    mean_rating = rating.mean()
    median_rating = rating.median()
    total_ratings = rating.count()

    # Output non-visual metrics
    logging.info(f"Mean Rating: {mean_rating:.2f}")
    logging.info(f"Median Rating: {median_rating:.2f}")
    for score, n in rating_counts.items():
        logging.info(f"Rating {score} has {n} occurrences, which is {n / total_ratings * 100:.2f}%")

    # Create and show the plot
    plt.figure(figsize=(7, 4))
    plt.bar(rating_counts.index, rating_counts.values)

    plt.xlabel("Rating (1–5)")
    plt.ylabel("Number of Ratings")
    plt.title("MovieLens 1M - Rating Distribution")
    plt.xticks([1, 2, 3, 4, 5])
    plt.tight_layout()
    save_plot("rating_distribution", output_dir, show)


def show_movie_popularity(ratings, output_dir=None, show=True):
    """
    Calculate and show the movie popularity (long-tail distribution)
    :param ratings: Rating dataset
    :return:
    """
    logging.info(f"--------- Logging the movie popularity distribution: ---------")

    # Count ratings per movie
    movie_popularity = ratings.groupby("movie_id").size()
    total_movies = len(movie_popularity)
    total_ratings = movie_popularity.sum()

    logging.info(f"Average ratings per movie: {round(movie_popularity.mean(), 2)}")
    logging.info(f"Median ratings per movie: {round(movie_popularity.median(), 2)}")

    # Count how many movies have bellow X ratings
    under_5 = (movie_popularity < 5).sum()
    under_10 = (movie_popularity < 10).sum()
    under_30 = (movie_popularity < 30).sum()

    logging.info(f"Movies with < 5 ratings:  {under_5:,} ({under_5 / total_movies * 100:.2f}%)")
    logging.info(f"Movies with < 10 ratings: {under_10:,} ({under_10 / total_movies * 100:.2f}%)")
    logging.info(f"Movies with < 30 ratings: {under_30:,} ({under_30 / total_movies * 100:.2f}%)")

    # Count how many movies capture most of the ratings:
    popularity_sorted = movie_popularity.sort_values(ascending=False)
    cumulative_ratings = popularity_sorted.cumsum()

    movies_5 =  (cumulative_ratings <= 0.05 * total_ratings).sum() + 1
    movies_10 = (cumulative_ratings <= 0.10 * total_ratings).sum() + 1
    movies_50 = (cumulative_ratings <= 0.50 * total_ratings).sum() + 1
    movies_80 = (cumulative_ratings <= 0.80 * total_ratings).sum() + 1

    logging.info(f"Top {movies_5} movies ({movies_5 / total_movies * 100:.2f}% of rated catalog) account for 5% of ratings")
    logging.info(f"Top {movies_10} movies ({movies_10 / total_movies * 100:.2f}% of rated catalog) account for 10% of ratings")
    logging.info(f"Top {movies_50} movies ({movies_50 / total_movies * 100:.2f}% of rated catalog) account for 50% of ratings")
    logging.info(f"Top {movies_80} movies ({movies_80 / total_movies * 100:.2f}% of rated catalog) account for 80% of ratings")

    # Plot movie popularity (long-tail distribution)
    plt.plot(
        sorted(movie_popularity, reverse=True)
    )
    plt.title("Movie Popularity - Long Tail")
    plt.xlabel("Movies (Ranked by Popularity)")
    plt.ylabel("Number of Ratings")
    plt.tight_layout()
    save_plot("movie_popularity", output_dir, show)

    ranks = np.arange(1, len(movie_popularity) + 1)
    plt.loglog(ranks, popularity_sorted.to_numpy())
    plt.title("Movie Popularity - (log-log)")
    plt.xlabel("Movies (Ranked by Popularity, log scale)")
    plt.ylabel("Number of Ratings (log scale)")
    save_plot("movie_popularity_log", output_dir, show)


def show_user_activity(ratings, output_dir=None, show=True):
    """
    Calculate and show the user activity (long-tail distribution)
    :param ratings: Rating dataset
    :return:
    """
    logging.info(f"--------- Logging the user activity distribution: ---------")

    # Count ratings per user
    user_activity = ratings.groupby("user_id").size()

    total_users = len(user_activity)
    total_ratings = user_activity.sum()

    # Overview stats
    logging.info(f"Average ratings per user: {round(user_activity.mean(), 2)}")
    logging.info(f"Median ratings per user: {round(user_activity.median(), 2)}")
    logging.info(f"Max ratings (most active):  {user_activity.max()}")
    logging.info(f"Min ratings (least active): {user_activity.min()}")

    # User activity distribution:
    under_20 = (user_activity == 20).sum()
    under_50 = (user_activity < 50).sum()
    over_100 = (user_activity >= 100).sum()
    over_500 = (user_activity >= 500).sum()

    logging.info(f"Users with 20 ratings:   {under_20} ({under_20 / total_users * 100:.2f}%)")
    logging.info(f"Users with < 50 ratings:  {under_50} ({under_50 / total_users * 100:.2f}%)")
    logging.info(f"Users with >= 100 ratings: {over_100} ({over_100 / total_users * 100:.2f}%)")
    logging.info(f"Users with  >= 500 ratings: {over_500} ({over_500 / total_users * 100:.2f}%)")

    # How many users responsible for X amount of ratings
    activity_sorted = user_activity.sort_values(ascending=False)
    cumulative_ratings = activity_sorted.cumsum()

    users_50 = (cumulative_ratings <= 0.50 * total_ratings).sum() + 1
    users_80 = (cumulative_ratings <= 0.80 * total_ratings).sum() + 1

    logging.info(f"Top {users_50} users ({users_50 / total_users * 100:.2f}% of users) generate 50% of all ratings")
    logging.info(f"Top {users_80} users ({users_80 / total_users * 100:.2f}% of users) generate 80% of all ratings")

    # Plot user activity distribution
    plt.hist(user_activity, bins=50)
    plt.title("Ratings per User")
    plt.xlabel("Number of Ratings")
    plt.ylabel("Number of Users")

    plt.tight_layout()
    save_plot("user_activity", output_dir, show)


    bins = np.logspace(np.log10(user_activity.min()), np.log10(user_activity.max()), 40)
    plt.hist(user_activity, bins=bins)
    plt.xscale("log")
    plt.title("Ratings per User (log)")
    plt.xlabel("Number of Ratings (log scale)")
    plt.ylabel("Number of Users")
    save_plot("user_activity_log", output_dir, show)


def show_rating_activity(ratings, output_dir=None, show=True):
    """
    Calculate and show the rating activity
    :param ratings: Rating dataset
    :return:
    """
    logging.info(f"--------- Logging the rating activity distribution: ---------")

    # Convert timestamps to dates
    ratings["date"] = pd.to_datetime(
        ratings["timestamp"], unit="s", utc=True
    )

    # Time span
    first_date = ratings["date"].min()
    last_date = ratings["date"].max()
    duration_days = (last_date - first_date).days

    logging.info(f"First Rating: {first_date}")
    logging.info(f"Last Rating: {last_date}")
    logging.info(f"Duration (in days): {duration_days}")

    # Monthly rating activity
    monthly_ratings = ratings.set_index("date").resample("MS").size()

    # Busiest month
    busiest_month = monthly_ratings.idxmax()
    logging.info(f"Busiest month: {busiest_month.strftime('%Y-%m')} "
          f"({monthly_ratings.max() / len(ratings) * 100:.2f}% of ratings)")

    # User activity length:
    per_user = ratings.groupby("user_id")["timestamp"]
    span_days = (per_user.max() - per_user.min()) / 86400 # 86400 = 24h
    for days in [1, 7, 30]:
        share = (span_days <= days).mean() * 100
        logging.info(f"Users with all ratings within {days} day(s): {share:.2f}%")

    # Plot the monthly activity
    plt.figure(figsize=(10, 4))
    plt.plot(monthly_ratings.index, monthly_ratings.values)

    plt.title("Rating Activity Over Time")
    plt.xlabel("Date")
    plt.ylabel("Number of Ratings")
    plt.tight_layout()
    save_plot("rating_activity", output_dir, show)


def show_binary_feedback_statistics(ratings, movies, users, threshold=4):
    """
    Show how the data looks after the feedback treatment (keeping only ratings >= threshold)
    :param ratings: Rating dataset the treatment is applied to (e.g. the training set)
    :param movies: Movies dataset
    :param users: Users dataset
    :param threshold: Minimum rating counted as positive
    :return:
    """
    logging.info(f"--------- Logging the binary feedback statistics (rating >= {threshold}): ---------")

    positive = ratings[ratings["rating"] >= threshold]

    n_ratings = len(ratings)
    n_positive = len(positive)
    n_users = len(users)
    n_movies = movies["movie_id"].nunique()

    # What the treatment removes
    logging.info(f"Ratings: {n_ratings}, positive: {n_positive} ({n_positive / n_ratings * 100:.2f}%)")
    dropped = ratings[ratings["rating"] < threshold]["rating"].value_counts().sort_index()
    for score, n in dropped.items():
        logging.info(f"Rating {score} dropped: {n} ({n / n_ratings * 100:.2f}% of ratings)")

    density_before = n_ratings / (n_users * n_movies) * 100
    density_after = n_positive / (n_users * n_movies) * 100
    logging.info(f"Density (all movies): {density_before:.2f}% -> {density_after:.2f}%")

    # What it does to users
    user_all = ratings.groupby("user_id").size()
    user_pos = positive.groupby("user_id").size().reindex(user_all.index, fill_value=0)

    logging.info(f"Users with a rating: {len(user_all)}, with no positive rating: {(user_pos == 0).sum()}")
    logging.info(f"Median per user: {user_all.median():.0f} ratings -> {user_pos.median():.0f} positives")
    logging.info(f"Users with < 5 positives: {(user_pos < 5).sum()} ({(user_pos < 5).mean() * 100:.2f}%)")

    # What it does to movies
    movie_all = ratings.groupby("movie_id").size()
    movie_pos = positive.groupby("movie_id").size().reindex(movie_all.index, fill_value=0)

    logging.info(f"Movies with no positive rating: {n_movies - positive['movie_id'].nunique()} of {n_movies}")
    logging.info(f"Median per movie: {movie_all.median():.1f} ratings -> {movie_pos.median():.1f} positives")
    for thr in [10, 30]:
        before = (movie_all < thr).sum()
        after = (movie_pos < thr).sum()
        logging.info(f"Movies with < {thr}: {before} ({before / len(movie_all) * 100:.2f}%) "
                     f"-> {after} ({after / len(movie_all) * 100:.2f}%)")

    # Do the two popularity variants differ?
    top_all = movie_all.sort_values(ascending=False, kind="stable")
    top_pos = movie_pos.sort_values(ascending=False, kind="stable")
    for n in [10, 100]:
        overlap = len(set(top_all.head(n).index) & set(top_pos.head(n).index)) / n
        logging.info(f"Popularity (all) vs (positive), top {n} overlap: {overlap * 100:.0f}%")


def show_statistics(movies, ratings, users, output_dir=None, show=True):
    """
    Show all the statistics regarding the dataset, namely:
        - Print statistics
        - Show rating distribution
        - Show movie popularity
        - Show user activity
        - Show rating activity
    :return:
    """
    print_statistics(movies, ratings, users)
    show_rating_distribution(ratings, output_dir, show)
    show_movie_popularity(ratings, output_dir, show)
    show_user_activity(ratings, output_dir, show)
    show_rating_activity(ratings, output_dir, show)
