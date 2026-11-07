"""Explore data"""
import matplotlib.pyplot as plt
import pandas as pd


def print_statistics(movies, ratings, users):
    """
    Calculate and print the sparsity/density of the user-movie matrix
    :param movies: Movies dataset
    :param ratings: Rating dataset
    :param users: Users dataset
    :return:
    """
    n_users = len(users)
    n_movies = movies["movie_id"].nunique()
    n_ratings = len(ratings)

    density = (n_ratings / (n_users * n_movies)) * 100

    print("Total Users:", n_users)
    print("Total Movies:", n_movies)
    print("Total Ratings:", n_ratings)
    print(f"Dataset Density: {density:.2f}%")
    print()


def show_rating_distribution(ratings):
    """
    Calculate and show the rating distribution
    :param ratings: Rating dataset
    :return:
    """
    # Count how frequent each rating (1-5) appears in the dataset
    rating_counts = ratings["rating"].value_counts().sort_index()

    # Create and show the plot
    plt.figure(figsize=(7, 4))
    plt.bar(rating_counts.index, rating_counts.values)

    plt.xlabel("Rating (1–5)")
    plt.ylabel("Number of Ratings")
    plt.title("MovieLens 1M - Rating Distribution")
    plt.xticks([1, 2, 3, 4, 5])
    plt.tight_layout()
    plt.show()


def show_movie_popularity(ratings):
    """
    Calculate and show the movie popularity (long-tail distribution)
    :param ratings: Rating dataset
    :return:
    """
    # Count ratings per movie
    movie_popularity = ratings.groupby("movie_id").size()

    # Movie popularity (long-tail distribution)
    plt.plot(
        sorted(movie_popularity, reverse=True)
    )
    plt.title("Movie Popularity - Long Tail")
    plt.xlabel("Movies (Ranked by Popularity)")
    plt.ylabel("Number of Ratings")

    plt.tight_layout()
    plt.show()

    print("Average ratings per movie:", round(movie_popularity.mean(), 2))
    print()


def show_user_activity(ratings):
    """
    Calculate and show the user activity (long-tail distribution)
    :param ratings: Rating dataset
    :return:
    """
    # Count ratings per user
    user_activity = ratings.groupby("user_id").size()

    # User activity distribution
    plt.hist(user_activity, bins=50)
    plt.title("Ratings per User")
    plt.xlabel("Number of Ratings")
    plt.ylabel("Number of Users")

    plt.tight_layout()
    plt.show()

    print("Average ratings per user:", round(user_activity.mean(), 2))
    print()


def show_rating_activity(ratings):
    """
    Calculate and show the rating activity
    :param ratings: Rating dataset
    :return:
    """
    # Convert timestamps to dates
    ratings["date"] = pd.to_datetime(
        ratings["timestamp"], unit="s", utc=True
    )

    # Time span
    print("First Rating:", ratings["date"].min())
    print("Last Rating:", ratings["date"].max())
    print()

    # Monthly rating activity
    monthly_ratings = ratings.set_index("date").resample("MS").size()

    plt.figure(figsize=(10, 4))
    plt.plot(monthly_ratings.index, monthly_ratings.values)

    plt.title("Rating Activity Over Time")
    plt.xlabel("Date")
    plt.ylabel("Number of Ratings")
    plt.tight_layout()
    plt.show()


def show_statistics(movies, ratings, users):
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
    show_rating_distribution(ratings)
    show_movie_popularity(ratings)
    show_user_activity(ratings)
    show_rating_activity(ratings)
