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
    n_movies_with_ratings = ratings["movie_id"].nunique()

    density = (n_ratings / (n_users * n_movies)) * 100
    density_adjusted = (n_ratings / (n_users * n_movies_with_ratings)) * 100

    print("Total Users:", n_users)
    print("Total Movies:", n_movies)
    print("Total Movies with ratings:", n_movies_with_ratings)
    print("Total Ratings:", n_ratings)
    print(f"Dataset Density: {density:.2f}%")
    print(f"Dataset Density (adjusted): {density_adjusted:.2f}%")
    print()


def show_rating_distribution(ratings):
    """
    Calculate and show the rating distribution
    :param ratings: Rating dataset
    :return:
    """
    # Count how frequent each rating (1-5) appears in the dataset
    rating = ratings["rating"]
    rating_counts = rating.value_counts().sort_index()

    # Gather overall stats on ratings
    mean_rating = rating.mean()
    median_rating = rating.median()
    total_ratings = rating.count()

    # Output non-visual metrics
    print(f"Mean Rating: {mean_rating:.2f}")
    print(f"Median Rating: {median_rating:.2f}")
    for score, n in rating_counts.items():
        print(f"Rating {score} has {n} occurrences, which is {n / total_ratings * 100:.2f}%")

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
    total_movies = len(movie_popularity)
    total_ratings = movie_popularity.sum()

    print("Average ratings per movie:", round(movie_popularity.mean(), 2))
    print("Median ratings per movie:", round(movie_popularity.median(), 2))

    # Count how many movies have bellow X ratings
    under_5 = (movie_popularity < 5).sum()
    under_10 = (movie_popularity < 10).sum()
    under_30 = (movie_popularity < 30).sum()

    print(f"Movies with < 5 ratings:  {under_5:,} ({under_5 / total_movies * 100:.2f}%)")
    print(f"Movies with < 10 ratings: {under_10:,} ({under_10 / total_movies * 100:.2f}%)")
    print(f"Movies with < 30 ratings: {under_30:,} ({under_30 / total_movies * 100:.2f}%)")

    # Count how many movies capture most of the ratings:
    popularity_sorted = movie_popularity.sort_values(ascending=False)
    cumulative_ratings = popularity_sorted.cumsum()

    movies_5 =  (cumulative_ratings <= 0.05 * total_ratings).sum() + 1
    movies_10 = (cumulative_ratings <= 0.10 * total_ratings).sum() + 1
    movies_50 = (cumulative_ratings <= 0.50 * total_ratings).sum() + 1
    movies_80 = (cumulative_ratings <= 0.80 * total_ratings).sum() + 1

    print(f"Top {movies_5} movies ({movies_5 / total_movies * 100:.2f}% of rated catalog) account for 5% of ratings")
    print(f"Top {movies_10} movies ({movies_10 / total_movies * 100:.2f}% of rated catalog) account for 10% of ratings")
    print(f"Top {movies_50} movies ({movies_50 / total_movies * 100:.2f}% of rated catalog) account for 50% of ratings")
    print(f"Top {movies_80} movies ({movies_80 / total_movies * 100:.2f}% of rated catalog) account for 80% of ratings")

    # Plot movie popularity (long-tail distribution)
    plt.plot(
        sorted(movie_popularity, reverse=True)
    )
    plt.title("Movie Popularity - Long Tail")
    plt.xlabel("Movies (Ranked by Popularity)")
    plt.ylabel("Number of Ratings")

    plt.tight_layout()
    plt.show()

    print()


def show_user_activity(ratings):
    """
    Calculate and show the user activity (long-tail distribution)
    :param ratings: Rating dataset
    :return:
    """
    # Count ratings per user
    user_activity = ratings.groupby("user_id").size()

    total_users = len(user_activity)
    total_ratings = user_activity.sum()

    # Overview stats
    print("Average ratings per user:", round(user_activity.mean(), 2))
    print("Median ratings per user:", round(user_activity.median(), 2))
    print(f"Max ratings (most active):  {user_activity.max()}")
    print(f"Min ratings (least active): {user_activity.min()}")

    # User activity distribution:
    under_20 = (user_activity == 20).sum()
    under_50 = (user_activity < 50).sum()
    over_100 = (user_activity >= 100).sum()
    over_500 = (user_activity >= 500).sum()

    print(f"Users with 20 ratings:   {under_20} ({under_20 / total_users * 100:.2f}%)")
    print(f"Users with < 50 ratings:  {under_50} ({under_50 / total_users * 100:.2f}%)")
    print(f"Users with >= 100 ratings: {over_100} ({over_100 / total_users * 100:.2f}%)")
    print(f"Users with  >= 500 ratings: {over_500} ({over_500 / total_users * 100:.2f}%)")

    # How many users responsible for X amount of ratings
    activity_sorted = user_activity.sort_values(ascending=False)
    cumulative_ratings = activity_sorted.cumsum()

    users_50 = (cumulative_ratings <= 0.50 * total_ratings).sum() + 1
    users_80 = (cumulative_ratings <= 0.80 * total_ratings).sum() + 1

    print(f"Top {users_50} users ({users_50 / total_users * 100:.2f}% of users) generate 50% of all ratings")
    print(f"Top {users_80} users ({users_80 / total_users * 100:.2f}% of users) generate 80% of all ratings")

    # Plot user activity distribution
    plt.hist(user_activity, bins=50)
    plt.title("Ratings per User")
    plt.xlabel("Number of Ratings")
    plt.ylabel("Number of Users")

    plt.tight_layout()
    plt.show()

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
    first_date = ratings["date"].min()
    last_date = ratings["date"].max()
    duration_days = (last_date - first_date).days

    print("First Rating:", first_date)
    print("Last Rating:", last_date)
    print("Duration (in days):", duration_days)
    print()

    # Monthly rating activity
    monthly_ratings = ratings.set_index("date").resample("MS").size()

    # Busiest month
    busiest_month = monthly_ratings.idxmax()
    print(f"Busiest month: {busiest_month.strftime('%Y-%m')} "
          f"({monthly_ratings.max() / len(ratings) * 100:.2f}% of ratings)")

    # User activity length:
    per_user = ratings.groupby("user_id")["timestamp"]
    span_days = (per_user.max() - per_user.min()) / 86400 # 86400 = 24h
    for days in [1, 7, 30]:
        share = (span_days <= days).mean() * 100
        print(f"Users with all ratings within {days} day(s): {share:.2f}%")

    # Plot the monthly activity
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
