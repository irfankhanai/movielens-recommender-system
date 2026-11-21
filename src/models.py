"""Additional recommendation algorithms (Part 2)."""
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'


def create_genre_matrix(movies):
    """
    Creates the genre matrix, which is the cosine similarity between each movie based on their genres
    :param movies: The movie dataframe
    :return: The cosine similarity matrix between each movie with regard to their genres, The dictionary of unique genres
    """
    unique_genres = {}

    c = 0 # Counter for the unique genre ids
    for movie_genre_pair in zip(movies["genres"], movies["movie_id"]):
        # Extract the different genres from the movie
        genres = movie_genre_pair[0].split("|")

        for genre in genres:
            # Create a new entry for genres if it is seen for the first time
            if genre not in unique_genres:
                unique_genres[genre] = c
                c += 1

    # Placeholder matrix that will hold 1s if the movie has the genre, 0s otherwise
    genre_matrix = np.zeros((max(movies["movie_id"]), len(unique_genres)))

    for movie_genre_pair in zip(movies["genres"], movies["movie_id"]):
        # Extract the different genres from the movie
        genres = movie_genre_pair[0].split("|")

        # Set the correct genre as 1
        for genre in genres:
            genre_matrix[movie_genre_pair[1] - 1, unique_genres[genre]] = 1

    # Get the cosine similarity, while setting the diagonal as 0 to remove the trivial similarities
    gen_sim = cosine_similarity(genre_matrix, genre_matrix)
    np.fill_diagonal(gen_sim, 0)

    return gen_sim, unique_genres


def content_based_recommendation(user_id, movies, train, gen_sim, diversify=False, n=10):
    """
    Recommend the top n movies for the given user based on only their preferences
    :param user_id: The user's id
    :param movies: The movie dataframe
    :param train: The training dataset
    :param gen_sim: The genre similarity matrix
    :param diversify: Boolean flag to return different genres
    :param n: Number of items to recommend
    :return: Top n recommended movies for the user_id
    """

    # Grab only the rated movies by the given user
    rated_movies = train.loc[train["user_id"] == user_id]

    # Place a marker to help filter the movies dataframe
    rated_movies['marker'] = rated_movies.apply(lambda x: 1, axis=1, result_type='reduce')

    # Merge movies with marked movies, making unrated movies to have Null value
    marked_movies = pd.merge(movies, rated_movies, on=['movie_id'], how='left')

    # Keep only the Null marked movies, dropping the marker column gives the users unrated movies.
    unrated_movies = marked_movies[pd.isnull(marked_movies['marker'])][movies.columns]

    predictions = []

    for movie_id in unrated_movies["movie_id"]:
        # score holds the weighted sum of the similarity matrix, scaled by the user's rating
        score = 0
        # sum_sim holds the unweighted sum of the similarity matrix
        sum_sim = 0

        # Traverse through user's rated movies, and calculate the sums accordingly
        for rated_movie in zip(rated_movies["movie_id"], rated_movies["rating"]):
            score += gen_sim[movie_id - 1, rated_movie[0] - 1] * rated_movie[1]
            sum_sim += gen_sim[movie_id - 1, rated_movie[0] - 1]

        # Trivial zero division check, and the floating point inaccuracy check
        if sum_sim != 0 and (score / sum_sim) < 5.0:
            predictions.append((score / sum_sim, movie_id))

    # Sort the results based on their predicted scores
    predictions = sorted(predictions, key=lambda x: x[0], reverse=True)

    if not diversify:
        # Only return the movie_id
        return [pred[1] for pred in predictions[:n]]

    result = []
    prev_genre = ""

    for p in predictions:
        # Top n recommendations achieved
        if len(result) == n:
            break

        # If the movie has the exact genre as the previous added movie, continue
        # Otherwise, add it to the results
        if movies[movies["movie_id"] == p[1]]["genres"].values[0] != prev_genre:
            result.append(p)
            prev_genre = movies[movies["movie_id"] == p[1]]["genres"].values[0]

    return [pred[1] for pred in result[:n]]
