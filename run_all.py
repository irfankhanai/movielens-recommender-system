"""Project entry point (starter placeholder; no experiments implemented yet).

Final target:
    python run_all.py --data-dir ./data/ml-1m
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from src import preprocessing as pp
from src import analysis as analysis
from src import baselines as base
from src import evaluation as eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce MovieLens recommender experiments")
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to extracted ml-1m data")
    args = parser.parse_args()
    if not args.data_dir.is_dir():
        parser.error(f"Dataset directory does not exist: {args.data_dir}")

    ####################################################################################################################

    # Load the datasets, and print their sizes
    movies, ratings, users = pp.get_data()
    print("Movies:", movies.shape)
    print("Ratings:", ratings.shape)
    print("Users:", users.shape)

    print("\nSample Ratings:")
    print(ratings.head())

    # Show the relevant statistics regarding the datasets
    analysis.show_statistics(movies, ratings, users)

    # Split the ratings dataset into training, validation
    # and testing datasets using 80/10/10 split
    train, val, test = pp.split_data(ratings)
    print("Training ratings:", len(train))
    print("Validation ratings:", len(val))
    print("Testing ratings:", len(test))
    print()

    ####################################################################################################################

    # Popularity scores using training data only
    all_counts = train["movie_id"].value_counts()
    positive_counts = train[train["rating"] >= 4]["movie_id"].value_counts()

    all_movies = sorted(movies["movie_id"].unique())

    # Rank the full movie catalogue
    popular_all = sorted(
        all_movies, key=lambda m: (-all_counts.get(m, 0), m)
    )
    popular_positive = sorted(
        all_movies, key=lambda m: (-positive_counts.get(m, 0), m)
    )

    # Example recommendations for User 1
    print("All Ratings:", base.recommend_popular(1, train, popular_all, n=10))
    print("Positive Ratings:", base.recommend_popular(1, train, popular_positive, n=10))

    # Treat ratings >= 4 as positive feedback
    positive = train[train["rating"] >= 4]

    movie_ids = movies["movie_id"].unique()
    movie_idx = {m: i for i, m in enumerate(movie_ids)}

    cos_similarity, X = pp.get_cosine_similarity(train, movies, users)

    k = 20
    neighbors = np.argsort(cos_similarity, axis=1)[:, -k:]

    # Test for User 1
    print("Item-based kNN:", base.recommend_knn(1, positive, movie_ids, movie_idx, neighbors, cos_similarity, train, movies, n=10))
    print()

    ####################################################################################################################

    # Evaluate baseline models
    results = {
        "Popular (All)": eval.evaluate(
            lambda u: base.recommend_popular(u, train, popular_all), val, len(movies)
        ),
        "Popular (Positive)": eval.evaluate(
            lambda u: base.recommend_popular(u, train, popular_positive), val, len(movies)
        ),
        "Item-kNN (k=20)": eval.evaluate(
            lambda u: base.recommend_knn(u, positive, movie_ids, movie_idx,neighbors, cos_similarity, train, movies), val, len(movies)
        )
    }

    results_df = pd.DataFrame(
        results,
        index=["Precision@10", "Recall@10", "NDCG@10", "Coverage"]
    ).T

    print(results_df.round(4))
    print()

    ####################################################################################################################

    jaccard = pp.get_jaccard_similarity(X)

    # Test different k values on validation data
    methods = {"Cosine": cos_similarity, "Jaccard": jaccard}
    tuning = []

    for method, sim in methods.items():
        similarity = sim

        for k in [10, 30]:
            neighbors = np.argsort(similarity, axis=1)[:, -k:]
            metrics = eval.evaluate(
                lambda u: base.recommend_knn(u, positive, movie_ids, movie_idx,neighbors, similarity, train, movies), val, len(movies)
            )
            tuning.append([method, k, *metrics])

    tuning_df = pd.DataFrame(
        tuning,
        columns=["Similarity", "k", "Precision", "Recall", "NDCG", "Coverage"]
    )

    print(tuning_df.round(4))
    print()

    # Select the best using NDCG
    best = tuning_df.loc[tuning_df["NDCG"].idxmax()]
    similarity = methods[best["Similarity"]]
    k = int(best["k"])
    neighbors = np.argsort(similarity, axis=1)[:, -k:]

    print("\nBest model:", best["Similarity"], "k =", k)
    print()

    ####################################################################################################################

    # Evaluate on test data
    test_results = {
        "Popular (All)": eval.evaluate(
            lambda u: base.recommend_popular(u, train, popular_all), test, len(movies)
        ),
        "Popular (Positive)": eval.evaluate(
            lambda u: base.recommend_popular(u, train, popular_positive), test, len(movies)
        ),
        "Item-kNN (Cosine, k=10)": eval.evaluate(
            lambda u: base.recommend_knn(u, positive, movie_ids, movie_idx,neighbors, similarity, train, movies), test, len(movies)
        )
    }

    test_df = pd.DataFrame(
        test_results,
        index=["Precision@10", "Recall@10", "NDCG@10", "Coverage"]
    ).T

    print(test_df.round(4))
    print()

    ####################################################################################################################

    # Create results folder
    Path("results").mkdir(exist_ok=True)

    # Save final test results
    test_df.to_csv("results/test_results.csv")

    # Save kNN tuning results
    tuning_df.to_csv("results/knn_tuning.csv", index=False)

    print("Results saved successfully!")


if __name__ == "__main__":
    main()
