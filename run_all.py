"""Run the MovieLens baseline experiments.

Final target:
    python run_all.py --data-dir ./data/ml-1m
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import logging

from src import preprocessing as pp
from src import analysis as analysis
from src import baselines as base
from src import evaluation as eval


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    parser = argparse.ArgumentParser(description="Reproduce MovieLens recommender experiments")
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to extracted ml-1m data")
    parser.add_argument("--output-dir", type=Path, default=Path("results"),
                        help="Directory for CSV tables and plots")
    parser.add_argument("--show-plots", action="store_true",
                        help="Also display plots interactively")
    args = parser.parse_args()
    if not args.data_dir.is_dir():
        parser.error(f"Dataset directory does not exist: {args.data_dir}")

    for filename in ("movies.dat", "ratings.dat", "users.dat"):
        if not (args.data_dir / filename).is_file():
            parser.error(f"Missing dataset file: {args.data_dir / filename}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.show_plots:
        analysis.plt.switch_backend("Agg")

    ##########################################   PREPROCESSING AND ANALYSIS   ##########################################


    # Load the datasets, and print their sizes
    movies, ratings, users = pp.get_data(args.data_dir)
    logging.info(f"Movies: {movies.shape}")
    logging.info(f"Ratings: {ratings.shape}")
    logging.info(f"Users: {users.shape}")

    logging.info(f"Sample Ratings:\n{ratings.head()}")


    # Show the relevant statistics regarding the datasets
    analysis.show_statistics(movies, ratings, users, args.output_dir / "plots", args.show_plots)


    # Split the ratings dataset into training, validation
    # and testing datasets using 80/10/10 split
    train, val, test = pp.split_data(ratings)

    ###############################################   BASELINE MODELS  #################################################

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

    # Treat ratings >= 4 as positive feedback
    positive = train[train["rating"] >= 4]
    eval_users = set(positive["user_id"])

    movie_ids = movies["movie_id"].unique()
    movie_idx = {m: i for i, m in enumerate(movie_ids)}

    cos_similarity, X = pp.get_cosine_similarity(train, movies, users)
    analysis.show_binary_feedback_statistics(ratings, movies, users, 4)

    k = 20
    neighbors = np.argsort(cos_similarity, axis=1)[:, -k:]

    # Example recommendations for User 1
    logging.info(f"--------- Logging the example recommendations based on the training set for the User 1: ---------")
    logging.info(f"All Ratings: {base.recommend_popular(1, train, popular_all, n=10)}")
    logging.info(f"Positive Ratings: {base.recommend_popular(1, train, popular_positive, n=10)}")
    logging.info(f"Item-based kNN: {base.recommend_knn(1, positive, movie_ids, movie_idx, neighbors, cos_similarity, train, movies, n=10)}")

    ##################################################   EVALUATION   ##################################################

    # Evaluate baseline models
    results = {
        "Popular (All)": eval.evaluate(
            lambda u: base.recommend_popular(u, train, popular_all), val, len(movies), eval_users
        ),
        "Popular (Positive)": eval.evaluate(
            lambda u: base.recommend_popular(u, train, popular_positive), val, len(movies), eval_users
        ),
        f"Item-kNN (k={k})": eval.evaluate(
            lambda u: base.recommend_knn(u, positive, movie_ids, movie_idx,neighbors, cos_similarity, train, movies), val, len(movies), eval_users
        )
    }

    results_df = pd.DataFrame(
        results,
        index=["Precision@10", "Recall@10", "NDCG@10", "Coverage"]
    ).T

    logging.info(f"--------- Logging the evaluation statistics on the validation set: ---------\n{results_df.round(4)}")

    ############################################   SIMILARITY COMPARISON   #############################################

    jaccard = pp.get_jaccard_similarity(X)

    # Test different k values on validation data
    methods = {"Cosine": cos_similarity, "Jaccard": jaccard}
    tuning = []

    for method, sim in methods.items():
        similarity = sim

        for k in [5, 10, 20, 30, 40, 50]:
            neighbors = np.argsort(similarity, axis=1)[:, -k:]
            metrics = eval.evaluate(
                lambda u: base.recommend_knn(u, positive, movie_ids, movie_idx,neighbors, similarity, train, movies), val, len(movies), eval_users
            )
            tuning.append([method, k, *metrics])

    tuning_df = pd.DataFrame(
        tuning,
        columns=["Similarity", "k", "Precision", "Recall", "NDCG", "Coverage"]
    )

    logging.info(f"--------- Logging the comparison between cosine and jaccard similarity measures: ---------\n{tuning_df.round(4)}")

    # Select the best using NDCG
    best = tuning_df.loc[tuning_df["NDCG"].idxmax()]
    similarity = methods[best["Similarity"]]
    k = int(best["k"])
    neighbors = np.argsort(similarity, axis=1)[:, -k:]

    logging.info(f"Best model: {best["Similarity"]}, k = {k}")


    ######################################   EVALUATION ON TEST DATA   #################################################

    # Evaluate on test data
    test_frames = []
    for label, users in [("positive_history", eval_users), ("all_users", None)]:
        test_results = {
            "Popular (All)": eval.evaluate(
                lambda u: base.recommend_popular(u, train, popular_all), test, len(movies), users
            ),
            "Popular (Positive)": eval.evaluate(
                lambda u: base.recommend_popular(u, train, popular_positive), test, len(movies), users
            ),
            f"Item-kNN ({best["Similarity"]}, k = {k})": eval.evaluate(
                lambda u: base.recommend_knn(u, positive, movie_ids, movie_idx, neighbors, similarity, train, movies),
                test, len(movies), users
            )
        }

        test_df = pd.DataFrame(
            test_results,
            index=["Precision@10", "Recall@10", "NDCG@10", "Coverage"]
        ).T
        test_df.insert(0, "Users", label)
        test_frames.append(test_df)

        logging.info(
            f"--------- Logging the evaluation metrics on the test dataset ({label}): ---------\n{test_df.round(4)}")

    test_df = pd.concat(test_frames)
    test_df.index.name = "Model"

    ########################################   SAVING THE RESULTS   ####################################################

    # Create results folder
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save final test results
    test_df.to_csv(args.output_dir / "test_results.csv", index_label="Model")
    results_df.to_csv(args.output_dir / "validation_results.csv", index_label="Model")

    # Save kNN tuning results
    tuning_df.to_csv(args.output_dir / "knn_tuning.csv", index=False)

    logging.info(f"Results saved successfully!")


if __name__ == "__main__":
    main()
