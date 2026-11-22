"""Run the MovieLens experiments: baselines (Part 1) and the Multi-VAE (Part 2).

Usage:
    python run_all.py                      # uses config.DATA_DIR
    python run_all.py --data-dir ./data
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import logging

from src import config as cfg
from src import preprocessing as pp
from src import analysis as analysis
from src import baselines as base
from src import evaluation as eval
from src import models as md


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    parser = argparse.ArgumentParser(description="Reproduce MovieLens recommender experiments")
    parser.add_argument("--data-dir", type=Path, default=cfg.DATA_DIR, help="Path to extracted ml-1m data")
    parser.add_argument("--output-dir", type=Path, default=Path("results"),
                        help="Directory for CSV tables and plots")
    parser.add_argument("--show-plots", action="store_true",
                        help="Also display plots interactively")
    parser.add_argument("--part", choices=["1", "2", "all"], default="all",
                        help="Run Part 1, Part 2, or both (default: all)")
    args = parser.parse_args()
    if not args.data_dir.is_dir():
        parser.error(f"Dataset directory does not exist: {args.data_dir}")

    for filename in ("movies.dat", "ratings.dat", "users.dat"):
        if not (args.data_dir / filename).is_file():
            parser.error(f"Missing dataset file: {args.data_dir / filename}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.part == "2":
        from run_part2 import main as part2_main
        part2_main(args.data_dir, args.output_dir / "part2")
        return
    if not args.show_plots:
        analysis.plt.switch_backend("Agg")

    threshold = cfg.POSITIVE_THRESHOLD
    n = cfg.TOP_N
    metric_names = [f"Precision@{n}", f"Recall@{n}", f"NDCG@{n}", "Coverage"]

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
    positive_counts = train[train["rating"] >= threshold]["movie_id"].value_counts()

    all_movies = sorted(movies["movie_id"].unique())

    # Rank the full movie catalogue
    popular_all = sorted(
        all_movies, key=lambda m: (-all_counts.get(m, 0), m)
    )
    popular_positive = sorted(
        all_movies, key=lambda m: (-positive_counts.get(m, 0), m)
    )

    # Treat ratings >= threshold as positive feedback
    positive = train[train["rating"] >= threshold]
    eval_users = set(positive["user_id"])

    movie_ids = movies["movie_id"].unique()
    movie_idx = {m: i for i, m in enumerate(movie_ids)}

    cos_similarity, X = pp.get_cosine_similarity(train, movies, users)
    analysis.show_binary_feedback_statistics(ratings, movies, users, threshold)
    analysis.show_split_statistics(train, val, test, threshold)

    k = cfg.KNN_DEFAULT_K
    neighbors = np.argsort(cos_similarity, axis=1)[:, -k:]

    gen_sim, _ = md.create_genre_matrix(movies)

    # Example recommendations for User 1
    logging.info(f"--------- Logging the example recommendations based on the training set for the User 1: ---------")
    logging.info(f"All Ratings: {[movies[movies["movie_id"] == i]["title"].values[0] for i in base.recommend_popular(1, ratings, popular_all, n=n)]}")
    logging.info(f"Positive Ratings: {[movies[movies["movie_id"] == i]["title"].values[0] for i in base.recommend_popular(1, ratings, popular_positive, n=n)]}")
    logging.info(f"Item-based kNN: {[movies[movies["movie_id"] == i]["title"].values[0] for i in base.recommend_knn(1, positive, movie_ids, movie_idx, neighbors, cos_similarity, ratings, movies, n=n)]}")
    logging.info(f"Content based, unique genres: {[movies[movies["movie_id"] == i]["title"].values[0] for i in md.content_based_recommendation(1, movies, ratings, gen_sim, diversify=True, n=n)]}")

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
        index=metric_names
    ).T

    logging.info(f"--------- Logging the evaluation statistics on the validation set: ---------\n{results_df.round(4)}")

    ############################################   SIMILARITY COMPARISON   #############################################

    jaccard = pp.get_jaccard_similarity(X)

    # Test different k values on validation data
    methods = {"Cosine": cos_similarity, "Jaccard": jaccard}
    tuning = []

    for method, sim in methods.items():
        similarity = sim

        for k in cfg.KNN_K_VALUES:
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

    ###############################################   MULTI-VAE MODEL   ################################################

    # Same temporal split as above; inputs are built from the training period only
    vae_data = md.get_multi_vae_data(ratings, movies, users, threshold)
    assert len(vae_data["train"]) == len(train), "Multi-VAE split differs from the baseline split"

    vae_model, vae_history = md.train_multi_vae(
        vae_data, len(movies),
        hidden_dim=cfg.VAE_HIDDEN_DIM,
        latent_dim=cfg.VAE_LATENT_DIM,
        dropout=cfg.VAE_DROPOUT,
        lr=cfg.VAE_LEARNING_RATE,
        batch_size=cfg.VAE_BATCH_SIZE,
        anneal_cap=cfg.VAE_ANNEAL_CAP,
        anneal_steps=cfg.VAE_ANNEAL_STEPS,
        max_epochs=cfg.VAE_MAX_EPOCHS,
        patience=cfg.VAE_PATIENCE,
        seed=cfg.SEED,
    )

    # Score every user once with the best weights; reused for validation and test
    vae_scores = md.score_all_users(vae_model, vae_data["X_train"], vae_data["seen_train"])
    vae_recommend = md.make_recommend_fn(vae_scores, vae_data["movie_ids"], n=n)

    results_df.loc["Multi-VAE"] = eval.evaluate(vae_recommend, val, len(movies), eval_users)
    logging.info(f"--------- Logging the Multi-VAE evaluation statistics on the validation set: ---------\n"
                 f"{results_df.loc[['Multi-VAE']].round(4)}")

    ######################################   EVALUATION ON TEST DATA   #################################################

    # Evaluate on test data
    test_frames = []
    for label, user_set in [("positive_history", eval_users), ("all_users", None)]:
        test_results = {
            "Popular (All)": eval.evaluate(
                lambda u: base.recommend_popular(u, train, popular_all), test, len(movies),
                user_set, show_completion=True
            ),
            "Popular (Positive)": eval.evaluate(
                lambda u: base.recommend_popular(u, train, popular_positive), test, len(movies),
                user_set, show_completion=True
            ),
            f"Item-kNN ({best["Similarity"]}, k = {k})": eval.evaluate(
                lambda u: base.recommend_knn(u, positive, movie_ids, movie_idx, neighbors, similarity, train, movies),
                test, len(movies), user_set, show_completion=True
            ),
            "Multi-VAE": eval.evaluate(
                vae_recommend, test, len(movies), user_set, show_completion=True
            ),
            "Content based, unique genres": eval.evaluate(
                lambda u: md.content_based_recommendation(u, movies, train, gen_sim, diversify=True), test, len(movies),
                user_set, show_completion=True
            )
        }

        test_df = pd.DataFrame(
            test_results,
            index=metric_names
        ).T
        test_df.insert(0, "Users", label)
        test_frames.append(test_df)

        logging.info(
            f"--------- Logging the evaluation metrics on the test dataset ({label}): ---------\n{test_df.round(4).to_string()}")

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

    # Save the Multi-VAE training history (loss, beta and validation metrics per epoch)
    vae_history.to_csv(args.output_dir / "vae_history.csv", index=False)

    logging.info(f"Results saved successfully!")
    if args.part == "all":
        from run_part2 import main as part2_main
        part2_main(args.data_dir, args.output_dir / "part2")


if __name__ == "__main__":
    main()
