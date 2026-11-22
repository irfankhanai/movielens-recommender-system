"""Small, inspectable Part 2 models. No validation/test feedback enters fitting."""
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

import torch.nn as nn
import torch.nn.functional as F
import torch
import logging

from torch.utils.data import DataLoader, TensorDataset

#####
# NEURAL paradigm - Multi-VAE model
#####

# TEMPORARY data processing first:

from src.preprocessing import split_data
import copy
from src import evaluation as evaluation
from scipy.sparse import csr_matrix


############################################################
# Multi-VAE model by Reinis Ignatjevs                      #
############################################################


def get_interaction_matrix(ratings, movies, users, threshold=4):
    """
    Build a binary user-movie interaction matrix.
    Rows are user_id - 1 (0..6039), columns follow the order of movies.dat (0..3882).
    :param ratings: Ratings to include (e.g. the training set)
    :param movies: Movies dataset
    :param users: Users dataset
    :param threshold: Minimum rating counted as 1. None counts every rating (used as a "seen" mask)
    :return: The interaction matrix X, the movie ids and the movie_id -> column index mapping
    """
    if threshold is not None:
        ratings = ratings[ratings["rating"] >= threshold]

    movie_ids = movies["movie_id"].unique()
    movie_idx = {m: i for i, m in enumerate(movie_ids)}

    X = csr_matrix(
        (np.ones(len(ratings), dtype=np.float32),
         (ratings["user_id"] - 1, ratings["movie_id"].map(movie_idx))),
        shape=(users["user_id"].max(), len(movie_ids))
    )

    return X, movie_ids, movie_idx


def get_multi_vae_data(ratings, movies, users, threshold=4):
    """
    Split the ratings with the same global temporal 80/10/10 split as the baselines,
    and build the Multi-VAE inputs from the training period.
    :param ratings: Full ratings dataset
    :param movies: Movies dataset
    :param users: Users dataset
    :param threshold: Minimum rating counted as positive
    :return: Dictionary with
             X_train: binary positives matrix from train (model input / training target),
             seen_train: every movie rated in train (masked out when recommending),
             train_rows: row indices of users with at least one positive in train,
             movie_ids: column index -> movie_id,
             train, val, test: the split DataFrames (for evaluation.evaluate)
    """
    train, val, test = split_data(ratings)

    X_train, movie_ids, _ = get_interaction_matrix(train, movies, users, threshold)
    seen_train, _, _ = get_interaction_matrix(train, movies, users, threshold=None)
    train_rows = np.flatnonzero(X_train.getnnz(axis=1) > 0)

    return {
        "X_train": X_train,
        "seen_train": seen_train,
        "train_rows": train_rows,
        "movie_ids": movie_ids,
        "train": train,
        "val": val,
        "test": test,
    }


def get_train_loader(X, rows, batch_size=128, seed=42):
    """
    DataLoader over the training users, reshuffled every epoch.
    :param X: Sparse user-movie matrix
    :param rows: Row indices of users to train on
    """
    X_dense = torch.from_numpy(X[rows].toarray())
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(X_dense), batch_size=batch_size,
                      shuffle=True, generator=generator)

def vae_loss(logits, x, mean, log_var, beta):
    """
    Multi-VAE loss: multinomial reconstruction + beta * KL divergence.
    :param logits: Decoder output, shape (batch, n_items)
    :param x: Original binary input batch, shape (batch, n_items)
    :param mean: Latent means, shape (batch, latent_dim)
    :param log_var: Latent log variances, shape (batch, latent_dim)
    :param beta: Weight of the KL term (annealed during training)
    :return: Total loss, reconstruction loss, KL loss
    """
    predicted_probabilities = F.log_softmax(logits, dim=1) # for each user's movies
    probabilities_for_liked_movies = predicted_probabilities * x # X is 1 where the user liked the movie, otherwise 0

    # Get the mean of how well we managed to predict the liked movies and then turn it negative for the reconstruction loss to minimize
    reconstruction_loss = -torch.sum(probabilities_for_liked_movies, dim=1).mean()

    # Now get KL loss:
    kl = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp()).sum(dim=1).mean()

    return reconstruction_loss + beta * kl, reconstruction_loss, kl

class MultiVAE(nn.Module):
    def __init__(self, n_items, hidden_dim = 500, latent_dim=100, dropout=0.5):
        # TODO: normalisation
        super().__init__()

        # Defining the model:
        self.encoder = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(n_items, hidden_dim),  # hidden layer
            nn.Tanh(),
        )

        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_items),
        )

    def forward(self, X):
        norm_X = F.normalize(X) # Each user row has length 1
        h = self.encoder(norm_X)

        mean = self.mean(h)
        log_var = self.log_var(h)

        # Sample from the latent in training:
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mean + std * eps
        else:
            z = mean

        output = self.decoder(z)
        return output, mean, log_var


def train_one_epoch(model, train_loader, optimizer, step, anneal_cap=0.2, anneal_steps=20000):
    """
    Train the model for one pass over all training users.
    :param model: The MultiVAE model
    :param train_loader: DataLoader from get_train_loader
    :param optimizer: e.g. torch.optim.Adam(model.parameters(), lr=1e-3)
    :param step: Number of batches trained so far (carried across epochs, used for beta)
    :param anneal_cap: Maximum value of beta
    :param anneal_steps: Number of batches over which beta grows from 0 to anneal_cap
    :return: Average loss, reconstruction and KL over the epoch, the updated step and the current beta
    """
    model.train()  # dropout and sampling on

    total_loss, total_recon, total_kl, n_users = 0.0, 0.0, 0.0, 0
    beta = 0.0

    for (x,) in train_loader:
        # TODO: decide the KL annealing!
        # KL annealing: beta rises linearly with the number of batches seen, then stays at the cap
        beta = min(anneal_cap, anneal_cap * step / anneal_steps)

        logits, mean, log_var = model(x)
        loss, recon, kl = vae_loss(logits, x, mean, log_var, beta)

        optimizer.zero_grad()  # clear gradients from the previous batch
        loss.backward()        # compute gradients
        optimizer.step()       # update the weights

        step += 1
        batch_users = x.shape[0]
        total_loss += loss.item() * batch_users
        total_recon += recon.item() * batch_users
        total_kl += kl.item() * batch_users
        n_users += batch_users

    return total_loss / n_users, total_recon / n_users, total_kl / n_users, step, beta


@torch.no_grad()
def score_all_users(model, X, seen, batch_size=1000):
    """
    Compute a score for every (user, movie) pair, with already-rated movies masked out.
    :param model: Trained MultiVAE
    :param X: Sparse binary positives matrix used as input (X_train)
    :param seen: Sparse matrix of every rated movie in the history (seen_train)
    :param batch_size: Number of users scored at once
    :return: numpy array of shape (n_users, n_items); row i is user_id i + 1
    """
    model.eval()  # dropout off, z = mean

    X_dense = torch.from_numpy(X.toarray())
    scores = []
    for start in range(0, X_dense.shape[0], batch_size):
        logits, _, _ = model(X_dense[start:start + batch_size])
        scores.append(logits.numpy())

    scores = np.vstack(scores)
    scores[seen.toarray() > 0] = -np.inf  # never recommend a movie the user already rated
    return scores


def make_recommend_fn(scores, movie_ids, n=10):
    """
    :return: Function user_id -> list of the top n movie ids
    """
    def recommend(user_id):
        top_columns = np.argsort(-scores[user_id - 1])[:n]
        return movie_ids[top_columns].tolist()
    return recommend


def train_multi_vae(data, n_movies, hidden_dim=500, latent_dim=100, dropout=0.5,
                    lr=1e-3, batch_size=128, anneal_cap=0.2, anneal_steps=20000,
                    max_epochs=200, patience=20, seed=42):
    """
    Train a MultiVAE with early stopping on validation NDCG@10.
    :param data: Dictionary from get_multi_vae_data
    :param n_movies: Total number of movies (for catalogue coverage)
    :return: The model with the best validation weights, and a per-epoch history
    """
    torch.manual_seed(seed)
    model = MultiVAE(n_items=data["X_train"].shape[1], hidden_dim=hidden_dim,
                     latent_dim=latent_dim, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = get_train_loader(data["X_train"], data["train_rows"], batch_size, seed)
    eval_users = set(data["train_rows"] + 1)

    step, best_ndcg, best_state, epochs_without_improvement = 0, -1.0, None, 0
    history = []

    for epoch in range(1, max_epochs + 1):
        loss, recon, kl, step, beta = train_one_epoch(
            model, train_loader, optimizer, step, anneal_cap, anneal_steps)

        scores = score_all_users(model, data["X_train"], data["seen_train"])
        precision, recall, ndcg, coverage = evaluation.evaluate(
            make_recommend_fn(scores, data["movie_ids"]), data["val"], n_movies, eval_users)

        history.append([epoch, loss, recon, kl, beta, precision, recall, ndcg, coverage])
        logging.info(f"epoch {epoch:3d}  loss {loss:.2f}  beta {beta:.3f}  "
                     f"val NDCG@10 {ndcg:.4f}  R@10 {recall:.4f}")

        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logging.info(f"Early stopping at epoch {epoch}, best val NDCG@10 {best_ndcg:.4f}")
                break

    if best_state is not None:  # e.g. no validation users: keep the last weights
        model.load_state_dict(best_state)
    history_df = pd.DataFrame(history, columns=["epoch", "loss", "recon", "kl", "beta",
                                                "Precision", "Recall", "NDCG", "Coverage"])
    return model, history_df


class MultiVAERecommender:
    """Multi-VAE behind the shared Part 2 interface: ModelData in, recommend(user) out.

    Trained on data.x (training positives only); the validation set is used only for
    early stopping. Scores are softmax probabilities (always > 0), so ModelData.rank
    applies the same seen-mask, tie policy and cold-user fallback as the other models.
    """

    def __init__(self, data, val, **settings):
        self.data = data
        x = data.x.astype(np.float32).tocsr()

        # Seen matrix from ModelData.seen, which stays the full training set during the ablation
        rows, cols = [], []
        for user, items in data.seen.items():
            row = data.user_index.get(user)
            if row is not None:
                rows.extend([row] * len(items))
                cols.extend(items)
        seen = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=x.shape)

        vae_data = {"X_train": x, "seen_train": seen,
                    "train_rows": np.flatnonzero(x.getnnz(axis=1) > 0),
                    "movie_ids": data.items, "val": val}
        self.model, self.history = train_multi_vae(vae_data, len(data.items), **settings)
        self.scores = self._score_all(x)

    @torch.no_grad()
    def _score_all(self, x, batch_size=1000):
        self.model.eval()
        x = torch.from_numpy(x.toarray())
        return np.vstack([F.softmax(self.model(x[start:start + batch_size])[0], dim=1).numpy()
                          for start in range(0, x.shape[0], batch_size)])

    def recommend(self, user):
        row = self.data.user_index.get(user)
        if row is None:  # user unknown to the model
            return self.data.rank(user, self.data.popular)
        return self.data.rank(user, self.scores[row])


############################################################
# Item-kNN and Hybrid model by Irfan Khan                  #
############################################################


def unit_rows(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)


class ModelData:
    """Training-only feedback and shared full-catalogue ranking policy."""

    def __init__(self, train, movies, users, seen_train=None):
        self.movies = movies.reset_index(drop=True)
        self.items = movies.movie_id.to_numpy()
        self.item_index = {m: i for i, m in enumerate(self.items)}
        self.users = users.user_id.to_numpy()
        self.user_index = {u: i for i, u in enumerate(self.users)}
        positive = train[train.rating >= 4]
        self.x = csr_matrix((np.ones(len(positive)),
                             (positive.user_id.map(self.user_index),
                              positive.movie_id.map(self.item_index))),
                            shape=(len(users), len(movies)))
        self.popular = np.asarray(self.x.sum(axis=0)).ravel()
        self.all_counts = train.movie_id.value_counts().reindex(self.items, fill_value=0).to_numpy()
        # During ablation, keep the original exclusion set fixed: removed
        # training events are hidden evidence, not newly recommendable movies.
        seen_train = train if seen_train is None else seen_train
        self.seen = {u: np.array([self.item_index[m] for m in rows.movie_id])
                     for u, rows in seen_train.groupby('user_id')}

    def rank(self, user, scores, n=10, legacy_ties=False):
        scores = np.asarray(scores, dtype=float).copy()
        scores[self.seen.get(user, np.array([], dtype=int))] = -np.inf
        if np.max(scores) <= 0:
            scores = self.popular.astype(float).copy()
            scores[self.seen.get(user, np.array([], dtype=int))] = -np.inf
        # Preserve Part 1's item-kNN tie policy; new models use movie ID.
        order = np.argsort(-scores) if legacy_ties else np.lexsort((self.items, -scores))
        return self.items[order[np.isfinite(scores[order])][:n]].tolist()


class UserKNN:
    """Cosine neighbours on binary likes, then similarity-weighted votes."""

    def __init__(self, data, k=40):
        self.data, self.k = data, k
        self.normalized = data.x.multiply(
            1 / np.sqrt(np.maximum(np.asarray(data.x.sum(axis=1)).ravel(), 1))[:, None]
        ).tocsr()

    def scores(self, user):
        row = self.data.user_index.get(user)
        if row is None:
            return np.zeros(len(self.data.items))
        similarity = (self.normalized @ self.normalized[row].T).toarray().ravel()
        similarity[row] = 0  # A user must never be their own neighbour.
        order = np.lexsort((self.data.users, -similarity))[:self.k]
        order = order[similarity[order] > 0]
        return np.asarray(self.data.x[order].T @ similarity[order]).ravel()

    def recommend(self, user):
        return self.data.rank(user, self.scores(user))


class ContentProfile:
    """Self-implementation: feature construction, profiles and cosine scoring.

    Lecture 2.2, slides 4-7. Genre multi-hot vectors are length-normalized;
    release year is represented by a one-hot decade for a readable control.
    z_i = [sqrt(1-w)*genre_i, sqrt(w)*decade_i].
    p_u = unit(sum(z_i for training likes)); score(u,i) = p_u dot z_i.
    NumPy performs arithmetic only; no recommender library is used.
    """

    def __init__(self, data, year_weight=0.25):
        self.data = data
        genres = data.movies.genres.str.get_dummies(sep='|').to_numpy(dtype=float)
        years = data.movies.title.str.extract(r'\((\d{4})\)$', expand=False)
        decades = years.fillna('0').astype(int) // 10 * 10
        decade_vectors = (decades.to_numpy()[:, None] == np.sort(decades.unique())).astype(float)
        self.features = np.column_stack((np.sqrt(1-year_weight) * unit_rows(genres),
                                         np.sqrt(year_weight) * decade_vectors))
        self.profiles = unit_rows(np.asarray(data.x @ self.features))

    def scores(self, user):
        row = self.data.user_index.get(user)
        return (self.features @ self.profiles[row] if row is not None
                else np.zeros(len(self.data.items)))

    def recommend(self, user):
        return self.data.rank(user, self.scores(user))


class Hybrid:
    """Convex blend of fixed user-kNN and content components.

    Normalize by each component's maximum over eligible candidates before
    blending, so alpha is meaningful. Zero components stay zero.
    """

    def __init__(self, data, alpha=0.5):
        self.data, self.alpha = data, alpha
        self.cf = UserKNN(data, k=40)
        self.content = ContentProfile(data, year_weight=0.25)

    def recommend(self, user):
        components = []
        for model in (self.cf, self.content):
            scores = model.scores(user)
            scores[self.data.seen.get(user, np.array([], dtype=int))] = 0
            maximum = scores.max()
            components.append(scores / maximum if maximum > 0 else scores)
        return self.data.rank(user, self.alpha * components[0] + (1-self.alpha) * components[1])


class ItemKNN:
    """Part 1's outgoing-neighbour scoring, cached for efficient evaluation."""

    def __init__(self, data, k=40, similarity='Cosine'):
        self.data = data
        if similarity == 'Cosine':
            sim = cosine_similarity(data.x.T)
        else:
            co = (data.x.T @ data.x).toarray()
            union = data.popular[:, None] + data.popular[None, :] - co
            sim = np.divide(co, union, out=np.zeros_like(co), where=union > 0)
        np.fill_diagonal(sim, 0)
        neighbors = np.argsort(sim, axis=1)[:, -k:]
        rows = np.repeat(np.arange(len(data.items)), k)
        self.weights = csr_matrix((sim[rows, neighbors.ravel()], (rows, neighbors.ravel())),
                                  shape=sim.shape)

    def recommend(self, user):
        row = self.data.user_index.get(user)
        scores = ((self.data.x[row] @ self.weights).toarray().ravel() if row is not None
                  else np.zeros(len(self.data.items)))
        return self.data.rank(user, scores, legacy_ties=True)


############################################################
# Self-implemented Content-based model by Kaan Pide        #
############################################################


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
