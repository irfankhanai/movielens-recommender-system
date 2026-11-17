"""Additional recommendation algorithms (Part 2)."""
import torch.nn as nn
import torch.nn.functional as F
import torch

from torch.utils.data import DataLoader, TensorDataset

#####
# NEURAL paradigm - Multi-VAE model
#####

# TEMPORARY data processing first:

from src.preprocessing import get_ratings, split_data
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np


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
    def __init__(self, n_items=3706, hidden_dim = 500, latent_dim=100, dropout=0.5):
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
        h = self.encoder(X)

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