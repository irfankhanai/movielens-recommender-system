"""Part 3, Track A: MMR re-ranking with genre vectors and beyond-accuracy metrics.

Everything here is plain NumPy so each step can be read and checked by hand.

Item representation: a binary genre vector per movie (18 MovieLens genres).
Item distance: Jaccard distance between two movies' genre sets,
    d(i, j) = 1 - |G_i & G_j| / |G_i | G_j|
i.e. the share of their combined genres that the two movies do NOT share.
The same distance is used inside MMR and for intra-list diversity (ILD).
"""

import numpy as np


# --------------------------------------------------------------------------- #
# Item representation and distance                                            #
# --------------------------------------------------------------------------- #

def genre_matrix(movies):
    """Binary movie x genre matrix in the row order of movies.dat."""
    genres = movies.genres.str.get_dummies(sep='|')
    return genres.to_numpy(dtype=np.float32), list(genres.columns)


def jaccard_distance(genres):
    """All-pairs Jaccard distance between genre sets (items x items)."""
    shared = genres @ genres.T                       # |G_i & G_j|
    sizes = genres.sum(axis=1)
    union = sizes[:, None] + sizes[None, :] - shared  # |G_i | G_j|
    similarity = np.divide(shared, union, out=np.zeros_like(shared), where=union > 0)
    return 1.0 - similarity


# --------------------------------------------------------------------------- #
# The intervention: Maximal Marginal Relevance (Lecture 5.1, slide 9)         #
# --------------------------------------------------------------------------- #

def mmr_rerank(candidates, relevance, distance, lam, n=10):
    """Greedy MMR.

    At every step pick the candidate i that maximises
        lam * relevance(i) - (1 - lam) * max_{j in selected} similarity(i, j)
    with similarity = 1 - distance. lam = 1 returns the base ranking unchanged;
    lam = 0 ignores relevance after the first pick.

    :param candidates: item indices, already in base-model rank order
    :param relevance: base scores of the candidates, scaled to [0, 1]
    :param distance: full item x item distance matrix
    :return: list of n item indices
    """
    candidates = np.asarray(candidates)
    relevance = np.asarray(relevance, dtype=float)
    remaining = np.ones(len(candidates), dtype=bool)
    # Highest similarity of each candidate to anything already selected.
    max_similarity = np.zeros(len(candidates))
    selected = []
    for _ in range(min(n, len(candidates))):
        mmr = lam * relevance - (1 - lam) * max_similarity
        mmr[~remaining] = -np.inf
        best = int(np.argmax(mmr))  # first maximum = best base rank on ties
        selected.append(candidates[best])
        remaining[best] = False
        max_similarity = np.maximum(max_similarity, 1 - distance[candidates, candidates[best]])
    return selected


# --------------------------------------------------------------------------- #
# Beyond-accuracy metrics (Lecture 5.1, slide 7)                              #
# --------------------------------------------------------------------------- #

def intra_list_diversity(items, distance):
    """Mean pairwise distance between the items of one list."""
    items = np.asarray(items)
    if len(items) < 2:
        return np.nan
    block = distance[np.ix_(items, items)]
    return block[np.triu_indices(len(items), k=1)].mean()


def gini(counts):
    """Gini coefficient of how often each catalogue item is recommended.

    0 = every item recommended equally often, 1 = all exposure on one item.
    Computed over the full catalogue, so never-recommended items count as 0.
    """
    x = np.sort(np.asarray(counts, dtype=float))
    n = len(x)
    if x.sum() == 0:
        return np.nan
    ranks = np.arange(1, n + 1)
    return ((2 * ranks - n - 1) * x).sum() / (n * x.sum())


def genre_distribution(items, genres):
    """p(g): each movie spreads one unit of weight equally over its genres."""
    weights = genres[np.asarray(items)]
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1)
    total = weights.sum(axis=0)
    return total / total.sum() if total.sum() > 0 else total


def calibration_error(history_dist, list_dist, alpha=0.01):
    """Steck (2018) miscalibration: KL(p || q~), q~ = (1 - alpha) q + alpha p.

    p is the genre distribution of the user's liked training movies and q the
    distribution of the recommended list. 0 means the list mirrors the user's
    genre mix exactly; alpha avoids division by zero for missing genres.
    """
    p = history_dist
    q = (1 - alpha) * list_dist + alpha * p
    mask = p > 0
    return float((p[mask] * np.log(p[mask] / q[mask])).sum())


def head_items(positive_counts, share=0.5):
    """Boolean mask of the 'popular head': the fewest most-liked movies that
    together hold `share` of all positive training interactions."""
    order = np.argsort(-positive_counts, kind='stable')
    cumulative = np.cumsum(positive_counts[order]) / positive_counts.sum()
    n_head = int(np.searchsorted(cumulative, share) + 1)
    mask = np.zeros(len(positive_counts), dtype=bool)
    mask[order[:n_head]] = True
    return mask
