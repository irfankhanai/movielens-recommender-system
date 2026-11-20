"""Small, inspectable Part 2 models. No validation/test feedback enters fitting."""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


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
