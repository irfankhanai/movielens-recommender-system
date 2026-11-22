"""Per-user metrics and paired bootstrap intervals for Part 2."""

import numpy as np
import pandas as pd

METRICS = ['Precision@10', 'Recall@10', 'NDCG@10']


def per_user(recommend, heldout, eligible=None):
    rows = []
    for user, group in heldout[heldout.rating >= 4].groupby('user_id'):
        if eligible is not None and user not in eligible:
            continue
        actual = set(group.movie_id)
        predicted = recommend(user)[:10]
        hits = np.array([m in actual for m in predicted], dtype=float)
        ideal = (1 / np.log2(np.arange(min(10, len(actual))) + 2)).sum()
        rows.append([user, hits.sum()/10, hits.sum()/len(actual),
                     (hits / np.log2(np.arange(len(hits)) + 2)).sum()/ideal, predicted])
    return pd.DataFrame(rows, columns=['user_id', *METRICS, 'recommendations'])


def aggregate(frame, n_items):
    result = frame[METRICS].mean().to_dict()
    result['Catalogue coverage'] = (len({m for recs in frame.recommendations for m in recs})
                                     / n_items if len(frame) else np.nan)
    return result


def mean_ci(values, seed=42, repeats=2000):
    """Percentile user bootstrap; paired differences must be aligned first."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(repeats)]
    return tuple(np.quantile(means, [0.025, 0.975]))


def paired_test(delta, seed=42, repeats=10000):
    """Two-sided paired sign-flip test of the mean difference across users.

    Under the null, the two model labels are exchangeable within each user.
    The +1 correction avoids zero Monte Carlo p-values.
    """
    delta = np.asarray(delta, dtype=float)
    rng = np.random.default_rng(seed)
    observed = abs(delta.mean())
    extreme = 0
    for start in range(0, repeats, 100):
        signs = rng.choice([-1, 1], size=(min(100, repeats-start), len(delta)))
        extreme += np.count_nonzero(np.abs((signs * delta).mean(axis=1)) >= observed)
    return (extreme + 1) / (repeats + 1)
