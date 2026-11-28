"""Part 3, Track A: diversity and popularity bias via MMR re-ranking.

Base recommender: User-kNN (k=160), the Part 2 model with the highest test NDCG@10.
Intervention: MMR re-ranking of its top-100 candidates with genre vectors.
Sweep lambda from 1.0 (base, unchanged) to 0.0 (diversity only), on validation and test.
The operating point is picked on validation with a rule fixed before looking at test.

Usage:
    python run_all.py --data-dir ./data/ml-1m --part 3
"""

import argparse
import hashlib
import json
import logging
import platform
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.preprocessing import get_data, split_data
from src.models import ModelData, UserKNN
from src.part2_evaluation import per_user, aggregate, mean_ci, paired_test
from src import part3_diversity as div

SEED = 42
BASE_K = 160            # User-kNN k selected on validation in Part 2
POOL = 100              # MMR re-ranks the base model's top-100 unseen movies
LAMBDAS = [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
MAX_NDCG_LOSS = 0.05    # selection rule: lowest lambda losing <= 5% validation NDCG
CAL_ALPHA = 0.01        # smoothing in the calibration KL (Steck, 2018)
DISTANCE_NAME = 'Jaccard distance on genre sets (1 - shared genres / union of genres)'


class MMRUserKNN:
    """Builds the candidate pool once per user and re-ranks it for any lambda."""

    def __init__(self, data, knn, distance, pool=POOL):
        self.data, self.knn, self.distance, self.pool = data, knn, distance, pool
        self.cache = {}

    def candidates(self, user):
        if user not in self.cache:
            scores = self.knn.scores(user).astype(float)
            seen = self.data.seen.get(user, np.array([], dtype=int))
            scores[seen] = -np.inf
            if np.max(scores) <= 0:  # same popularity fallback as ModelData.rank
                scores = self.data.popular.astype(float).copy()
                scores[seen] = -np.inf
            order = np.lexsort((self.data.items, -scores))  # ties by movie id, as in Part 2
            order = order[np.isfinite(scores[order])][:self.pool]
            relevance = scores[order] / scores[order[0]]     # best candidate = 1
            self.cache[user] = (order, relevance)
        return self.cache[user]

    def indices(self, user, lam):
        order, relevance = self.candidates(user)
        return div.mmr_rerank(order, relevance, self.distance, lam)

    def recommend_fn(self, lam):
        return lambda user: self.data.items[self.indices(user, lam)].tolist()


def evaluate_lambda(model, lam, heldout, eligible, data, genres, distance, history, head):
    frame = per_user(model.recommend_fn(lam), heldout, eligible)
    ild, cal, tail = [], [], []
    counts = np.zeros(len(data.items))
    for user, recs in zip(frame.user_id, frame.recommendations):
        idx = np.array([data.item_index[m] for m in recs])
        counts[idx] += 1
        ild.append(div.intra_list_diversity(idx, distance))
        cal.append(div.calibration_error(history[user], div.genre_distribution(idx, genres), CAL_ALPHA))
        tail.append(1 - head[idx].mean())
    frame['ILD'], frame['Calibration error'], frame['Long-tail share'] = ild, cal, tail
    summary = aggregate(frame, len(data.items))
    summary.update({'Intra-list diversity': np.mean(ild), 'Gini': div.gini(counts),
                    'Calibration error': np.mean(cal), 'Long-tail share': np.mean(tail)})
    return frame, summary


def plot_frontier(sweep, selected, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for split, style in [('validation', 'o--'), ('test', 's-')]:
        rows = sweep[sweep.Split == split]
        axes[0].plot(rows['Intra-list diversity'], rows['NDCG@10'], style, label=split, ms=5)
    test = sweep[sweep.Split == 'test']
    for _, row in test.iterrows():
        if row['lambda'] in (1.0, 0.9, 0.7, 0.5, 0.3, 0.0):
            axes[0].annotate(f"λ={row['lambda']:g}", (row['Intra-list diversity'], row['NDCG@10']),
                             textcoords='offset points', xytext=(5, 4), fontsize=8)
    chosen = test[test['lambda'] == selected].iloc[0]
    axes[0].scatter(chosen['Intra-list diversity'], chosen['NDCG@10'], s=160, facecolors='none',
                    edgecolors='crimson', linewidths=2, label=f'chosen λ={selected:g}', zorder=3)
    axes[0].set_xlabel('Intra-list diversity (mean Jaccard genre distance)')
    axes[0].set_ylabel('NDCG@10')
    axes[0].set_title('(a) Accuracy–diversity frontier')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    for column, label in [('Intra-list diversity', 'Intra-list diversity'),
                          ('Calibration error', 'Calibration error (KL)'),
                          ('Catalogue coverage', 'Catalogue coverage'),
                          ('Long-tail share', 'Long-tail share of slots')]:
        axes[1].plot(test['lambda'], test[column], marker='.', label=label)
    axes[1].axvline(selected, color='crimson', ls=':', lw=1)
    axes[1].invert_xaxis()
    axes[1].set_xlabel('λ  (1 = base User-kNN, 0 = diversity only)')
    axes[1].set_ylabel('Value (test set)')
    axes[1].set_ylim(0, 1)
    axes[1].set_title('(b) Beyond-accuracy metrics along the sweep')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(data_dir=None, output_dir=None):
    if data_dir is None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--data-dir', type=Path, required=True)
        parser.add_argument('--output-dir', type=Path, default=Path('results/part3'))
        args = parser.parse_args()
        data_dir, output_dir = args.data_dir, args.output_dir
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    movies, ratings, users = get_data(data_dir)
    train, val, test = split_data(ratings)
    eligible = set(train.loc[train.rating >= 4, 'user_id'])
    data = ModelData(train, movies, users)
    knn = UserKNN(data, k=BASE_K)
    genres, genre_names = div.genre_matrix(data.movies)
    distance = div.jaccard_distance(genres)
    model = MMRUserKNN(data, knn, distance)
    head = div.head_items(data.popular)
    logging.info('Popular head: %d movies hold 50%% of positive training ratings', head.sum())

    # Each user's genre mix in their liked training movies (the calibration target).
    liked = train[train.rating >= 4].groupby('user_id').movie_id.apply(list)
    history = {u: div.genre_distribution([data.item_index[m] for m in ms], genres) for u, ms in liked.items()}

    # Sanity check: lambda = 1 must reproduce the Part 2 User-kNN lists exactly.
    for user in sorted(eligible)[:300]:
        assert model.recommend_fn(1.0)(user) == knn.recommend(user), user

    sweep, frames = [], {}
    for split, heldout in [('validation', val), ('test', test)]:
        for lam in LAMBDAS:
            logging.info('MMR %s lambda=%.2f', split, lam)
            frame, summary = evaluate_lambda(model, lam, heldout, eligible, data, genres, distance, history, head)
            frames[split, lam] = frame
            sweep.append({'Split': split, 'lambda': lam, 'Users': len(frame), **summary})
    sweep = pd.DataFrame(sweep)

    # Operating point: chosen on validation only, by a rule fixed in advance.
    validation = sweep[sweep.Split == 'validation'].set_index('lambda')
    floor = (1 - MAX_NDCG_LOSS) * validation.loc[1.0, 'NDCG@10']
    selected = float(min(l for l in LAMBDAS if validation.loc[l, 'NDCG@10'] >= floor))
    logging.info('Selected lambda on validation: %s', selected)

    # Paired comparison of the chosen point against the base on the same test users.
    base, chosen = frames['test', 1.0].set_index('user_id'), frames['test', selected].set_index('user_id')
    differences = []
    for metric in ['Precision@10', 'Recall@10', 'NDCG@10', 'ILD', 'Calibration error', 'Long-tail share']:
        delta = (chosen[metric] - base[metric]).to_numpy()
        lo, hi = mean_ci(delta, seed=SEED)
        differences.append({'Comparison': f'MMR lambda={selected:g} minus base (lambda=1)', 'Metric': metric,
                            'Base mean': base[metric].mean(), 'MMR mean': chosen[metric].mean(),
                            'Mean difference': delta.mean(), 'CI lower': lo, 'CI upper': hi,
                            'Users': len(delta), 'Paired sign-flip p': paired_test(delta, seed=SEED)})

    # Who bears the cost: break the change down by user activity and taste breadth,
    # at the chosen point and at a more aggressive setting (lambda = 0.5).
    activity = train.groupby('user_id').size()
    activity_cut = float(activity.median())  # 86, the same cut as Part 2
    entropy = pd.Series({u: -(p[p > 0] * np.log2(p[p > 0])).sum() for u, p in history.items()})
    breadth_cut = float(entropy.reindex(base.index).median())
    groups = [('User activity', 'low', f'<= {activity_cut:g} training ratings', base.index.map(activity) <= activity_cut),
              ('User activity', 'high', f'> {activity_cut:g} training ratings', base.index.map(activity) > activity_cut),
              ('Taste breadth', 'narrow', f'genre entropy <= {breadth_cut:.3f} bits', base.index.map(entropy) <= breadth_cut),
              ('Taste breadth', 'broad', f'genre entropy > {breadth_cut:.3f} bits', base.index.map(entropy) > breadth_cut)]
    breakdown = []
    for lam in sorted({selected, 0.5}, reverse=True):
        other = frames['test', lam].set_index('user_id').reindex(base.index)
        for dimension, group, definition, mask in groups:
            mask = np.asarray(mask)
            for metric in ['NDCG@10', 'ILD', 'Calibration error']:
                delta = (other[metric] - base[metric]).to_numpy()[mask]
                lo, hi = mean_ci(delta, seed=SEED)
                breakdown.append({'lambda': lam, 'Dimension': dimension, 'Group': group, 'Definition': definition,
                                  'Users': int(mask.sum()), 'Metric': metric,
                                  'Base (lambda=1)': base[metric].to_numpy()[mask].mean(),
                                  'MMR': other[metric].to_numpy()[mask].mean(),
                                  'Difference': delta.mean(), 'CI lower': lo, 'CI upper': hi})

    # Why popularity hardly moves: how much long tail is even in the candidate pool?
    pools = [model.candidates(u)[0] for u in base.index]
    pool_tail = float(np.mean([1 - head[p].mean() for p in pools]))
    pool_coverage = len(set(np.concatenate(pools))) / len(data.items)
    logging.info('Candidate pool: long-tail share %.4f, union covers %.4f of catalogue', pool_tail, pool_coverage)

    # Results-template rows (exact headers of the 'Part 3A - Diversity' sheet), test set.
    template = []
    for _, row in sweep[sweep.Split == 'test'].iterrows():
        lam = row['lambda']
        note = (f"test; n={int(row.Users)} positive-history users; pool=top-{POOL}; "
                f"long-tail share={row['Long-tail share']:.4f}; "
                + ('base recommender, no re-ranking' if lam == 1 else 'MMR re-ranked')
                + ('; SELECTED on validation' if lam == selected else ''))
        template.append({'Base model': f'User-kNN (k={BASE_K})', 'Re-ranking method': 'MMR' if lam < 1 else 'None (base)',
                         'Trade-off parameter': f'lambda={lam:g}', 'Item distance': DISTANCE_NAME,
                         'Precision@10': row['Precision@10'], 'Recall@10': row['Recall@10'], 'NDCG@10': row['NDCG@10'],
                         'Intra-list diversity': row['Intra-list diversity'],
                         'Catalogue coverage': row['Catalogue coverage'], 'Gini': row['Gini'],
                         'Calibration error': row['Calibration error'], 'Notes': note})

    # One typical user (base ILD closest to the median) to show the effect concretely.
    typical = (base.ILD - base.ILD.median()).abs().idxmin()
    example = []
    for lam in (1.0, selected):
        for rank, movie in enumerate(model.recommend_fn(lam)(typical), 1):
            row = data.movies.iloc[data.item_index[movie]]
            example.append({'user_id': typical, 'lambda': lam, 'rank': rank, 'movie_id': movie,
                            'title': row.title, 'genres': row.genres,
                            'relevant_in_test': movie in set(test[(test.user_id == typical) & (test.rating >= 4)].movie_id),
                            'popular_head': bool(head[data.item_index[movie]])})

    pd.DataFrame(template).to_csv(output_dir / 'diversity.csv', index=False)
    sweep.to_csv(output_dir / 'sweep.csv', index=False)
    pd.DataFrame(differences).to_csv(output_dir / 'paired_differences.csv', index=False)
    pd.DataFrame(breakdown).to_csv(output_dir / 'breakdown.csv', index=False)
    pd.DataFrame(example).to_csv(output_dir / 'example_user.csv', index=False)
    per_user_rows = pd.concat([frames['test', lam].assign(lambda_=lam) for lam in LAMBDAS])
    per_user_rows.drop(columns='recommendations').rename(columns={'lambda_': 'lambda'}).to_csv(
        output_dir / 'per_user_metrics.csv', index=False)
    plot_frontier(sweep, selected, output_dir / 'frontier.png')
    metadata = {'seed': SEED, 'base_model': f'User-kNN k={BASE_K}', 'candidate_pool': POOL, 'lambdas': LAMBDAS,
                'selection_rule': f'lowest lambda with validation NDCG@10 >= {1 - MAX_NDCG_LOSS:g} x base',
                'selected_lambda': selected, 'item_distance': DISTANCE_NAME, 'genres': genre_names,
                'calibration_alpha': CAL_ALPHA, 'popular_head_items': int(head.sum()),
                'test_pool_long_tail_share': pool_tail, 'test_pool_catalogue_coverage': pool_coverage,
                'activity_threshold': activity_cut, 'taste_breadth_threshold_bits': breadth_cut,
                'example_user': int(typical), 'python': platform.python_version(),
                'files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [data_dir/'ratings.dat', data_dir/'movies.dat', data_dir/'users.dat']}}
    (output_dir / 'run_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    logging.info('Part 3 results written to %s', output_dir)


if __name__ == '__main__':
    main()
