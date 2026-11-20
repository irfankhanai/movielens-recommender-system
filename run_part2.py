"""Minimal Part 2 experiments; outputs use the supplied results-template headers."""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing import get_data, split_data
from src.models import ModelData, ItemKNN, UserKNN, ContentProfile, Hybrid
from src.part2_evaluation import per_user, aggregate, mean_ci, paired_test

SEED = 42
GRIDS = {
    'Item-kNN': [dict(k=k, similarity=s) for s in ['Cosine', 'Jaccard'] for k in [20, 40]],
    'User-kNN': [dict(k=k) for k in [10, 20, 40, 80]],
    'Content': [dict(year_weight=w) for w in [0.1, 0.25, 0.5, 0.75]],
    'Hybrid': [dict(alpha=a) for a in [0.25, 0.5, 0.75, 0.9]],
}
PARADIGMS = {'Popular (All)': 'Popularity', 'Popular (Positive)': 'Popularity',
             'Item-kNN': 'Neighbourhood CF', 'User-kNN': 'Neighbourhood CF',
             'Content': 'Content-based', 'Hybrid': 'Hybrid'}


def build(name, data, settings):
    if name.startswith('Popular'):
        counts = data.all_counts if name == 'Popular (All)' else data.popular
        return lambda u: data.rank(u, counts)
    cls = {'Item-kNN': ItemKNN, 'User-kNN': UserKNN, 'Content': ContentProfile, 'Hybrid': Hybrid}[name]
    return cls(data, **settings).recommend


def main(data_dir=None, output_dir=None):
    if data_dir is None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--data-dir', type=Path, required=True)
        parser.add_argument('--output-dir', type=Path, default=Path('results/part2'))
        args = parser.parse_args()
        data_dir, output_dir = args.data_dir, args.output_dir
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    movies, ratings, users = get_data(data_dir)
    train, val, test = split_data(ratings)
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError('The temporal split must contain nonempty train, validation and test sets.')
    assert train.timestamp.max() < val.timestamp.min() <= val.timestamp.max() < test.timestamp.min()
    eligible = set(train.loc[train.rating >= 4, 'user_id'])
    data = ModelData(train, movies, users)
    selected = {'Popular (All)': {}, 'Popular (Positive)': {}}
    tuning = []
    for name, grid in GRIDS.items():
        best_score = -np.inf
        for settings in grid:
            logging.info('Validation %s %s', name, settings)
            scores = aggregate(per_user(build(name, data, settings), val, eligible), len(movies))
            tuning.append({'Model': name, 'Settings': json.dumps(settings), **scores})
            if scores['NDCG@10'] > best_score:
                selected[name], best_score = settings, scores['NDCG@10']
    pd.DataFrame(tuning).to_csv(output_dir/'tuning.csv', index=False)
    (output_dir/'selected_settings.json').write_text(json.dumps(selected, indent=2)+'\n')

    overall, groups, sparsity, differences, user_rows = [], [], [], [], []
    activity = train.groupby('user_id').size()
    activity_cut = float(activity.median())
    time_cut = int(test.timestamp.median())
    # One seeded permutation produces nested 20% <= 50% <= 100% samples.
    permutation = np.random.default_rng(SEED).permutation(len(train))
    for fraction in [1.0, 0.5, 0.2]:
        sample = train if fraction == 1 else train.iloc[permutation[:int(len(train)*fraction)]]
        data = ModelData(sample, movies, users, seen_train=train)
        fraction_frames = {}
        for name, settings in selected.items():
            logging.info('Test fraction=%s %s %s', fraction, name, settings)
            recommend = build(name, data, settings)
            # Cache rankings once per model, reuse for period/cohort analyses.
            predictions = {u: recommend(u) for u in sorted(test.loc[test.rating >= 4, 'user_id'].unique())}
            frame_all = per_user(predictions.__getitem__, test)
            frame = frame_all[frame_all.user_id.isin(eligible)].copy()
            fraction_frames[name] = frame.set_index('user_id')['NDCG@10']
            metrics = aggregate(frame, len(movies))
            low, high = mean_ci(frame['NDCG@10'])
            sparsity.append({'Model': name, 'Training interactions (%)': int(100*fraction),
                             **metrics, 'Seed(s)': SEED,
                             'Notes': f'n={len(frame)}; rows={len(sample)}; fixed cohort/seen; fixed settings; NDCG 95% CI [{low:.6f}, {high:.6f}]'})
            saved = frame.assign(Model=name, training_fraction=fraction).drop(columns='recommendations')
            user_rows.append(saved)
            if fraction != 1:
                continue
            for cohort, evaluated in [('positive_history', frame), ('all_users', frame_all)]:
                lo, hi = mean_ci(evaluated['NDCG@10'])
                overall.append({'Model': name, 'Paradigm': PARADIGMS[name],
                                'Self-implemented?': ('Yes (designated core algorithm)' if name == 'Content' else
                                                      'Yes' if name in ['User-kNN', 'Hybrid'] else 'Part 1 baseline'),
                                'Key hyperparameters': json.dumps(settings),
                                **aggregate(evaluated, len(movies)),
                                'Variability / CI / test': f'User bootstrap NDCG 95% CI [{lo:.6f}, {hi:.6f}]; paired differences in paired_differences.csv',
                                'Seed(s)': SEED, 'Notes': f'{cohort}; n={len(evaluated)}; fit train only'})
            splits = [('User activity', 'low', f'1..{activity_cut:g} training ratings',
                       frame[frame.user_id.map(activity) <= activity_cut]),
                      ('User activity', 'high', f'>{activity_cut:g} training ratings',
                       frame[frame.user_id.map(activity) > activity_cut])]
            for label, subset in [('early', test[test.timestamp <= time_cut]),
                                  ('late', test[test.timestamp > time_cut])]:
                splits.append(('Test period', label, f'early <= {time_cut}; late > {time_cut} UTC seconds',
                               per_user(predictions.__getitem__, subset, eligible)))
            for dimension, group, definition, subset in splits:
                lo, hi = mean_ci(subset['NDCG@10'])
                groups.append({'Model': name, 'Breakdown dimension': dimension, 'Group': group,
                               'Definition / threshold': definition, **aggregate(subset, len(movies)),
                               'Seed(s)': SEED, 'Notes': f'positive_history; n={len(subset)}; NDCG 95% CI [{lo:.6f}, {hi:.6f}]'})
        reference = fraction_frames['Item-kNN']
        for name, values in fraction_frames.items():
            if name == 'Item-kNN':
                continue
            delta = values.reindex(reference.index) - reference
            lo, hi = mean_ci(delta)
            differences.append({'Model': name, 'Reference': 'Item-kNN', 'Training interactions (%)': int(100*fraction),
                                'Metric': 'NDCG@10', 'Mean difference': delta.mean(),
                                'CI lower': lo, 'CI upper': hi, 'Users': len(delta),
                                'Paired sign-flip p': paired_test(delta)})

    # Holm correction across all 15 planned comparisons (5 models x 3 sizes).
    order = sorted(range(len(differences)), key=lambda i: differences[i]['Paired sign-flip p'])
    adjusted = 0
    for rank, index in enumerate(order):
        adjusted = max(adjusted, min(1, (len(order)-rank)*differences[index]['Paired sign-flip p']))
        differences[index]['Holm adjusted p'] = adjusted

    # Column names/order match row 6 of the supplied workbook's Part 2 tabs.
    for filename, rows in [('models.csv', overall), ('disaggregated.csv', groups),
                           ('sparsity.csv', sparsity), ('paired_differences.csv', differences)]:
        pd.DataFrame(rows).to_csv(output_dir/filename, index=False)
    pd.concat(user_rows).to_csv(output_dir/'per_user_metrics.csv', index=False)
    metadata = {'seed': SEED, 'bootstrap_repeats': 2000, 'sign_flip_repeats': 10000,
                'python': __import__('platform').python_version(),
                'split_sizes': dict(zip(['train', 'validation', 'test'], map(len, [train, val, test]))),
                'activity_threshold': activity_cut, 'test_time_threshold': time_cut,
                'track': 'A: diversity and popularity bias; planned MMR using genres',
                'files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [data_dir/'ratings.dat', data_dir/'movies.dat', data_dir/'users.dat']}}
    (output_dir/'run_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    logging.info('Part 2 results written to %s', output_dir)


if __name__ == '__main__':
    main()
