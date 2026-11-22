# MovieLens Recommender System

Master's-level Recommender Systems course project using the **MovieLens 1M** dataset.

## Project overview

Build, evaluate, and compare movie recommendation methods. The project has three parts:

1. **Part 1 (Week 3):** Data exploration, temporal train/validation/test split, most-popular variants and item-based kNN baselines.
2. **Part 2 (Week 5):** At least three additional models from at least two paradigms; one additional algorithm implemented from scratch; tuning, uncertainty analysis, group-wise comparisons, sparsity tests.
3. **Part 3 (Week 6/7):** Choose one track (diversity, fairness, or group recommendations), study trade-offs, present and report the results.

## Project structure

```text
movielens-recommender-system/
├── data/                 # Place local MovieLens 1M files here (not committed)
├── src/                  # Data processing, baselines, models, evaluation
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── baselines.py
│   ├── models.py
│   └── evaluation.py
├── results/              
│   ├── plots/            # Plot outputs
│   ├── knn_tuning.csv    # Outputs of the kNN tuning runs
│   └── test_results.csv  # Outputs of the final test result
├── reports/             
│   ├── Recommender_Systems_Report_Group_9.docx  # One evolving course report
│   └── Recommender_Systems_Results_Group_9.xlsm          # One evolving project results
├── run_all.py            # Experiment entry point
├── requirements.txt      # Pinned, locally verified dependencies
├── .gitignore
└── README.md
```

## Getting the data

Raw data is excluded from Git. Each teammate should do this after cloning:

1. Open the official [MovieLens 1M page](https://grouplens.org/datasets/movielens/1m/).
2. Download **ml-1m.zip** and extract it.
3. Put the extracted **ml-1m** folder inside the repository's **data** folder.

Expected files:

```text
data/ml-1m/movies.dat
data/ml-1m/ratings.dat
data/ml-1m/users.dat
```

Read the dataset's included README for usage terms and citation information.
Run the commands below from the repository root. If the files are already
directly in data/, use --data-dir ./data instead.

## Install dependencies

```bash
python -m venv .venv
# On macOS/Linux:
source .venv/bin/activate
# On Windows PowerShell:
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Reproduce results

Run with the directory containing movies.dat, ratings.dat, and users.dat:

```bash
python run_all.py --data-dir ./data/ml-1m
```

This now runs **Parts 1 and 2**. To run only the new experiments:

```bash
python run_all.py --data-dir ./data/ml-1m --part 2
```

Use `--part 1` for the original baseline workflow. Python **3.14.6** and the
versions in `requirements.txt` reproduce the checked-in results. No new runtime
dependencies are needed for Part 2. Run the small correctness checks with
`python -m unittest discover -s tests -v`.

## Part 2 on branch irfan

Exactly three new models are tested: user-kNN (neighbourhood CF), content profile
matching (content-based), and a weighted hybrid. `ContentProfile` in
`src/models.py` is the designated self-implemented core algorithm, following
the profile/cosine method in Lecture 2.2, slides 4-7. Its genre and release-decade
features, user profiles and scores are implemented directly with NumPy.

The content vector concatenates a normalized genre vector scaled by
`sqrt(1-year_weight)` with a one-hot release-decade vector scaled by
`sqrt(year_weight)`. A user's profile is the normalized sum of their liked movie
vectors. The score is their dot product (cosine similarity). Decades smooth the
release-year signal without learning extra parameters. Exact feature ties use
movie ID, which can affect this deliberately coarse model.

User-kNN uses cosine similarity on binary positive histories, excludes the user
from their own neighbours, and sums similarity-weighted likes. The hybrid mixes
user-kNN (`k=40`) and content (`year_weight=0.25`), with these components fixed in
advance. Scores are divided by each component's maximum over unseen candidates
before mixing: `alpha*CF + (1-alpha)*content`. The fixed components avoid giving
the hybrid a larger hidden tuning budget. All models exclude every training-rated
movie, including dislikes. Empty profiles fall back to positive popularity.

### Protocol and minimum search

The Part 1 global 80/10/10 timestamp split and rating >=4 relevance definition
are unchanged. Fit on train only, choose maximum validation NDCG@10, break
validation ties by grid order, and evaluate test after selection. No train+val
refit. Rank all 3,883 catalogue movies minus seen items; no negative sampling.
Primary cohort: users with a positive training history and at least one relevant
held-out movie (492 validation, 991 test). Also report all 1,188 eligible test
users, including cold users. Users without held-out positives are not evaluated.
Precision always divides by 10; recall divides by the user's relevant items;
NDCG uses binary gain. Coverage is the union of recommended movies / 3,883,
not the average of user coverage. These metrics agree with scikit-learn on a
small checked example.

Each tunable model gets **four validation configurations**, 16 in total:

| Model | Search space |
|---|---|
| Item-kNN baseline | k in {20, 40} x similarity in {Cosine, Jaccard} |
| User-kNN | k in {10, 20, 40, 80} |
| Content | year_weight in {0.1, 0.25, 0.5, 0.75} |
| Hybrid | alpha in {0.25, 0.5, 0.75, 0.9} |

Both popularity variants have no tunable parameters. Part 1's historical
12-setting kNN search remains reproducible; Part 2 repeats a four-setting subset
including its winner to keep this comparison small and equally budgeted. No
settings were added after seeing test results. This is a limited comparison,
not an exhaustive search for each model's best achievable performance.

Models are deterministic. Uncertainty uses a paired, two-sided sign-flip test
across the same 991 users (10,000 randomizations) against item-kNN, with Holm
correction over 15 comparisons (five alternatives x three training sizes).
Also report 95% percentile bootstrap intervals for mean NDCG and paired mean
differences (2,000 user resamples). All random operations use seed 42. Intervals
are pointwise, not simultaneous, and quantify user sampling variability, not
variation across temporal splits or training subsamples. Use the adjusted tests
for claims across multiple comparisons.

Breakdowns are training user activity (<=86 vs >86 ratings; median over users
observed in full train) and early/late test events (<=993074159 vs >993074159
UTC seconds, the median test timestamp). The latter only defines a descriptive
evaluation slice; it never enters training or tuning. For each period, reuse
the same top-10 rankings and recompute relevance in that period. Users may occur
in both periods, and their different relevance sets prevent interpreting the
period contrast as a causal drift estimate.

Sparsity uses nested, uniform samples without replacement of **all training
rating rows**, then applies the positive threshold. Fractions are 100%, 50%, 20%
(800,164 / 400,082 / 160,032 rows). Refit all six models at each fraction using
the full-data selected settings; do not retune. Keep test, the original warm
cohort, and original seen-item exclusions fixed. This isolates loss of learning
evidence; it does not simulate a service that forgets which movies were seen.
One subsample seed is intentional; the user-bootstrap intervals do not measure
subsampling uncertainty.

### Results and submission files

`results/part2/models.csv`, `disaggregated.csv`, and `sparsity.csv` use the exact
headers and order of row 6 in the supplied workbook's corresponding Part 2 tabs.
These are the updated submission CSVs; the `.xlsm` retains the original Part 1
workbook. Additional audit outputs are `tuning.csv`, `selected_settings.json`,
`paired_differences.csv`, `per_user_metrics.csv`, and `run_metadata.json`.
Metadata records data hashes, split sizes, seed and Python version. All are
regenerated by the command above. The existing cumulative report at
`reports/Recommender_Systems_Report_Group_9.docx` is extended in place.

At full data, item-kNN has NDCG 0.21685 and user-kNN 0.21611; their paired
difference interval includes zero. Content has much lower NDCG (0.01876), despite
higher catalogue coverage (20.45%). At 20% data, positive popularity scores
0.19789 versus item-kNN's 0.06668. Coverage alone therefore does not establish
recommendation quality. The coarse content representation and noisier item
similarities are plausible explanations, not proven causal findings.

Part 3 track announced for this branch: **Track A, diversity and popularity
bias**, with genre-based MMR planned. Part 3 is not implemented here.
AI assistance is disclosed in the cumulative report. Contributions on teammates'
separate branches are not included or attributed by this branch.




## Team

- Member 1: Irfan Khan / 153218338 / GitHub: Irfankhanai
- Member 2: Kaan Pide / 152267623 / GitHub: klsaerf
- Member 3: Reinis Gustavs Ignatjevs / 152215565 / GitHub: ExtraAngel

## Outputs and environment

The pinned dependencies were verified with Python 3.14. The source avoids
Python 3.12-only f-string syntax; older interpreters may need different package versions.

For files directly inside this checkout's data directory, run:

```bash
python run_all.py --data-dir ./data
```

By default the command saves six PNG plots in results/plots and writes
test_results.csv, validation_results.csv, and knn_tuning.csv.
Use --output-dir PATH to choose another output location and --show-plots
to also open interactive figures. Test and validation CSVs have an explicit
Model column.

Raw MovieLens files are ignored by Git. Download and extract them locally;
untracking them does not remove copies from earlier Git history.

## Report Links

Link of the report: https://1drv.ms/w/c/177f4cc807f3bf5e/IQD5x2IMgCwRRI5CtYQvwgJfASXGNfXRLlU4diSGVNa_t_Q?e=38vRHh

Link of the Excel sheet for the results: https://1drv.ms/x/c/177f4cc807f3bf5e/IQATyvUshz6gQ6-Ho7eYhV1xAZfoiG5a2xt3-WPx0-gvNgc?e=KIDiKW
