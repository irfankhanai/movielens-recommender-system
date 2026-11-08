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
├── experiments/          # Experiment configuration/scripts
├── results/              # Required output tables and plots
├── report/               # One evolving course report
├── slides/               # Week 7 examination slides
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
