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
├── run_all.py            # Entry point (starter placeholder)
├── requirements.txt      # Pin dependencies when selected
├── .gitignore
└── README.md
```

## Getting the data

Download **MovieLens 1M** from https://grouplens.org/datasets/movielens/1m/

Extract it locally so the files are accessible as:

- `data/ml-1m/ratings.dat`
- `data/ml-1m/movies.dat`
- `data/ml-1m/users.dat`

**Do not commit the dataset** to this repository. The `.gitignore` excludes it.

## Environment setup

Use Python (record the exact version your team chooses here). Create and activate a virtual environment, then install pinned dependencies after you fill in `requirements.txt`:

```bash
python -m venv .venv
# On macOS/Linux:
source .venv/bin/activate
# On Windows PowerShell:
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Reproduce results

The intended final command is:

```bash
python run_all.py --data-dir ./data/ml-1m
```

**Important:** This downloaded ZIP is a **starter folder structure only**. `run_all.py` and the Python modules are placeholders; they do not train or evaluate any models yet. Implement them before submitting. The final command must reproduce every metric and figure in the report, using fixed documented random seeds.

## Deliverables and team agreements

- Use the **official course report and results CSV templates** and keep one report throughout Parts 1–3.
- Commit source code, dependencies, reports, slides, and required results to GitHub.
- Document model choices, evaluation protocol, hyperparameter budgets, and seeds.
- Write contribution statements for each team member and disclose any AI-assistant use in the report.

## Team

- Member 1: [name / GitHub username]
- Member 2: [name / GitHub username]
- Member 3: [name / GitHub username]
