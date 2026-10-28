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




## Team

- Member 1: Irfan Khan / 153218338 / GitHub:Irfankhanai
- Member 2: [name / GitHub username]
- Member 3: [name / GitHub username]
