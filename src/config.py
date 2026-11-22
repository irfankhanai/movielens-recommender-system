"""Shared constants across the project"""
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

SEED = 42

# Data
POSITIVE_THRESHOLD = 4      # ratings >= this count as "liked"
VAL_QUANTILE = 0.80         # global temporal split: train < 80% < val < 90% < test
TEST_QUANTILE = 0.90

# Evaluation
TOP_N = 10

# Multi-VAE: architecture
VAE_HIDDEN_DIM = 300
VAE_LATENT_DIM = 100
VAE_DROPOUT = 0.5

# Multi-VAE: training
VAE_LEARNING_RATE = 1e-4
VAE_BATCH_SIZE = 128
VAE_MAX_EPOCHS = 100
VAE_PATIENCE = 20           # epochs without val improvement before stopping

# Multi-VAE: KL annealing
VAE_ANNEAL_CAP = 0.3        # maximum beta
VAE_ANNEAL_STEPS = 2000     # batches over which beta rises from 0 to the cap

# Item-kNN
KNN_DEFAULT_K = 20          # k for the initial kNN run before tuning
KNN_K_VALUES = [5, 10, 20, 30, 40, 50]
