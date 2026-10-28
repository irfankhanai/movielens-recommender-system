"""Project entry point (starter placeholder; no experiments implemented yet).

Final target:
    python run_all.py --data-dir ./data/ml-1m
"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce MovieLens recommender experiments")
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to extracted ml-1m data")
    args = parser.parse_args()
    if not args.data_dir.is_dir():
        parser.error(f"Dataset directory does not exist: {args.data_dir}")
    raise NotImplementedError("Starter project only: implement data loading, models, and evaluation in run_all.py")


if __name__ == "__main__":
    main()
