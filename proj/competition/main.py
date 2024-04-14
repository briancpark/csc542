"""Main script for the competition."""

import argparse
from src.training import training
from src.check import check_submission_format

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", "-b", type=int, default=64)
    parser.add_argument("--epochs", "-e", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--window-size", "-w", type=int, default=100)
    args = parser.parse_args()

    if args.check:
        check_submission_format()
    else:
        training(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            window_size=args.window_size,
        )
