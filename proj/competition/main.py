import numpy as np
import importlib
import src.fncs
import matplotlib.pyplot as plt
import random
import argparse

from src.training import training


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    training(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
