"""Run experiments on multiple real datasets and log runs to MLflow."""
import os
import sys

# Ensure repo root is on sys.path so imports like `real_model.train_real` work
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from real_model.train_real import train_and_log


def main():
    datasets = ["wine", "breast_cancer", "digits"]
    n_estimators_options = [50, 100]
    for ds in datasets:
        for n in n_estimators_options:
            train_and_log(ds, n)


if __name__ == "__main__":
    main()
