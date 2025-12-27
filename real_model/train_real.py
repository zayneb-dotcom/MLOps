import argparse
import os
import logging
from typing import Tuple

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.datasets import load_wine, load_breast_cancer, load_digits


def load_dataset(name: str) -> Tuple[pd.DataFrame, pd.Series]:
    if name == "wine":
        data = load_wine(as_frame=True)
        X = data.data
        y = data.target
    elif name == "breast_cancer":
        data = load_breast_cancer(as_frame=True)
        X = data.data
        y = data.target
    elif name == "digits":
        data = load_digits()
        X = pd.DataFrame(data.data)
        y = pd.Series(data.target)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return X, y


def train_and_log(dataset: str, n_estimators: int, random_state: int = 42):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    mlflow.set_experiment("real-model-experiments")
    with mlflow.start_run(tags={"dataset": dataset, "n_estimators": n_estimators}):
        X, y = load_dataset(dataset)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=random_state, stratify=y if len(y.unique())>1 else None
        )

        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="macro", zero_division=0)

        mlflow.log_param("dataset", dataset)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)

        # log the sklearn model
        mlflow.sklearn.log_model(model, "model")

        # feature importances (if available)
        fi = getattr(model, "feature_importances_", None)
        if fi is not None:
            cols = X.columns.tolist() if hasattr(X, "columns") else [f"f{i}" for i in range(len(fi))]
            df_fi = pd.DataFrame({"feature": cols, "importance": fi})
            os.makedirs("real_model/artifacts", exist_ok=True)
            path = os.path.join("real_model", "artifacts", f"feature_importances_{dataset}_{n_estimators}.csv")
            df_fi.to_csv(path, index=False)
            mlflow.log_artifact(path, artifact_path="feature_importances")

        logging.info(f"dataset={dataset} n_estimators={n_estimators} accuracy={acc:.4f} precision={prec:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wine", help="dataset to use: wine|breast_cancer|digits")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    train_and_log(args.dataset, args.n_estimators, args.random_state)


if __name__ == "__main__":
    main()
