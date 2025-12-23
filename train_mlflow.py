import argparse
import logging
import os
import json
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score


def main(n_estimators: int, random_state: int, test_size: float = 0.25):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Use a local file-based tracking URI by default to avoid DB/alembic issues in minimal setups
    if os.environ.get("MLFLOW_TRACKING_URI") is None:
        from pathlib import Path
        Path("mlruns").mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(Path("mlruns").absolute().as_uri())
    mlflow.set_experiment("iris-mlops")
    with mlflow.start_run():
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="macro", zero_division=0)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.sklearn.log_model(model, "model")

        # save and log feature importances as artifact
        fi = getattr(model, "feature_importances_", None)
        if fi is not None:
            cols = load_iris(as_frame=True).feature_names
            df_fi = pd.DataFrame({"feature": cols, "importance": fi})
            os.makedirs("artifacts", exist_ok=True)
            path = os.path.join("artifacts", "feature_importances.csv")
            df_fi.to_csv(path, index=False)
            mlflow.log_artifact(path, artifact_path="feature_importances")

        logging.info(f"Accuracy: {acc:.4f}  Precision: {prec:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train iris RF and log with MLflow")
    parser.add_argument("--n_estimators", type=int, default=150)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.25)
    args = parser.parse_args()
    main(args.n_estimators, args.random_state, args.test_size)
