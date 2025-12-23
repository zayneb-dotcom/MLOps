import argparse
import logging
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def main(n_estimators: int = 100, test_size: float = 0.25, random_state: int = 42):
	logging.basicConfig(level=logging.INFO, format="%(message)s")
	X, y = load_iris(return_X_y=True)
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=test_size, random_state=random_state, stratify=y
	)
	model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
	model.fit(X_train, y_train)
	preds = model.predict(X_test)
	acc = accuracy_score(y_test, preds)
	logging.info(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train a RandomForest on Iris")
	parser.add_argument("--n_estimators", type=int, default=100)
	parser.add_argument("--test_size", type=float, default=0.25)
	parser.add_argument("--random_state", type=int, default=42)
	args = parser.parse_args()
	main(args.n_estimators, args.test_size, args.random_state)
