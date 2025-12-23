import os
import sys

# ensure project root is importable when running pytest from the project folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import train
import train_mlflow


def test_train_runs():
	# smoke test: should run without error
	train.main(n_estimators=10, test_size=0.2, random_state=0)


def test_train_mlflow_runs(tmp_path):
	# smoke test: runs the MLflow training using a small RF
	train_mlflow.main(n_estimators=10, random_state=0, test_size=0.2)
