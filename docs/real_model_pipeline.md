# Real-model pipeline: Git + DVC + MLflow + GitHub Actions

This document describes the added real-model experiment pipeline, tools used, advanced feature, and how to run/verify it.

Overview
- Tools: Git, DVC, MLflow, scikit-learn, GitHub Actions
- New code: `real_model/train_real.py`, `real_model/experiment_runner.py`
- DVC: added `train_real` stage in `dvc.yaml` which runs `python real_model/experiment_runner.py`.

Datasets
- Uses sklearn bundled datasets: `wine`, `breast_cancer`, `digits` (three different, real datasets).
- The `experiment_runner.py` runs training on each dataset with two hyperparameter settings (2 values of `n_estimators`) — this satisfies changing dataset >2 times.

What the pipeline does
1. `dvc repro` will run existing stages. New `train_real` stage runs `real_model/experiment_runner.py`.
2. Each experiment trains a `RandomForestClassifier`, logs parameters and metrics (`accuracy`, `precision`) and the model to MLflow under `mlruns/real-model-experiments`.
3. Feature importances are saved as artifacts per run.

Advanced feature
- Hyperparameter sweep & multi-dataset experiment runner: multiple MLflow runs are created automatically (one per dataset × hyperparameter configuration). This provides simple experiment tracking and comparability.

How to run locally
```powershell
# create venv and install deps (if not done)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# reproduce DVC pipeline (will run train + train_real stages if needed)
dvc repro

# or run the real experiments directly
python real_model/experiment_runner.py

# view MLflow UI
mlflow ui --backend-store-uri ./mlruns
# open http://localhost:5000
```

How to verify on GitHub Actions
- Push changes to `main`/`master`.
- The `ml-pipeline.yml` workflow runs `dvc repro`, which will execute `train_real` stage and upload `mlruns` artifact.
- Download the `mlruns` artifact from the Action run to inspect MLflow runs locally or run `mlflow ui` against it.

Notes
- If you want to use a cloud dataset, modify `real_model/train_real.py` to fetch it and mark the downloaded file in DVC.
- For larger experiments consider using `dvc run`/stages that capture model outputs separately (e.g., `models/` directory) and store cache in a remote (S3/GCS).
