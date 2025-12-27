# GitHub Actions: ML pipeline setup and usage

This document explains how to configure GitHub and GitHub Actions for the project's CI/CD pipeline, and the steps performed by the workflow.

1) Repository setup (GitHub)
- Ensure GitHub Actions are enabled for the repository (default on GitHub).
- Add repository secrets (Settings → Secrets → Actions):
  - `DVC_REMOTE_URL` — DVC remote URL (example: `s3://my-bucket/path` or `gs://my-bucket/path`)
  - If using S3: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
  - (Optional) any cloud provider creds required by your DVC remote

2) Workflow file
- Location: `.github/workflows/ml-pipeline.yml`
- Triggers: runs on every `push` to `main`/`master`.

3) What the workflow does (high level)
- Checkout the repository with full history (`fetch-depth: 0`).
- Install Python and Python dependencies from `requirements.txt`.
- Install `dvc` (with S3 extras in the example).
- Configure a DVC remote from the secret `DVC_REMOTE_URL` if provided.
- Run `dvc pull` to restore cached data, then `dvc repro` to reproduce the pipeline stages defined in `dvc.yaml`.
- Run `train_mlflow.py` (fallback training command) to ensure an MLflow run is executed.
- Run tests (`pytest`).
- Upload the `mlruns` directory as an artifact so MLflow runs are available in the Actions UI.

4) How to verify the pipeline
- Make a change to the `data/` files or to code (for example `train_mlflow.py`).
- Commit and push the change to `main`/`master`.
- Open the repository **Actions** tab → select the latest `ML Pipeline` run → inspect logs for `DVC pull`, `dvc repro`, training and tests.

5) MLflow traceability
- Each run of `train_mlflow.py` will create a run under the repository `mlruns/` directory.
- The workflow uploads `mlruns/` as an artifact; download it from the Actions run to inspect runs locally.
- To compare runs locally, run `mlflow ui` in a copy of the repository containing `mlruns/` and browse `http://localhost:5000`.

6) Local quick commands
```bash
# install deps
python -m pip install -r requirements.txt

# reproduce pipeline locally
dvc pull -r origin
dvc repro

# run training and view MLflow UI
python train_mlflow.py --n_estimators 150 --random_state 42
mlflow ui
```

7) Notes and troubleshooting
- Make sure the DVC remote is a true object/storage remote (S3, GCS, Azure) or a local filesystem path — GitHub (https://github.com/...) is not a valid DVC remote for caches.
- If `dvc repro` fails due to missing cache, check `dvc status -c` locally and ensure necessary cache files are present in the DVC remote.
- Ensure `requirements.txt` pins versions compatible with the workflow Python version (3.10 in the workflow).
