# mlops-mlflow-tp

Projet d'exemple pour MLOps avec MLflow (jeu de données Iris)

Structure:
- requirements.txt
- train.py (exécution simple)
- train_mlflow.py (tracking MLflow avec paramètres et métriques)

Installation (Windows PowerShell):

1. Créer et activer un environnement virtuel (recommandé)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Installer les dépendances

```powershell
pip install -r requirements.txt
```

Exemples d'utilisation:

- Lancer l'entraînement simple:

```powershell
python train.py
```

- Lancer l'entraînement avec MLflow (paramètres modifiables):

```powershell
python train_mlflow.py --n_estimators 150 --random_state 42
python train_mlflow.py --n_estimators 50 --random_state 0
```

- Démarrer l'UI MLflow (dans un terminal séparé):

```powershell
mlflow ui
```

Puis ouvrir http://localhost:5000

Ce que vous verrez dans l'UI MLflow:
- Liste des runs pour l'experiment `iris-mlops`
- Comparaison des métriques (accuracy, precision)
- Paramètres utilisés (`n_estimators`, `random_state`)
- Modèles sauvegardés (section "Artifacts" -> `model`)

Explication du code (`train_mlflow.py`):
- On fixe l'experiment avec `mlflow.set_experiment("iris-mlops")` pour regrouper les runs.
- Pour chaque exécution on ouvre un `mlflow.start_run()` qui crée un run isolé.
- Le modèle `RandomForestClassifier` est entraîné sur Iris, puis on calcule `accuracy` et `precision`.
- On loggue les paramètres (`mlflow.log_param`) et les métriques (`mlflow.log_metric`).
- On sauvegarde le modèle avec `mlflow.sklearn.log_model` (stocké comme artifact du run).

Modifications demandées (exemples):
- Changer `n_estimators` en passant `--n_estimators` à `train_mlflow.py`.
- Changer `random_state` en passant `--random_state`.
- Ajouter la métrique `precision` (déjà implémentée dans `train_mlflow.py`):

```python
from sklearn.metrics import precision_score
mlflow.log_metric("precision", precision_score(y_test, preds, average="macro"))
```

## Automatisation (CI & versioning de données)

Deux workflows GitHub Actions ont été ajoutés pour automatiser les tests et le versioning des données :

- `.github/workflows/automation.yml` : installe les dépendances (`requirements.txt`) et exécute `pytest` sur push/PR vers `main` ou `master`.
- `.github/workflows/dvc_data_versioning.yml` : déclenché quand `data/**` change (ou manuellement). Il installe `dvc`, exécute `dvc add -R data`, commit les fichiers `.dvc` / changements Git, pousse le commit et exécute `dvc push` vers le remote configuré.

Secrets recommandés pour DVC (configurer dans les Settings > Secrets du dépôt) :
- `DVC_REMOTE_URL` (ex: `s3://my-bucket/path` ou autre remote DVC)
- pour S3 : `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`

Exemples locaux pour initialiser un remote DVC et pousser les données :

```bash
# ajouter un remote (ex: s3)
dvc remote add -d origin s3://my-bucket/path
dvc remote modify origin access_key_id $AWS_ACCESS_KEY_ID
dvc remote modify origin secret_access_key $AWS_SECRET_ACCESS_KEY

# tracker et pousser
dvc add -R data
git add -A
git commit -m "chore(dvc): track data"
git push
dvc push
```

Notes : si vous utilisez un autre provider (GCP, Azure), configurez le remote DVC correspondant et fournissez les secrets nécessaires dans GitHub.

