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

Questions / Réponses:

- Pourquoi MLflow est-il indispensable en MLOps ?
  MLflow fournit un tracking centralisé des expériences, paramètres, métriques et modèles. Il facilite la reproductibilité, la comparaison de runs, la gestion des artefacts et l'intégration dans des pipelines CI/CD et de déploiement. En MLOps, le suivi systématique et la traçabilité sont essentiels; MLflow automatise et standardise ces aspects.

- Quelle différence entre un `run` et un `experiment` ?
  Un `experiment` est un conteneur logique rassemblant plusieurs `runs`. Un `run` correspond à une exécution unique (une tentative d'entraînement) avec ses paramètres, métriques et artefacts; un `experiment` permet d'organiser et comparer plusieurs runs du même projet.

- Peut-on reproduire un modèle sans tracking ?
  Théoriquement oui si vous conservez manuellement le code, la seed, les données et les hyperparamètres. En pratique, sans tracking il est facile d'oublier des détails (versions, preprocessings, seeds) rendant la reproduction difficile. Le tracking réduit ces risques.

---

Si vous voulez, je peux:
- Installer les dépendances et lancer une exécution de `train_mlflow.py` ici.
- Lancer `mlflow ui` pour vous (si vous confirmez).

Dites quelle action je dois faire ensuite.
