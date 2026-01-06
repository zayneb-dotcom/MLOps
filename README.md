# Fraud Detection MLOps Pipeline

Complete MLOps pipeline for fraud detection with data versioning, model tracking, deployment, and monitoring.

##  Project Structure

```
mlOps-dataset/
├── data/
│   ├── raw/                    # Raw data tracked with DVC
│   │   └── fraudTest.csv
│   ├── processed/              # Cleaned data
│   │   └── transactions_clean.csv
│   └── features/               # Feature-engineered data
│       └── transactions_features.parquet
├── scripts/
│   ├── validate_data.py        # Data validation (Pandera)
│   ├── preprocess.py           # Data preprocessing
│   ├── feature_engineering.py  # Feature engineering
│   └── train_model.py          # Model training (MLflow)
├── models/
│   ├── xgboost_fraud_model.pkl # Trained model
│   └── scaler.pkl              # Feature scaler
├── api/
│   └── main.py                 # FastAPI deployment
├── monitoring/
│   └── generate_reports.py     # Evidently AI monitoring
├── dvc.yaml                    # DVC pipeline
├── params.yaml                 # Pipeline parameters
├── Dockerfile                  # Docker container
└── requirements.txt            # Python dependencies
```

##  Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Initialize Git & DVC

```bash
# Initialize git repository
git init
git add .
git commit -m "Initial commit"

# Initialize DVC
dvc init
git add .dvc .gitignore
git commit -m "Initialize DVC"
```

### 3. Move Raw Data & Track with DVC

```bash
# Move your dataset to the correct location
move data\fraudTest.csv data\raw\fraudTest.csv

# Track raw data with DVC
dvc add data/raw/fraudTest.csv
git add data/raw/fraudTest.csv.dvc data/raw/.gitignore
git commit -m "Track raw fraud dataset with DVC"
```

### 4. Run Complete Pipeline

```bash
# Run entire DVC pipeline (validation → preprocessing → feature engineering → training)
dvc repro
```

This will:
- ✅ Validate data quality (Pandera)
- ✅ Preprocess and clean data
- ✅ Engineer features
- ✅ Train XGBoost model with MLflow tracking

### 5. View MLflow Experiments

```bash
mlflow ui
```

Open http://localhost:5000 to view:
- Hyperparameters
- Metrics (Precision, Recall, F1, ROC-AUC)
- Model artifacts
- Feature importance

##  Data Validation

Validation rules enforced by Pandera:
- ✅ No missing values
- ✅ `amt` > 0
- ✅ `is_fraud` ∈ {0, 1}
- ✅ Consistent distributions

```bash
python scripts/validate_data.py --input data/raw/fraudTest.csv --output data/raw/fraudTest_validated.csv
```

## 🔧 Preprocessing & Feature Engineering

**Preprocessing:**
- Normalize `amt` (amount)
- Transform temporal features (`hour`, `day_of_week`)
- Encode categorical variables
- Handle missing values

**Feature Engineering:**
- Distance between customer and merchant
- Temporal features (night, weekend)
- Amount transformations (log, squared)
- Population features

##  Model Training

**XGBoost Classifier** (recommended for fraud detection):
- Handles imbalanced data well
- Captures complex patterns
- Production-proven for fraud detection

**Class Imbalance Handling:**
- SMOTE (Synthetic Minority Over-sampling)
- Class weight adjustment

**Metrics Tracked:**
- Precision
- Recall
- F1-Score
- ROC-AUC

```bash
python scripts/train_model.py --input data/features/transactions_features.parquet --model models/xgboost_fraud_model.pkl
```

##  API Deployment

FastAPI REST API for real-time predictions:

```bash
# Start API server
uvicorn api.main:app --reload
```

**Endpoints:**
- `GET /` - Health check
- `GET /health` - Detailed health status
- `POST /predict` - Single transaction prediction
- `POST /predict/batch` - Batch predictions

**Example request:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "amt": 107.23,
    "lat": 40.7128,
    "long": -74.0060,
    "merch_lat": 40.7589,
    "merch_long": -73.9851,
    "city_pop": 8000000,
    "unix_time": 1609459200,
    "hour": 14,
    "day_of_week": 3,
    "day_of_month": 15,
    "gender": 1
  }'
```

**Response:**

```json
{
  "is_fraud": 0,
  "fraud_probability": 0.05,
  "risk_level": "LOW",
  "confidence_score": 0.95
}
```

API documentation: http://localhost:8000/docs

##  Docker Deployment

```bash
# Build Docker image
docker build -t fraud-detection-api .

# Run container
docker run -p 8000:8000 fraud-detection-api
```

##  Monitoring

Evidently AI for data drift and performance monitoring:

```bash
python monitoring/generate_reports.py \
  --reference data/features/transactions_features.parquet \
  --current data/features/transactions_features.parquet \
  --output monitoring/reports
```

**Monitored Indicators:**
- Distribution of `amt`
- False negative rate
- Prediction confidence scores
- Data drift detection

Reports saved as HTML in `monitoring/reports/`:
- `data_drift_report.html` - Data drift analysis
- `data_quality_report.html` - Data quality metrics

## DVC Remote Storage (Optional)

```bash
# Add remote storage (S3, GCS, Azure, etc.)
dvc remote add -d myremote s3://mybucket/fraud-detection

# Push data to remote
dvc push

# Pull data from remote (on another machine)
dvc pull
```

##  MLflow Model Registry

Register models for production:

```python
# Register model
mlflow.register_model("runs:/<run_id>/model", "fraud_detection_model")

# Transition to production
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="fraud_detection_model",
    version=1,
    stage="Production"
)
```

**Model States:**
- `Staging` - Under testing
- `Production` - Live deployment
- `Archived` - Deprecated

##  Key Features

✅ **Data Versioning** - DVC tracks all data versions  
✅ **Reproducibility** - Complete pipeline reproducibility  
✅ **Data Lineage** - Track data transformations  
✅ **Validation** - Pandera ensures data quality  
✅ **Experiment Tracking** - MLflow logs all experiments  
✅ **Model Registry** - Version and manage models  
✅ **REST API** - FastAPI deployment  
✅ **Monitoring** - Evidently AI drift detection  
✅ **Containerization** - Docker support  

##  Expected Performance

With XGBoost on imbalanced fraud data:
- **Precision:** ~0.85-0.95
- **Recall:** ~0.75-0.85
- **F1-Score:** ~0.80-0.90
- **ROC-AUC:** ~0.90-0.98


### Modify Parameters

Edit `params.yaml`:

```yaml
train:
  test_size: 0.3        # Change train/test split
  use_smote: false      # Disable SMOTE

model:
  max_depth: 8          # Deeper trees
  n_estimators: 300     # More trees
```

Then rerun: `dvc repro`

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200, 300]
}

grid_search = GridSearchCV(XGBClassifier(), param_grid, cv=3)
grid_search.fit(X_train, y_train)
```

##  Notes

- Raw data (`fraudTest.csv`) is tracked with DVC (not in Git)
- Processed data outputs are reproducible via `dvc repro`
- All model versions linked to data versions
- MLflow tracks all metrics and artifacts

##  Contributing

1. Create feature branch
2. Make changes
3. Run validation: `dvc repro`
4. Commit: `git commit -m "feat: description"`
5. Push: `git push`

---

**Built with:** Python, XGBoost, MLflow, DVC, FastAPI, Pandera, Evidently AI, Docker
