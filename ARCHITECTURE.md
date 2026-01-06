# MLOps Architecture Diagram

## Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FRAUD DETECTION MLOps                        │
│                                                                      │
│  ┌────────────────┐     ┌────────────────┐     ┌─────────────────┐ │
│  │   Data Layer   │────▶│ Pipeline Layer │────▶│  Serving Layer  │ │
│  └────────────────┘     └────────────────┘     └─────────────────┘ │
│          │                      │                       │           │
│          │                      │                       │           │
│  ┌───────▼──────┐      ┌────────▼────────┐     ┌───────▼────────┐ │
│  │ DVC Tracking │      │ MLflow Tracking │     │ FastAPI Server │ │
│  └──────────────┘      └─────────────────┘     └────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                        ┌───────────▼───────────┐
                        │  Monitoring Layer     │
                        │  (Evidently AI)       │
                        └───────────────────────┘
```

## Data Flow

```
Raw Data (fraudTest.csv)
    │
    │ DVC Tracks
    ▼
┌──────────────────────┐
│  1. Validation       │
│  (Pandera)           │
│  • Check amt > 0     │
│  • Check fraud ∈ {0,1}│
│  • No missing values │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  2. Preprocessing    │
│  • Drop columns      │
│  • Encode categories │
│  • Normalize amounts │
│  • Extract datetime  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  3. Feature Eng      │
│  • Distance calc     │
│  • Time features     │
│  • Amount transforms │
│  • Population bins   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  4. Training         │
│  • SMOTE balancing   │
│  • XGBoost training  │
│  • MLflow logging    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Trained Model       │
│  xgboost_model.pkl   │
└──────────────────────┘
```

## MLflow Experiment Tracking

```
┌─────────────────────────────────────────┐
│         MLflow Tracking Server          │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │  Experiment: fraud_detection      │ │
│  │                                   │ │
│  │  Run 1 (2024-01-01)              │ │
│  │    ├─ params: {max_depth: 6}     │ │
│  │    ├─ metrics: {f1: 0.85}        │ │
│  │    └─ model: xgboost_v1.pkl      │ │
│  │                                   │ │
│  │  Run 2 (2024-01-02)              │ │
│  │    ├─ params: {max_depth: 8}     │ │
│  │    ├─ metrics: {f1: 0.87}        │ │
│  │    └─ model: xgboost_v2.pkl      │ │
│  │                                   │ │
│  │  Run 3 (2024-01-03) ★ BEST       │ │
│  │    ├─ params: {max_depth: 7}     │ │
│  │    ├─ metrics: {f1: 0.90}        │ │
│  │    └─ model: xgboost_v3.pkl      │ │
│  └───────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

## Model Registry Workflow

```
┌──────────────┐
│ Development  │
│   Training   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Staging    │ ◀─── Testing & Validation
│   (v1.0.0)   │
└──────┬───────┘
       │ ✓ Tests Pass
       ▼
┌──────────────┐
│  Production  │ ◀─── Live Serving
│   (v1.0.0)   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Archived   │ ◀─── Deprecated
│   (v0.9.0)   │
└──────────────┘
```

## API Architecture

```
┌─────────────────────────────────────────┐
│         Client Application              │
└────────────────┬────────────────────────┘
                 │ HTTP Request
                 ▼
┌─────────────────────────────────────────┐
│         FastAPI Server                  │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │ Endpoints:                        │ │
│  │  • GET  /health                   │ │
│  │  • POST /predict                  │ │
│  │  • POST /predict/batch            │ │
│  └───────────────┬───────────────────┘ │
│                  │                     │
│  ┌───────────────▼───────────────────┐ │
│  │ Load Model                        │ │
│  │  • xgboost_fraud_model.pkl        │ │
│  │  • scaler.pkl                     │ │
│  └───────────────┬───────────────────┘ │
│                  │                     │
│  ┌───────────────▼───────────────────┐ │
│  │ Feature Engineering               │ │
│  │  • Calculate distance             │ │
│  │  • Add time features              │ │
│  │  • Normalize amounts              │ │
│  └───────────────┬───────────────────┘ │
│                  │                     │
│  ┌───────────────▼───────────────────┐ │
│  │ Prediction                        │ │
│  │  • is_fraud: 0 or 1               │ │
│  │  • probability: 0.0 - 1.0         │ │
│  │  • risk_level: LOW/MED/HIGH       │ │
│  └───────────────────────────────────┘ │
└─────────────────────────────────────────┘
                 │ JSON Response
                 ▼
┌─────────────────────────────────────────┐
│         Client receives result          │
└─────────────────────────────────────────┘
```

## Monitoring Flow

```
┌─────────────────┐      ┌─────────────────┐
│ Reference Data  │      │  Current Data   │
│  (Training)     │      │  (Production)   │
└────────┬────────┘      └────────┬────────┘
         │                        │
         └────────────┬───────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │   Evidently AI         │
         │                        │
         │  ┌──────────────────┐  │
         │  │ Data Drift       │  │
         │  │ • Features       │  │
         │  │ • Target         │  │
         │  └──────────────────┘  │
         │                        │
         │  ┌──────────────────┐  │
         │  │ Data Quality     │  │
         │  │ • Missing values │  │
         │  │ • Outliers       │  │
         │  └──────────────────┘  │
         │                        │
         │  ┌──────────────────┐  │
         │  │ Model Performance│  │
         │  │ • Accuracy drop  │  │
         │  │ • Bias detection │  │
         │  └──────────────────┘  │
         └────────────┬───────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │   HTML Reports         │
         │   • Visualizations     │
         │   • Alerts             │
         │   • Recommendations    │
         └────────────────────────┘
```

## Docker Deployment

```
┌─────────────────────────────────────────┐
│          Docker Container               │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │  Base Image: python:3.9-slim      │ │
│  └───────────────────────────────────┘ │
│                 │                      │
│  ┌──────────────▼──────────────────┐  │
│  │  Install Dependencies           │  │
│  │  • pandas, numpy, scikit-learn  │  │
│  │  • xgboost, mlflow              │  │
│  │  • fastapi, uvicorn             │  │
│  └──────────────┬──────────────────┘  │
│                 │                      │
│  ┌──────────────▼──────────────────┐  │
│  │  Copy Application               │  │
│  │  • api/main.py                  │  │
│  │  • models/                      │  │
│  └──────────────┬──────────────────┘  │
│                 │                      │
│  ┌──────────────▼──────────────────┐  │
│  │  Expose Port 8000               │  │
│  └──────────────┬──────────────────┘  │
│                 │                      │
│  ┌──────────────▼──────────────────┐  │
│  │  Start FastAPI Server           │  │
│  │  uvicorn api.main:app           │  │
│  └─────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

## DVC Pipeline Graph

```
┌─────────────────┐
│ fraudTest.csv   │ (Raw Data)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  validate       │ (Stage 1)
│  deps:          │
│   - raw data    │
│  outs:          │
│   - validated   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  preprocess     │ (Stage 2)
│  deps:          │
│   - validated   │
│  outs:          │
│   - clean.csv   │
│   - scaler.pkl  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ feature_eng     │ (Stage 3)
│  deps:          │
│   - clean.csv   │
│  outs:          │
│   - features    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  train          │ (Stage 4)
│  deps:          │
│   - features    │
│  outs:          │
│   - model.pkl   │
│  metrics:       │
│   - mlruns/     │
└─────────────────┘
```

## Component Interaction

```
┌──────────┐
│   User   │
└────┬─────┘
     │
     ▼
┌────────────────────────────────────────────────────────┐
│                    MLOps Platform                      │
│                                                        │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐          │
│  │   DVC    │  │  MLflow   │  │ FastAPI  │          │
│  │          │  │           │  │          │          │
│  │ Version  │  │  Track    │  │  Serve   │          │
│  │  Data    │  │ Experiments│  │  Model   │          │
│  └────┬─────┘  └─────┬─────┘  └────┬─────┘          │
│       │              │              │                 │
│       └──────────────┼──────────────┘                 │
│                      │                                │
│              ┌───────▼────────┐                       │
│              │   Monitoring   │                       │
│              │  (Evidently)   │                       │
│              └────────────────┘                       │
└────────────────────────────────────────────────────────┘
```

## Full MLOps Lifecycle

```
1. DATA
   └─ DVC tracks versions
      └─ Reproducible history

2. VALIDATE
   └─ Pandera checks quality
      └─ Fail fast on bad data

3. TRANSFORM
   └─ Preprocessing pipeline
      └─ Consistent features

4. TRAIN
   └─ XGBoost + SMOTE
      └─ MLflow logs everything

5. EVALUATE
   └─ Metrics (P, R, F1, AUC)
      └─ Choose best model

6. REGISTER
   └─ MLflow Model Registry
      └─ Version management

7. DEPLOY
   └─ FastAPI + Docker
      └─ Production serving

8. MONITOR
   └─ Evidently AI
      └─ Detect drift & issues

9. RETRAIN (Loop back to step 1)
   └─ When drift detected
      └─ Continuous improvement
```

---

**Legend:**
- `│` : Flow/Connection
- `▼` : Direction of flow
- `┌─┐` : Component boundary
- `★` : Best/Selected item
- `✓` : Validation passed
