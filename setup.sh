#!/bin/bash
# Quick setup script for MLOps Fraud Detection Pipeline (Linux/Mac)

echo "========================================"
echo "MLOps Fraud Detection Pipeline Setup"
echo "========================================"
echo ""

echo "[1/6] Installing Python dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi
echo ""

echo "[2/6] Initializing Git repository..."
git init
git add .
git commit -m "Initial commit - Fraud detection MLOps pipeline"
echo ""

echo "[3/6] Initializing DVC..."
dvc init
git add .dvc .gitignore
git commit -m "Initialize DVC"
echo ""

echo "[4/6] Tracking raw data with DVC..."
dvc add data/raw/fraudTest.csv
git add data/raw/fraudTest.csv.dvc data/raw/.gitignore
git commit -m "Track raw fraud dataset with DVC"
echo ""

echo "[5/6] Running DVC pipeline (validation, preprocessing, feature engineering, training)..."
dvc repro
if [ $? -ne 0 ]; then
    echo "WARNING: Pipeline execution encountered issues"
    echo "You may need to check data/raw/fraudTest.csv"
fi
echo ""

echo "[6/6] Setup complete!"
echo ""
echo "========================================"
echo "Next Steps:"
echo "========================================"
echo "1. View MLflow experiments: mlflow ui"
echo "2. Start API server: uvicorn api.main:app --reload"
echo "3. API docs: http://localhost:8000/docs"
echo "4. Generate monitoring reports: python monitoring/generate_reports.py --reference data/features/transactions_features.parquet --current data/features/transactions_features.parquet"
echo ""
echo "For detailed instructions, see README.md"
echo "========================================"
