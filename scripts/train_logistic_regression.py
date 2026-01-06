"""
Logistic Regression baseline model with MLflow tracking
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import joblib
import os


def train_logistic_regression(
    input_path: str,
    model_path: str,
    experiment_name: str = "fraud_detection_lr",
    use_smote: bool = True,
    test_size: float = 0.2
):
    """
    Train Logistic Regression baseline model with MLflow tracking
    """
    print(f"Loading features from {input_path}...")
    
    if input_path.endswith('.parquet'):
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
    
    print(f"Dataset shape: {df.shape}")
    
    # Separate features and target
    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']
    
    print(f"Features: {X.shape[1]}")
    print(f"Fraud rate: {y.mean():.2%}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Handle imbalance with SMOTE
    if use_smote:
        print("Applying SMOTE for class balancing...")
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE - Fraud rate: {y_train.mean():.2%}")
    
    # Set up MLflow
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        # Model parameters
        params = {
            'penalty': 'l2',
            'C': 1.0,
            'solver': 'lbfgs',
            'max_iter': 1000,
            'class_weight': 'balanced',
            'random_state': 42
        }
        
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("use_smote", use_smote)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("model_type", "LogisticRegression")
        
        # Train model
        print("Training Logistic Regression baseline model...")
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Log metrics
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        # Print metrics
        print(f"\n{'='*50}")
        print(f"Logistic Regression Baseline Performance:")
        print(f"{'='*50}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        print(f"\n✓ Model saved to {model_path}")
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(model_path)
        
        print("\n✓ MLflow tracking completed")
        print(f"Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Logistic Regression baseline model")
    parser.add_argument("--input", type=str, required=True, help="Input features path")
    parser.add_argument("--model", type=str, default="models/logistic_regression_fraud_model.pkl", 
                        help="Model save path")
    parser.add_argument("--experiment", type=str, default="fraud_detection_lr", 
                        help="MLflow experiment name")
    parser.add_argument("--no-smote", action="store_true", help="Disable SMOTE")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size")
    
    args = parser.parse_args()
    
    train_logistic_regression(
        args.input,
        args.model,
        args.experiment,
        use_smote=not args.no_smote,
        test_size=args.test_size
    )
