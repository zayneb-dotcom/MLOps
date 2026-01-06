"""
Model training with MLflow tracking
XGBoost Classifier for fraud detection
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.xgboost
import joblib
import os


def train_model(
    input_path: str,
    model_path: str,
    experiment_name: str = "fraud_detection",
    use_smote: bool = True,
    test_size: float = 0.2
):
    """
    Train XGBoost model with MLflow tracking
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
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'scale_pos_weight': (len(y_train) - y_train.sum()) / y_train.sum(),
            'random_state': 42
        }
        
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("use_smote", use_smote)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("n_features", X.shape[1])
        
        # Train model
        print("Training XGBoost model...")
        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
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
        print(f"Model Performance:")
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
        mlflow.xgboost.log_model(model, "model")
        mlflow.log_artifact(model_path)
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv('feature_importance.csv', index=False)
        mlflow.log_artifact('feature_importance.csv')
        
        print("\n✓ MLflow tracking completed")
        print(f"Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument("--input", type=str, required=True, help="Input features path")
    parser.add_argument("--model", type=str, default="models/xgboost_fraud_model.pkl", 
                        help="Model save path")
    parser.add_argument("--experiment", type=str, default="fraud_detection", 
                        help="MLflow experiment name")
    parser.add_argument("--no-smote", action="store_true", help="Disable SMOTE")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size")
    
    args = parser.parse_args()
    
    train_model(
        args.input,
        args.model,
        args.experiment,
        use_smote=not args.no_smote,
        test_size=args.test_size
    )
