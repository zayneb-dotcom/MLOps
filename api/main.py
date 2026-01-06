"""
FastAPI deployment for fraud detection model
Real-time prediction endpoint
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
import mlflow
import os
from typing import Dict, List


app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection using XGBoost",
    version="1.0.0"
)


# Load model
MODEL_PATH = os.getenv("MODEL_PATH", "models/xgboost_fraud_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    print(f"✓ Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"⚠ Could not load model: {e}")
    model = None
    scaler = None


class Transaction(BaseModel):
    """Transaction input schema"""
    amt: float = Field(..., gt=0, description="Transaction amount")
    lat: float = Field(..., description="Customer latitude")
    long: float = Field(..., description="Customer longitude")
    merch_lat: float = Field(..., description="Merchant latitude")
    merch_long: float = Field(..., description="Merchant longitude")
    city_pop: int = Field(..., ge=0, description="City population")
    unix_time: int = Field(..., description="Transaction timestamp")
    hour: int = Field(..., ge=0, le=23, description="Hour of transaction")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week")
    day_of_month: int = Field(..., ge=1, le=31, description="Day of month")
    gender: int = Field(..., description="Gender (0=F, 1=M)")
    
    class Config:
        schema_extra = {
            "example": {
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
            }
        }


class PredictionResponse(BaseModel):
    """Prediction output schema"""
    is_fraud: int
    fraud_probability: float
    risk_level: str
    confidence_score: float


@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }


@app.get("/health")
def health():
    """Detailed health check"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_path": MODEL_PATH,
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    """
    Predict fraud for a single transaction
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to dataframe
        data = pd.DataFrame([transaction.dict()])
        
        # Add engineered features
        data['distance'] = np.sqrt(
            (data['lat'] - data['merch_lat'])**2 + 
            (data['long'] - data['merch_long'])**2
        )
        data['is_night'] = ((data['hour'] >= 22) | (data['hour'] <= 6)).astype(int)
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        data['amt_log'] = np.log1p(data['amt'])
        data['amt_squared'] = data['amt'] ** 2
        data['city_pop_log'] = np.log1p(data['city_pop'])
        data['is_high_pop'] = (data['city_pop'] > 100000).astype(int)
        
        # Normalize amount if scaler available
        if scaler is not None and 'amt' in data.columns:
            data['amt_normalized'] = scaler.transform(data[['amt']])
        
        # Predict
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0, 1]
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "LOW"
        elif probability < 0.7:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        return PredictionResponse(
            is_fraud=int(prediction),
            fraud_probability=float(probability),
            risk_level=risk_level,
            confidence_score=float(max(probability, 1 - probability))
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch")
def predict_batch(transactions: List[Transaction]):
    """
    Predict fraud for multiple transactions
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        for txn in transactions:
            result = predict(txn)
            results.append(result.dict())
        
        return {
            "predictions": results,
            "count": len(results),
            "fraud_count": sum(1 for r in results if r['is_fraud'] == 1)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
