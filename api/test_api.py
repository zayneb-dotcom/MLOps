"""
Test script to verify API functionality
"""
import requests
import json

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{API_URL}/health")
    print(f"Health check: {response.json()}")
    assert response.status_code == 200

def test_predict():
    """Test single prediction"""
    transaction = {
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
    
    response = requests.post(f"{API_URL}/predict", json=transaction)
    result = response.json()
    
    print(f"\nPrediction result:")
    print(json.dumps(result, indent=2))
    
    assert response.status_code == 200
    assert "is_fraud" in result
    assert "fraud_probability" in result

def test_batch_predict():
    """Test batch prediction"""
    transactions = [
        {
            "amt": 50.0,
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
        },
        {
            "amt": 5000.0,
            "lat": 40.7128,
            "long": -74.0060,
            "merch_lat": 35.0,
            "merch_long": -120.0,
            "city_pop": 100,
            "unix_time": 1609459200,
            "hour": 2,
            "day_of_week": 6,
            "day_of_month": 15,
            "gender": 0
        }
    ]
    
    response = requests.post(f"{API_URL}/predict/batch", json=transactions)
    result = response.json()
    
    print(f"\nBatch prediction result:")
    print(json.dumps(result, indent=2))
    
    assert response.status_code == 200
    assert "predictions" in result
    assert result["count"] == 2

if __name__ == "__main__":
    print("Testing Fraud Detection API...")
    print("="*50)
    
    try:
        test_health()
        test_predict()
        test_batch_predict()
        print("\n✓ All tests passed!")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
