"""
Create sample dataset for CI/CD testing
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create sample fraud detection dataset
np.random.seed(42)
n_samples = 1000

# Generate sample data
data = {
    'trans_date_trans_time': [
        (datetime.now() - timedelta(days=np.random.randint(0, 365))).strftime('%Y-%m-%d %H:%M:%S')
        for _ in range(n_samples)
    ],
    'amt': np.random.lognormal(4, 1, n_samples),
    'lat': np.random.uniform(25, 50, n_samples),
    'long': np.random.uniform(-125, -65, n_samples),
    'merch_lat': np.random.uniform(25, 50, n_samples),
    'merch_long': np.random.uniform(-125, -65, n_samples),
    'city_pop': np.random.randint(1000, 10000000, n_samples),
    'unix_time': [int((datetime.now() - timedelta(days=np.random.randint(0, 365))).timestamp()) for _ in range(n_samples)],
    'gender': np.random.choice(['M', 'F'], n_samples),
    'category': np.random.choice(['gas_transport', 'grocery_pos', 'shopping_net', 'entertainment', 'food_dining'], n_samples),
    'job': np.random.choice(['Engineer', 'Teacher', 'Doctor', 'Sales', 'Manager'], n_samples),
    'is_fraud': np.random.choice([0, 1], n_samples, p=[0.98, 0.02])
}

df = pd.DataFrame(data)

# Save to test data folder
import os
os.makedirs('data/test', exist_ok=True)
df.to_csv('data/test/sample_fraud_test.csv', index=False)
print(f"✓ Created sample dataset with {len(df)} records")
print(f"  Fraud rate: {df['is_fraud'].mean():.2%}")
