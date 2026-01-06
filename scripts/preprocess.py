"""
Preprocessing: cleaning and initial transformations
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os


def preprocess_data(input_path: str, output_path: str, scaler_path: str = None):
    """
    Preprocess fraud detection data:
    - Normalize Amount
    - Transform Time
    - Handle missing values
    - Separate features/target
    """
    print(f"Loading raw data from {input_path}...")
    df = pd.read_csv(input_path)
    
    print(f"Initial shape: {df.shape}")
    
    # Drop unnecessary columns
    drop_cols = ['Unnamed: 0', 'cc_num', 'first', 'last', 'street', 
                 'city', 'state', 'zip', 'dob', 'trans_num', 'merchant']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    # Convert datetime
    if 'trans_date_trans_time' in df.columns:
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        df['hour'] = df['trans_date_trans_time'].dt.hour
        df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
        df['day_of_month'] = df['trans_date_trans_time'].dt.day
        df = df.drop(columns=['trans_date_trans_time'])
    
    # Encode categorical variables
    if 'category' in df.columns:
        df = pd.get_dummies(df, columns=['category'], prefix='cat')
    
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'M': 1, 'F': 0})
    
    if 'job' in df.columns:
        # Keep only top jobs, others as 'other'
        top_jobs = df['job'].value_counts().head(20).index
        df['job'] = df['job'].apply(lambda x: x if x in top_jobs else 'other')
        df = pd.get_dummies(df, columns=['job'], prefix='job')
    
    # Handle missing values
    df = df.fillna(df.median(numeric_only=True))
    
    # Normalize amount
    scaler = StandardScaler()
    if 'amt' in df.columns:
        df['amt_normalized'] = scaler.fit_transform(df[['amt']])
    
    # Save scaler
    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"✓ Scaler saved to {scaler_path}")
    
    # Move target to end
    if 'is_fraud' in df.columns:
        target = df.pop('is_fraud')
        df['is_fraud'] = target
    
    print(f"Final shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✓ Preprocessed data saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess fraud detection data")
    parser.add_argument("--input", type=str, required=True, help="Input CSV path")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--scaler", type=str, default="models/scaler.pkl", help="Scaler save path")
    args = parser.parse_args()
    
    preprocess_data(args.input, args.output, args.scaler)
