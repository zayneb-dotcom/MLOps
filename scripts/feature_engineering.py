"""
Feature Engineering: create advanced features
"""
import argparse
import pandas as pd
import numpy as np
import os


def engineer_features(input_path: str, output_path: str):
    """
    Create engineered features:
    - Temporal features
    - Distance features
    - Statistical aggregations
    """
    print(f"Loading preprocessed data from {input_path}...")
    df = pd.read_csv(input_path)
    
    print(f"Input shape: {df.shape}")
    
    # Distance between customer and merchant
    if all(col in df.columns for col in ['lat', 'long', 'merch_lat', 'merch_long']):
        df['distance'] = np.sqrt(
            (df['lat'] - df['merch_lat'])**2 + 
            (df['long'] - df['merch_long'])**2
        )
        print("✓ Created distance feature")
    
    # Time-based features
    if 'hour' in df.columns:
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int) if 'day_of_week' in df.columns else 0
        print("✓ Created temporal features")
    
    # Amount features
    if 'amt' in df.columns:
        df['amt_log'] = np.log1p(df['amt'])
        df['amt_squared'] = df['amt'] ** 2
        
        # Amount bins
        df['amt_bin'] = pd.cut(df['amt'], bins=[0, 10, 50, 100, 500, np.inf], 
                               labels=[0, 1, 2, 3, 4]).astype(int)
        print("✓ Created amount features")
    
    # Population features
    if 'city_pop' in df.columns:
        df['city_pop_log'] = np.log1p(df['city_pop'])
        df['is_high_pop'] = (df['city_pop'] > df['city_pop'].median()).astype(int)
        print("✓ Created population features")
    
    print(f"Output shape: {df.shape}")
    print(f"New features count: {df.shape[1] - pd.read_csv(input_path).shape[1]}")
    
    # Save engineered features
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as parquet for better performance
    if output_path.endswith('.parquet'):
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)
    
    print(f"✓ Feature-engineered data saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature engineering for fraud detection")
    parser.add_argument("--input", type=str, required=True, help="Input CSV path")
    parser.add_argument("--output", type=str, required=True, help="Output path (CSV or Parquet)")
    args = parser.parse_args()
    
    engineer_features(args.input, args.output)
