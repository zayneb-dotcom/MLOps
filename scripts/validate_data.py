"""
Data validation with Pandera
Guarantees data quality before training
"""
import argparse
import pandas as pd
import pandera as pa
from pandera import Column, Check, DataFrameSchema
import sys


# Validation schema
fraud_schema = DataFrameSchema({
    "amt": Column(float, Check.greater_than(0), nullable=False),
    "is_fraud": Column(int, Check.isin([0, 1]), nullable=False),
    "unix_time": Column(int, nullable=False),
    "lat": Column(float, nullable=False),
    "long": Column(float, nullable=False),
    "merch_lat": Column(float, nullable=False),
    "merch_long": Column(float, nullable=False),
    "city_pop": Column(int, Check.greater_than_or_equal_to(0), nullable=False),
})


def validate_data(input_path: str, output_path: str = None):
    """Validate fraud detection dataset"""
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    # Validate with Pandera
    try:
        validated_df = fraud_schema.validate(df, lazy=True)
        print("✓ Validation passed!")
        
        # Additional checks
        fraud_rate = df['is_fraud'].mean()
        print(f"Fraud rate: {fraud_rate:.2%}")
        
        if fraud_rate < 0.001 or fraud_rate > 0.5:
            print("⚠ Warning: Unusual fraud rate detected")
        
        # Check distributions
        print(f"\nAmount statistics:")
        print(df['amt'].describe())
        
        if output_path:
            validated_df.to_csv(output_path, index=False)
            print(f"✓ Validated data saved to {output_path}")
        
        return True
        
    except pa.errors.SchemaErrors as e:
        print("✗ Validation failed!")
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate fraud detection data")
    parser.add_argument("--input", type=str, required=True, help="Input CSV path")
    parser.add_argument("--output", type=str, help="Output validated CSV path")
    args = parser.parse_args()
    
    validate_data(args.input, args.output)
