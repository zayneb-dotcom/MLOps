"""
Model and data monitoring with Evidently AI
Detect data drift and performance degradation
"""
import argparse
import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_set import DataDriftPreset, DataQualityPreset
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    ColumnDriftMetric,
    ClassificationQualityMetric
)
import json
import os


def generate_monitoring_report(
    reference_path: str,
    current_path: str,
    output_dir: str = "monitoring/reports"
):
    """
    Generate monitoring report comparing reference and current data
    """
    print(f"Loading reference data from {reference_path}...")
    if reference_path.endswith('.parquet'):
        reference_df = pd.read_parquet(reference_path)
    else:
        reference_df = pd.read_csv(reference_path)
    
    print(f"Loading current data from {current_path}...")
    if current_path.endswith('.parquet'):
        current_df = pd.read_parquet(current_path)
    else:
        current_df = pd.read_csv(current_path)
    
    print(f"Reference shape: {reference_df.shape}")
    print(f"Current shape: {current_df.shape}")
    
    # Column mapping
    column_mapping = ColumnMapping(
        target='is_fraud',
        numerical_features=[
            'amt', 'lat', 'long', 'merch_lat', 'merch_long',
            'city_pop', 'unix_time', 'distance'
        ]
    )
    
    # Data Drift Report
    print("\nGenerating Data Drift Report...")
    drift_report = Report(metrics=[
        DataDriftPreset(),
        DatasetDriftMetric(),
        ColumnDriftMetric(column_name='amt'),
        DatasetMissingValuesMetric(),
    ])
    
    drift_report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping
    )
    
    # Save reports
    os.makedirs(output_dir, exist_ok=True)
    
    drift_html_path = os.path.join(output_dir, "data_drift_report.html")
    drift_report.save_html(drift_html_path)
    print(f"✓ Drift report saved to {drift_html_path}")
    
    # Extract drift metrics
    drift_json_path = os.path.join(output_dir, "drift_metrics.json")
    drift_report.save_json(drift_json_path)
    print(f"✓ Drift metrics saved to {drift_json_path}")
    
    # Data Quality Report
    print("\nGenerating Data Quality Report...")
    quality_report = Report(metrics=[
        DataQualityPreset(),
    ])
    
    quality_report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping
    )
    
    quality_html_path = os.path.join(output_dir, "data_quality_report.html")
    quality_report.save_html(quality_html_path)
    print(f"✓ Quality report saved to {quality_html_path}")
    
    # Summary
    print("\n" + "="*50)
    print("Monitoring Summary")
    print("="*50)
    print(f"Reference fraud rate: {reference_df['is_fraud'].mean():.2%}")
    print(f"Current fraud rate: {current_df['is_fraud'].mean():.2%}")
    
    if 'amt' in reference_df.columns and 'amt' in current_df.columns:
        print(f"\nAmount statistics:")
        print(f"Reference mean: ${reference_df['amt'].mean():.2f}")
        print(f"Current mean: ${current_df['amt'].mean():.2f}")
    
    print(f"\n✓ All monitoring reports generated in {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate monitoring reports")
    parser.add_argument("--reference", type=str, required=True, 
                        help="Reference dataset path")
    parser.add_argument("--current", type=str, required=True, 
                        help="Current dataset path")
    parser.add_argument("--output", type=str, default="monitoring/reports",
                        help="Output directory for reports")
    
    args = parser.parse_args()
    
    generate_monitoring_report(args.reference, args.current, args.output)
