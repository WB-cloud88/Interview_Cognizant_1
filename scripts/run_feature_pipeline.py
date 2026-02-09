#!/usr/bin/env python
"""
Batch feature engineering pipeline.
Schedule: Weekly (Sunday 2AM)
Purpose: Prepare training dataset
"""
import sys

sys.path.append(".")

from src.data.load_data import load_sensor_readings, load_maintenance_log
from src.data.merge_sources import merge_sensor_and_maintenance
from src.features.target_engineering import create_failure_labels
from src.features.build_features import build_all_features
from src.utils.config_loader import load_config


def main():
    print("=" * 50)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 50)

    config = load_config()

    # Load data
    print("\n[1/5] Loading data...")
    sensor_df = load_sensor_readings("data/raw/sensor_readings.csv")
    maint_log = load_maintenance_log("data/raw/maintenance_log.csv")
    print(f"  Sensor readings: {sensor_df.shape}")
    print(f"  Maintenance log: {maint_log.shape}")

    # Merge
    print("\n[2/5] Merging data sources...")
    merged = merge_sensor_and_maintenance(sensor_df, maint_log)
    print(f"  Merged shape: {merged.shape}")

    # Create targets
    print("\n[3/5] Creating target variables...")
    labeled = create_failure_labels(merged, maint_log)
    print(f"  Failure rate: {labeled['failure_30d'].mean():.2%}")

    # Build features
    print("\n[4/5] Engineering features...")
    features = build_all_features(labeled)
    print(f"  Final features: {features.shape}")

    # Save
    print("\n[5/5] Saving processed data...")
    output_path = "data/processed/features.parquet"
    features.to_parquet(output_path, index=False)
    print(f"  Saved to: {output_path}")

    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()
