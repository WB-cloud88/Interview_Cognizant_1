#!/usr/bin/env python
"""
Daily batch scoring pipeline.
Schedule: Daily (1AM)
Purpose: Generate risk report for ops dashboard
"""
import sys

sys.path.append(".")

import pandas as pd
import joblib
from datetime import datetime

from src.data.load_data import load_sensor_readings, load_maintenance_log
from src.data.merge_sources import merge_sensor_and_maintenance
from src.features.build_features import build_all_features


def main():
    print("=" * 50)
    print("SCORING PIPELINE")
    print("=" * 50)

    # Load latest sensor data
    print("\n[1/5] Loading latest sensor data...")
    sensor_df = load_sensor_readings("data/raw/sensor_readings.csv")
    maint_log = load_maintenance_log("data/raw/maintenance_log.csv")

    # Prepare features
    print("\n[2/5] Engineering features...")
    merged = merge_sensor_and_maintenance(sensor_df, maint_log)
    features_df = build_all_features(merged)

    # Load model
    print("\n[3/5] Loading trained model...")
    model = joblib.load("data/models/classifier.pkl")

    # Score
    print("\n[4/5] Generating predictions...")
    feature_cols = [
        c
        for c in features_df.columns
        if c
        not in [
            "failure_30d",
            "days_to_failure",
            "equipment_id",
            "timestamp",
            "install_date",
            "last_inspection_date",
        ]
    ]

    X = features_df[feature_cols]
    failure_prob = model.predict_proba(X)[:, 1]

    # Create risk report
    risk_report = pd.DataFrame(
        {
            "equipment_id": features_df["equipment_id"],
            "timestamp": features_df["timestamp"],
            "failure_probability": failure_prob,
            "risk_category": pd.cut(
                failure_prob,
                bins=[0, 0.3, 0.7, 1.0],
                labels=["Low", "Medium", "High"],
            ),
        }
    )

    # Save
    print("\n[5/5] Saving risk report...")
    output_path = (
        f'data/processed/risk_report_{datetime.now().strftime("%Y%m%d")}.csv'
    )
    risk_report.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")

    # Summary
    print("\n  Risk Distribution:")
    print(risk_report["risk_category"].value_counts().to_string())

    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()
