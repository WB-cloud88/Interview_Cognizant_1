#!/usr/bin/env python
"""
Batch model training pipeline.
Schedule: Weekly (Sunday 3AM)
Purpose: Retrain models on latest data
"""
import sys

sys.path.append(".")

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
import mlflow

from src.models.train_classifier import train_cost_weighted_xgboost
from src.models.evaluate import evaluate_classifier
from src.utils.config_loader import load_config
from src.utils.mlflow_utils import setup_mlflow


def main():
    print("=" * 50)
    print("MODEL TRAINING PIPELINE")
    print("=" * 50)

    config = load_config()
    setup_mlflow(config)

    # Load features
    print("\n[1/4] Loading processed features...")
    df = pd.read_parquet("data/processed/features.parquet")
    print(f"  Shape: {df.shape}")

    # Prepare data
    print("\n[2/4] Preparing train/test split...")
    feature_cols = [
        c
        for c in df.columns
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

    X = df[feature_cols]
    y = df["failure_30d"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Train failure rate: {y_train.mean():.2%}")

    # Train classifier
    print("\n[3/4] Training classifier...")
    with mlflow.start_run(run_name="xgboost_cost_weighted"):
        classifier = train_cost_weighted_xgboost(X_train, y_train, config)

        # Evaluate
        y_pred = classifier.predict(X_test)
        metrics = evaluate_classifier(y_test, y_pred, config)

        # Log metrics
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)
            if metric.endswith("_gbp"):
                print(f"  {metric}: \u00a3{value:,.0f}")
            else:
                print(f"  {metric}: {value:.4f}")

        # Save model
        model_path = "data/models/classifier.pkl"
        joblib.dump(classifier, model_path)
        mlflow.log_artifact(model_path)
        print(f"\n  Model saved to: {model_path}")

    print("\n[4/4] Training complete!")
    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()
