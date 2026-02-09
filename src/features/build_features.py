"""Domain-driven feature engineering for transformer failure prediction."""
import pandas as pd
import numpy as np


def create_temporal_features(df) -> pd.DataFrame:
    """Equipment age and maintenance recency already calculated in merge_sources."""
    return df


def create_temperature_features(df) -> pd.DataFrame:
    """
    Temperature features based on ops insight:
    'Oil temperature differentials and trends indicate cooling/insulation issues'
    """
    df = df.copy()

    # Temperature differentials
    df["oil_temp_differential"] = (
        df["oil_temp_top_celsius"] - df["oil_temp_bottom_celsius"]
    )
    df["temp_stress"] = df["winding_temp_celsius"] - df["ambient_temp_celsius"]

    # Rolling statistics (14-day window per equipment)
    df["oil_temp_14d_mean"] = df.groupby("equipment_id")[
        "oil_temp_top_celsius"
    ].transform(lambda x: x.rolling(14, min_periods=1).mean())
    df["oil_temp_14d_std"] = df.groupby("equipment_id")[
        "oil_temp_top_celsius"
    ].transform(lambda x: x.rolling(14, min_periods=1).std())

    # Trend (7-day change)
    df["oil_temp_trend"] = df.groupby("equipment_id")[
        "oil_temp_top_celsius"
    ].transform(lambda x: x.diff(7))

    return df


def create_load_features(df) -> pd.DataFrame:
    """Load pattern features. High sustained load correlates with failures."""
    df = df.copy()

    # Rolling load statistics
    df["load_14d_max"] = df.groupby("equipment_id")["load_mva"].transform(
        lambda x: x.rolling(14, min_periods=1).max()
    )
    df["load_14d_mean"] = df.groupby("equipment_id")["load_mva"].transform(
        lambda x: x.rolling(14, min_periods=1).mean()
    )

    # High load cycles (count of days >75 MVA in last 14 days)
    df["high_load_cycles"] = df.groupby("equipment_id")["load_mva"].transform(
        lambda x: (x > 75).rolling(14, min_periods=1).sum()
    )

    return df


def create_interaction_features(df) -> pd.DataFrame:
    """Interaction terms: age x stress."""
    df = df.copy()
    df["age_load_interaction"] = df["equipment_age_years"] * df["load_14d_max"]
    df["temp_age_interaction"] = df["oil_temp_14d_mean"] * df["equipment_age_years"]
    return df


def build_all_features(df) -> pd.DataFrame:
    """Orchestrate all feature engineering."""
    df = create_temporal_features(df)
    df = create_temperature_features(df)
    df = create_load_features(df)
    df = create_interaction_features(df)

    # Drop rows with NaN from rolling windows
    df = df.dropna()

    return df
