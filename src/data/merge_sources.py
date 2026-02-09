"""Merge sensor and maintenance data."""
import pandas as pd


def merge_sensor_and_maintenance(sensor_df, maint_df) -> pd.DataFrame:
    """
    Merge sensor readings with equipment metadata.
    Adds: install_date, last_inspection_date per equipment.
    Calculates: equipment_age_years, days_since_maintenance
    """
    # Get unique equipment records (latest metadata per equipment)
    equipment_meta = (
        maint_df.groupby("equipment_id")
        .agg({"install_date": "first", "last_inspection_date": "max"})
        .reset_index()
    )

    # Merge
    merged = sensor_df.merge(equipment_meta, on="equipment_id", how="left")

    # Calculate temporal features
    merged["equipment_age_years"] = (
        merged["timestamp"] - merged["install_date"]
    ).dt.days / 365.25
    merged["days_since_maintenance"] = (
        merged["timestamp"] - merged["last_inspection_date"]
    ).dt.days

    return merged
