"""Create target variables from failure events."""
import pandas as pd


def create_failure_labels(df, maint_log, horizon_days=30) -> pd.DataFrame:
    """
    Create binary classification target (failure_30d) and regression target (days_to_failure).

    Business logic:
    - failure_30d = 1 if equipment fails within next 30 days, else 0
    - days_to_failure = days until next failure (-1 if no failure in horizon)
    """
    # Extract failure events
    failures = maint_log[maint_log["event_type"] == "FAILURE"][
        ["equipment_id", "event_date"]
    ].copy()

    # Initialize labels
    df["failure_30d"] = 0
    df["days_to_failure"] = -1

    # For each failure, label preceding readings
    for _, failure in failures.iterrows():
        equipment = failure["equipment_id"]
        failure_date = failure["event_date"]

        # Readings within horizon before failure
        mask = (
            (df["equipment_id"] == equipment)
            & (df["timestamp"] < failure_date)
            & (df["timestamp"] >= failure_date - pd.Timedelta(days=horizon_days))
        )

        df.loc[mask, "failure_30d"] = 1
        df.loc[mask, "days_to_failure"] = (
            failure_date - df.loc[mask, "timestamp"]
        ).dt.days

    return df
