"""Load raw data files."""
import pandas as pd


def load_sensor_readings(path: str) -> pd.DataFrame:
    """Load sensor readings CSV with proper dtypes."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def load_maintenance_log(path: str) -> pd.DataFrame:
    """Load maintenance event log."""
    df = pd.read_csv(
        path, parse_dates=["install_date", "last_inspection_date", "event_date"]
    )
    return df
