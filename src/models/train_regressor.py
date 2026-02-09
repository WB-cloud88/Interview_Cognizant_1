"""Train time-to-failure regressor."""
import mlflow
from xgboost import XGBRegressor


def train_time_to_failure_regressor(X_train, y_train, config):
    """Train XGBoost regressor for days-to-failure prediction."""
    mlflow.log_param("regressor_type", "XGBRegressor")

    model = XGBRegressor(**config["model_params"]["regressor"])
    model.fit(X_train, y_train)

    return model
