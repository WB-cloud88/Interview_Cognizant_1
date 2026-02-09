"""Train cost-weighted failure classifier."""
import mlflow
import numpy as np
from xgboost import XGBClassifier


def calculate_cost_ratio(config) -> float:
    """Calculate cost ratio from business params."""
    failure_cost = config["business_params"]["failure_cost_gbp"]
    alarm_cost = config["business_params"]["false_alarm_cost_gbp"]
    return failure_cost / alarm_cost


def calculate_beta(cost_ratio) -> float:
    """Calculate F-beta parameter from cost ratio."""
    return np.sqrt(cost_ratio)


def train_cost_weighted_xgboost(X_train, y_train, config) -> XGBClassifier:
    """
    Train XGBoost classifier optimized for business cost.
    Uses scale_pos_weight for class imbalance + cost weighting.
    Logs to MLflow.
    """
    cost_ratio = calculate_cost_ratio(config)
    beta = calculate_beta(cost_ratio)

    # Log params
    mlflow.log_param("model_type", "XGBClassifier")
    mlflow.log_param("cost_ratio", f"{cost_ratio:.1f}")
    mlflow.log_param("beta", f"{beta:.2f}")
    mlflow.log_param(
        "n_estimators", config["model_params"]["classifier"]["n_estimators"]
    )
    mlflow.log_param("max_depth", config["model_params"]["classifier"]["max_depth"])
    mlflow.log_param(
        "learning_rate", config["model_params"]["classifier"]["learning_rate"]
    )

    # Train model
    model = XGBClassifier(
        scale_pos_weight=cost_ratio,
        n_estimators=config["model_params"]["classifier"]["n_estimators"],
        max_depth=config["model_params"]["classifier"]["max_depth"],
        learning_rate=config["model_params"]["classifier"]["learning_rate"],
        random_state=config["model_params"]["classifier"]["random_state"],
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    return model
