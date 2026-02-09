"""Model evaluation with business metrics."""
import numpy as np
from sklearn.metrics import (
    recall_score,
    precision_score,
    fbeta_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
)


def evaluate_classifier(y_true, y_pred, config) -> dict:
    """
    Evaluate classifier with business-focused metrics.
    Returns: recall, precision, fbeta, annual_savings_gbp
    """
    # Standard metrics
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)

    # Cost-weighted F-beta
    cost_ratio = (
        config["business_params"]["failure_cost_gbp"]
        / config["business_params"]["false_alarm_cost_gbp"]
    )
    beta = np.sqrt(cost_ratio)
    fbeta = fbeta_score(y_true, y_pred, beta=beta)

    # Business savings calculation
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    failure_cost = config["business_params"]["failure_cost_gbp"]
    alarm_cost = config["business_params"]["false_alarm_cost_gbp"]

    prevented_costs = tp * failure_cost
    alarm_costs = fp * alarm_cost
    missed_costs = fn * failure_cost

    # Annualize (assuming 30-day prediction horizon)
    annual_savings = (prevented_costs - alarm_costs - missed_costs) * (365 / 30)

    return {
        "recall": float(recall),
        "precision": float(precision),
        f"fbeta_{beta:.2f}": float(fbeta),
        "annual_savings_gbp": float(annual_savings),
    }


def evaluate_regressor(y_true, y_pred) -> dict:
    """Evaluate regression model."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # Business metric: % predictions within 6 days
    within_6days = np.mean(np.abs(y_true - y_pred) <= 6)

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "within_6days_pct": float(within_6days),
    }
