"""MLflow utilities."""
import mlflow


def setup_mlflow(config):
    """Setup MLflow tracking."""
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
