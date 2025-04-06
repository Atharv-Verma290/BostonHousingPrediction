from typing import Tuple
import pandas as pd
import os
import sys
import logging
import mlflow
from sklearn.base import RegressorMixin
# Add the root directory of your project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_metrics_for_model(eval_report: list[dict], model_name: str) -> Tuple[float, float]:
    """
    Retrieves the evaluation metrics (MSE and R² score) for a specific model from the evaluation report.

    Parameters:
    eval_report (list[dict]): A list of dictionaries containing evaluation metrics for each model.
    model_name (str): The name of the model for which metrics are to be retrieved.

    Returns:
    Tuple[float, float]: A tuple containing the Mean Squared Error (MSE) and R² score for the specified model.
    """
    logging.info(f"Retrieving metrics for model: {model_name}")
    for entry in eval_report:
        if entry['model'] == model_name:
            return entry['mse'], entry['r2_score']
    return None, None


def model_registration_step(eval_report: list[dict], model: RegressorMixin, input_example: pd.DataFrame):
    """
    Registers the best model to MLflow, along with its parameters and evaluation metrics.

    Parameters:
    eval_report (list[dict]): A list of dictionaries containing evaluation metrics for all models.
    model (RegressorMixin): The best model instance to be registered.
    input_example (pd.DataFrame): An example input DataFrame for the model.

    Returns:
    None
    """
    logging.info(f"Registering Best Model: {type(model).__name__}")

    with mlflow.start_run(run_name=f"Model Registration: {type(model).__name__}") as run:
        # Retrieve evaluation metrics for the best model
        mse, r2 = get_metrics_for_model(eval_report, f"{type(model).__name__}")

        # Log evaluation metrics to MLflow
        if mse is not None and r2 is not None:
            mlflow.log_metrics({"mse": mse, "r2": r2})

        # Log and Register the model to MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=f"sklearn-{type(model).__name__}",
            input_example=input_example,
            registered_model_name=f"sklearn-{type(model).__name__}"
        )

        mlflow.set_tag("registering", f"{type(model).__name__}")

    logging.info(f"Model {type(model).__name__} registered successfully.")


