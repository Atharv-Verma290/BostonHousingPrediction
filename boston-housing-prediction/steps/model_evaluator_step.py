from typing import Tuple
import pandas as pd
import os
import sys
import logging
from sklearn.base import RegressorMixin
# Add the root directory of your project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from boston_housing_prediction.model_evaluator import ModelEvaluator, RegressionModelEvaluationStrategy


def model_evaluator_step(trained_model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[dict, float]:
    """
    Evaluates the trained model using ModelEvaluator and RegressionModelEvaluationStrategy.

    Parameters:
    trained_model (Pipeline): The trained model instance.
    X_test (pd.DataFrame): The test data features.
    y_test (pd.Series): The test data labels/target.

    Returns:
    dict: A dictionary containing evaluation metrics.
    """
    # Ensure the inputs are of the correct type
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame.")
    if not isinstance(y_test, pd.Series):
        raise TypeError("y_test must be a pandas Series.")
    
    # Initialize the evaluator with the regression strategy
    evaluator = ModelEvaluator(strategy=RegressionModelEvaluationStrategy())

    # Perform the evaluation
    evaluation_metrics = evaluator.evaluate(trained_model, X_test, y_test)

    # Ensure that the evaluation metrics are returned as a dictionary
    if not isinstance(evaluation_metrics, dict):
        raise ValueError("Evaluation metrics must be returned as a dictionary.")
    mse = evaluation_metrics.get("Mean Squared Error", None)
    return evaluation_metrics, mse