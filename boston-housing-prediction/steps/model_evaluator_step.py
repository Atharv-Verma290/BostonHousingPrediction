from typing import Tuple
import pandas as pd
import os
import sys
import logging
from sklearn.base import RegressorMixin
# Add the root directory of your project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from boston_housing_prediction.model_evaluator import ModelEvaluator, RegressionModelEvaluationStrategy


def model_evaluator_step(trained_models: list[RegressorMixin], X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[list[dict], RegressorMixin]:
    """
    Evaluates the trained model using ModelEvaluator and RegressionModelEvaluationStrategy.

    Parameters:
    trained_models (list[RegressorMixin]): A list of trained regression models to evaluate.
    X_test (pd.DataFrame): The test data features.
    y_test (pd.Series): The test data labels/target.

    Returns:
    dict: A list of dictionaries with evaluation metrics for each model.
    best_model: The best performing model
    """
    # Ensure the inputs are of the correct type
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame.")
    if not isinstance(y_test, pd.Series):
        raise TypeError("y_test must be a pandas Series.")
    
    # Initialize the evaluator with the regression strategy
    evaluator = ModelEvaluator(strategy=RegressionModelEvaluationStrategy())

    # Perform the evaluation
    evaluation_metrics, best_model = evaluator.evaluate(trained_models, X_test, y_test)


    return evaluation_metrics, best_model