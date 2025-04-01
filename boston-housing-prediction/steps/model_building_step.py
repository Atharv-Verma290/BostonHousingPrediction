from typing import Tuple
import pandas as pd
import os
import sys
import logging
# Add the root directory of your project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mlflow
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from boston_housing_prediction.model_building import ModelBuilder, LinearRegressionStrategy, RandomForestRegressionStrategy


def model_building_step(X_train: pd.DataFrame, y_train: pd.Series) -> list[RegressorMixin]:
    """
    Builds and trains a Linear Regression model using scikit-learn wrapped in a pipeline.

    Parameters:
    X_train (pd.DataFrame): The training data features.
    y_train (pd.Series): The training data labels/target.

    Returns:
    RegressorMixin: The trained scikit-learn model instance.
    """

    # Ensure the inputs are of the correct type
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series.")
    
    model_list = ["linear_regression", "random_forest"]
    params = {
        "linear_regression": {},
        "random_forest": {
            'n_estimators': [100, 300, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
            'max_features': ['sqrt', 'log2', None]
        }
    }
    trained_models = []

    model_builder = ModelBuilder(LinearRegressionStrategy())
    for model_name in model_list:
        if model_name == "linear_regression":
            model_builder.set_strategy(LinearRegressionStrategy())
        elif model_name == "random_forest": 
            model_builder.set_strategy(RandomForestRegressionStrategy())
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        trained_model = model_builder.build_model(X_train=X_train, y_train=y_train, params=params[model_name])
        trained_models.append(trained_model)
    
    return trained_models
