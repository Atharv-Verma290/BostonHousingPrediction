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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from boston_housing_prediction.model_building import ModelBuilder, LinearRegressionStrategy


def model_building_step(X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
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
    
    model_builder = ModelBuilder(LinearRegressionStrategy())

    trained_model = model_builder.build_model(X_train=X_train, y_train=y_train)

    return trained_model
