import pandas as pd
import os
import sys
import logging
# Add the root directory of your project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from boston_housing_prediction.feature_engineering import (
    FeatureEngineer,
    LogTransformation,
    MinMaxScaling,
    StandardScaling,
    OneHotEncoding
)


def feature_engineering_step(train_df: pd.DataFrame, test_df:pd.DataFrame, strategy: str = "log", features: list = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Performs feature engineering using FeatureEngineer and selected strategy. on both training and testing datasets"""

    # Ensure features is a list, even if not provided
    if features is None:
        features = []  # or raise an error if features are required

    if strategy == "log":
        engineer = FeatureEngineer(LogTransformation(features))
    elif strategy == "standard_scaling":
        engineer = FeatureEngineer(StandardScaling(features))
    elif strategy == "minmax_scaling":
        engineer = FeatureEngineer(MinMaxScaling(features))
    elif strategy == "onehot_encoding":
        engineer = FeatureEngineer(OneHotEncoding(features))
    else:
        raise ValueError(f"Unsupported feature engineering strategy: {strategy}")
    
    transformed_train_df = engineer.apply_feature_engineering(train_df)
    transformed_test_df = engineer.apply_feature_engineering(test_df, is_testset=True)
    return transformed_train_df, transformed_test_df