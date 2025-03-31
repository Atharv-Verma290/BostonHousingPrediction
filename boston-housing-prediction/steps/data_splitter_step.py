from typing import Tuple
import pandas as pd
import os
import sys
import logging
# Add the root directory of your project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from boston_housing_prediction.data_splitter import DataSplitter, SimpleTrainTestSplitStrategy


def data_splitter_step(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits the data into training and testing sets using DataSplitter and a chosen strategy."""
    splitter = DataSplitter(strategy=SimpleTrainTestSplitStrategy())
    X_train, X_test, y_train, y_test = splitter.split(df, target_column)
    return X_train, X_test, y_train, y_test