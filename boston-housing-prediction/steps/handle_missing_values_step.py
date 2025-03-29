import pandas as pd
import os
import sys
# Add the root directory of your project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from boston_housing_prediction.handle_missing_values import (
    DropMissingValuesStrategy,
    FillMissingValuesStrategy,
    MissingValueHandler
)


def handle_missing_values_step(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """Handles missing values using MissingValueHandler and the specified strategy."""
    if strategy == "drop":
        handler = MissingValueHandler(DropMissingValuesStrategy(axis=0))
    elif strategy in ["mean", "median", "mode", "constant"]:
        handler = MissingValueHandler(FillMissingValuesStrategy(method=strategy))
    else:
        raise ValueError(f"Unsupported missing value handling strategy: {strategy}")
    
    cleaned_df = handler.handle_missing_values(df)
    return cleaned_df