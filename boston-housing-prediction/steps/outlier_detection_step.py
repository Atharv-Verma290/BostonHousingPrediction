import pandas as pd
import os
import sys
import logging
# Add the root directory of your project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from boston_housing_prediction.outlier_detection import OutlierDetector, IQROutlierDetection, ZScoreOutlierDetection


def outlier_detection_step(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Detects and removes outliers using OutlierDetector."""
    logging.info(f"Starting outlier detection step with DataFrame of shape: {df.shape}")
    if df is None:
        logging.error("Received a NoneType DataFrame.")
        raise ValueError("Input df must be a non-null pandas DataFrame.")
    
    if not isinstance(df, pd.DataFrame):
        logging.error(f"Expected pandas DataFrame, got {type(df)} instead.")
        raise ValueError("Input df must be a pandas DataFrame.")
    
    if column_name not in df.columns:
        logging.error(f"Column '{column_name}' does not exist in the DataFrame.")
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    
    print(df.shape)
    # Ensure only numeric columns are passed
    df_numeric = df.select_dtypes(include=[int, float]).drop(columns=['CHAS', 'RAD'], axis=1)
    
    outlier_detector = OutlierDetector(IQROutlierDetection())
    outliers = outlier_detector.detect_outliers(df_numeric)
    df_cleaned = outlier_detector.handle_outliers(df_numeric, method="remove")
    print(df_cleaned.shape)
    return df_cleaned
    