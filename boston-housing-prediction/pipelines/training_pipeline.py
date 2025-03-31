import os
import logging
import sys
# Add the root directory of your project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from steps.data_ingestion_step import data_ingestion_step
from steps.handle_missing_values_step import handle_missing_values_step
from steps.outlier_detection_step import outlier_detection_step
from steps.feature_engineering_step import feature_engineering_step
from steps.data_splitter_step import data_splitter_step

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def train_pipeline():
    """Define an end-to-end machine learning pipeline."""

    # Data Ingestion Step
    raw_data = data_ingestion_step(
        file_path=os.path.join('..', 'data', 'archive.zip')
    )

    # Handling Missing Values Step
    filled_data = handle_missing_values_step(raw_data)

    # Outlier Detection Step
    # clean_data = outlier_detection_step(filled_data, column_name="MEDV")

    # Data Splitting Step
    X_train, X_test, y_train, y_test = data_splitter_step(filled_data, target_column="MEDV")
    
    # # Feature Engineering Step
    # engineered_data = feature_engineering_step(
    #     clean_data, strategy="log", features=["Gr Liv Area", "SalePrice"]
    # )
    logging.info("train pipeline ran successully.")



if __name__ == "__main__":
    # Running the pipeline
    train_pipeline()