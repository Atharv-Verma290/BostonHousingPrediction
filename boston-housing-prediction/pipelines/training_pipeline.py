import os
import logging
import sys
import mlflow
# Add the root directory of your project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from steps.data_ingestion_step import data_ingestion_step
from steps.handle_missing_values_step import handle_missing_values_step
from steps.outlier_detection_step import outlier_detection_step
from steps.feature_engineering_step import feature_engineering_step
from steps.data_splitter_step import data_splitter_step
from steps.model_building_step import model_building_step
from steps.model_evaluator_step import model_evaluator_step
from steps.model_registration_step import model_registration_step

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Boston_Housing_Prediction")

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
    
    # Feature Engineering Step
    X_train_transformed, X_test_transformed = feature_engineering_step(train_df=X_train, test_df=X_test, strategy="minmax_scaling")

    # Model Building Step
    models = model_building_step(X_train=X_train_transformed, y_train=y_train)

    # Model Evaluation Step
    evaluation_report, best_model = model_evaluator_step(trained_models=models, X_test=X_test_transformed, y_test=y_test)

    # Best Model Registration Step
    model_registration_step(eval_report=evaluation_report, model=best_model, input_example=X_train)

    logging.info("train pipeline ran successully.")



if __name__ == "__main__":
    # Running the pipeline
    train_pipeline()