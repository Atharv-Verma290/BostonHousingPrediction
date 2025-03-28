import os
import sys
# Add the root directory of your project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from steps.data_ingestion_step import data_ingestion_step


def train_pipeline():
    """Define an end-to-end machine learning pipeline."""

    # Data Ingestion Step
    raw_data = data_ingestion_step(
        file_path=os.path.join('..', 'data', 'archive.zip')
    )

    print("train pipeline ran successully.")



if __name__ == "__main__":
    # Running the pipeline
    train_pipeline()