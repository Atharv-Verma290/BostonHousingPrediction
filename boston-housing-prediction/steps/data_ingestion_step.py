import pandas as pd
import os
import sys
# Add the root directory of your project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from boston_housing_prediction.ingest_data import DataIngestorFactory


def data_ingestion_step(file_path: str) -> pd.DataFrame:
    """Ingest data from a ZIP file using the appropriate DataIngestor."""
    # Dynamically determine the file extension
    file_extension = os.path.splitext(file_path)[1]

    # Get the appropriate DataIngestor
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)

    # Ingest the data and load it into a DataFrame
    df = data_ingestor.ingest(file_path)
    return df


# # Example usage for testing
# if __name__ == "__main__":
#     # Construct the file path dynamically
#     current_dir = os.getcwd()
#     file_path = os.path.join(current_dir, '..', 'boston_housing_prediction', 'ingest_data.py')

#     # Call the data ingestion step
#     df = data_ingestion_step(file_path=file_path)