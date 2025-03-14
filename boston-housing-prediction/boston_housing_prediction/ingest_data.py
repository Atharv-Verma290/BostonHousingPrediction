import os
import zipfile
from abc import ABC, abstractmethod

import pandas as pd

# Abstract class for Data Ingestor
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str, filename: str = None) -> pd.DataFrame:
        """Abstract method to ingest data from a given file."""
        pass 


# Implement a concrete class for ZIP Ingestion
class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path: str, filename: str = None) -> pd.DataFrame:
        """Extracts a .zip file and returns the content as a pandas DataFrame"""
        # Ensure the file is a .zip
        if not file_path.endswith(".zip"):
            raise ValueError("The provided file is not a .zip file.")
        
        # Extract the zip file
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall("../extracted_data")

        # Find the extracted CSV file (assuming there is one CSV file inside the zip file)
        extracted_files = os.listdir("../extracted_data")
        csv_files = [f for f in extracted_files if f.endswith(".csv")]

        if len(csv_files) == 0:
            raise FileNotFoundError("No CSV file found in the extracted data.")
        if len(csv_files) > 1:
            if filename is None:
                raise ValueError("Multiple CSV files found. Please specify which one to use.")
            elif filename not in csv_files:
                raise FileNotFoundError(f"The specified '{filename}' was not found in the extracted data.")
            csv_file_path = os.path.join("../extracted_data", filename)
        else:
            csv_file_path = os.path.join("../extracted_data", csv_files[0])

        # Read the CSV into a DataFrame
        df = pd.read_csv(csv_file_path)

        # Return the DataFrame
        return df
    

# Implement a Factory to create DataIngestors
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        """Returns the appropriate DataIngestor based on file extension."""
        if file_extension == ".zip":
            return ZipDataIngestor()
        else:
            raise ValueError(f"No ingestor available for file extension: {file_extension}.")
        

# Example Usage:
if __name__ == "__main__":
    # # Specify the file path
    # file_path = "/Users/atharv/MLProjects/BostonHousingPrediction/boston-housing-prediction/data/archive.zip"

    # # Determine the file extension
    # file_extension = os.path.splitext(file_path)[1]

    # # Get the appropriate DataIngestor
    # data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)

    # # Ingest the data and load it into a DataFrame
    # df = data_ingestor.ingest(file_path)

    # # Now df contains the DataFrame from the extracted CSV
    # print(df.head()) 
    pass