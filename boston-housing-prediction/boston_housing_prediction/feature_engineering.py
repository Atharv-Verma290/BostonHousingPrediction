import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Abstract Base Class for Feature Engineering Strategy
class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame, is_testset: bool = False) -> pd.DataFrame:
        """
        Abstract method to apply feature engineering transformation to the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.
        is_testset (bool): If the dataframe is testing dataframe or not.

        Returns:
        pd.DataFrame: A dataframe with the applied transformations.
        """
        pass


# Concrete Strategy for Log Transformation
class LogTransformation(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the LogTransformation with the specific features to transform.

        Parameters:
        features (list): The list of features to apply the log transformation to.
        """
        self.features = features
    
    def apply_transformation(self, df: pd.DataFrame, is_testset: bool = False) -> pd.DataFrame:
        """
        Applies a log transformation to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.
        is_testset (bool): If the dataframe is testing dataframe or not.

        Returns:
        pd.DataFrame: The dataframe with log-transformed features.
        """
        logging.info(f"Applying log transformation to features: {self.features}")
        df_transformed = df.copy()
        for feature in self.features:
            df_transformed[feature] = np.log1p(
                df[feature]
            )  # log1p handles log(0) by calculating log(1+x)
        logging.info("Log transformation completed.")
        return df_transformed
    

# Concrete Strategy for Standard Scaling
class StandardScaling(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the StandardScaling with the specific features to scale.

        Parameters:
        features (list): The list of features to apply the standard scaling to.
        """
        self.features = features
        self.scaler = StandardScaler()
    
    def apply_transformation(self, df: pd.DataFrame, is_testset: bool = False) -> pd.DataFrame:
        """
        Applies standard scaling to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.
        is_testset (bool): If the dataframe is testing dataframe or not.

        Returns:
        pd.DataFrame: The dataframe with scaled features.
        """
        logging.info(f"Applying standard scaling to features: {self.features if self.features else 'entire DataFrame'}")
        df_transformed = df.copy()
        if not self.features: # If features list is empty, apply scaling to the entire DataFrame
            if is_testset:
                scaled_array = self.scaler.transform(df)
            else:
                scaled_array = self.scaler.fit_transform(df)
            # Convert the scaled array back to a DataFrame with float64 dtype
            df_transformed = pd.DataFrame(scaled_array, columns=df.columns, index=df.index, dtype="float64")
        else:
            if is_testset:
                scaled_array = self.scaler.transform(df[self.features])
            else:
                scaled_array = self.scaler.fit_transform(df[self.features])
            # Assign the scaled values back to the specified features
            df_transformed[self.features] = pd.DataFrame(scaled_array, columns=self.features, index=df.index, dtype="float64")
        logging.info("Standard scaling completed.")
        return df_transformed
    

# Concrete Strategy for Min-Max Scaling
class MinMaxScaling(FeatureEngineeringStrategy):
    def __init__(self, features, feature_range=(0, 1)):
        """
        Initializes the MinMaxScaling with the specific features to scale and the target range.

        Parameters:
        features (list): The list of features to apply the Min-Max scaling to.
        feature_range (tuple): The target range for scaling, default is (0, 1).
        """
        self.features = features
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def apply_transformation(self, df: pd.DataFrame, is_testset: bool = False) -> pd.DataFrame:
        """
        Applies Min-Max scaling to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.
        is_testset (bool): If the dataframe is testing dataframe or not.

        Returns:
        pd.DataFrame: The dataframe with Min-Max scaled features.
        """
        logging.info(f"Applying Min-Max scaling to features: {self.features if self.features else 'entire DataFrame'} with range {self.scaler.feature_range}")
        df_transformed = df.copy()
        if not self.features: # If features list is empty, apply scaling to the entire DataFrame
            if is_testset:
                scaled_array = self.scaler.transform(df)
            else:
                scaled_array = self.scaler.fit_transform(df)
            # Convert the scaled array back to a DataFrame with float64 dtype
            df_transformed = pd.DataFrame(scaled_array, columns=df.columns, index=df.index, dtype="float64")
        else:
            if is_testset:
                scaled_array = self.scaler.transform(df[self.features])
            else:
                scaled_array = self.scaler.fit_transform(df[self.features])
            # Assign the scaled values back to the specified features
            df_transformed[self.features] = pd.DataFrame(scaled_array, columns=self.features, index=df.index, dtype="float64")

        logging.info("Min-Max scaling completed.")
        return df_transformed
    

# Concrete Strategy for One-Hot Encoding
class OneHotEncoding(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the OneHotEncoding with the specific features to encode.

        Parameters:
        features (list): The list of categorical features to apply the one-hot encoding to.
        """
        self.features = features
        self.encoder = OneHotEncoder(sparse=False, drop="first")

    def apply_transformation(self, df: pd.DataFrame, is_testset: bool = False) -> pd.DataFrame:
        """
        Applies one-hot encoding to the specified categorical features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.
        is_testset (bool): If the dataframe is testing dataframe or not.

        Returns:
        pd.DataFrame: The dataframe with one-hot encoded features.
        """
        logging.info(f"Applying one-hot encoding to features: {self.features}")
        df_transformed = df.copy()
        encoded_df = pd.DataFrame(
            self.encoder.fit_transform(df[self.features]),
            columns=self.encoder.get_feature_names_out(self.features),
        )
        df_transformed = df_transformed.drop(columns=self.features).reset_index(drop=True)
        df_transformed = pd.concat([df_transformed, encoded_df], axis=1)
        logging.info("One-hot encoding completed.")
        return df_transformed
    

# Context Class for Feature Engineering
class FeatureEngineer:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        """
        Initializes the FeatureEngineer with a specific feature engineering strategy.

        Parameters:
        strategy (FeatureEngineeringStrategy): The strategy to be used for feature engineering.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: FeatureEngineeringStrategy):
        """
        Sets a new strategy for the FeatureEngineer.

        Parameters:
        strategy (FeatureEngineeringStrategy): The new strategy to be used for feature engineering.
        """
        logging.info("Switching feature engineering strategy.")
        self._strategy = strategy

    def apply_feature_engineering(self, df: pd.DataFrame, is_testset: bool = False) -> pd.DataFrame:
        """
        Executes the feature engineering transformation using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.
        is_testset (bool): If the dataframe is testing dataframe or not.

        Returns:
        pd.DataFrame: The dataframe with applied feature engineering transformations.
        """
        logging.info("Applying feature engineering strategy.")
        return self._strategy.apply_transformation(df, is_testset)
    

# Example usage
if __name__ == "__main__":
    # Example dataframe
    # df = pd.read_csv('../extracted-data/your_data_file.csv')

    # Log Transformation Example
    # log_transformer = FeatureEngineer(LogTransformation(features=['SalePrice', 'Gr Liv Area']))
    # df_log_transformed = log_transformer.apply_feature_engineering(df)

    # Standard Scaling Example
    # standard_scaler = FeatureEngineer(StandardScaling(features=['SalePrice', 'Gr Liv Area']))
    # df_standard_scaled = standard_scaler.apply_feature_engineering(df)

    # Min-Max Scaling Example
    # minmax_scaler = FeatureEngineer(MinMaxScaling(features=['SalePrice', 'Gr Liv Area'], feature_range=(0, 1)))
    # df_minmax_scaled = minmax_scaler.apply_feature_engineering(df)

    # One-Hot Encoding Example
    # onehot_encoder = FeatureEngineer(OneHotEncoding(features=['Neighborhood']))
    # df_onehot_encoded = onehot_encoder.apply_feature_engineering(df)

    pass