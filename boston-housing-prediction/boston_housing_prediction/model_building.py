import logging
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd 
import mlflow
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Abstract Base Class for Model Building Strategy
class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, params: dict = None) -> RegressorMixin:
        """
        Abstract method to build and train a model.

        Parameters:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target.
        params (dict, optional): Hyperparameter grid for tuning. Defaults to None.

        Returns:
        RegressorMixin: A trained scikit-learn model instance.
        """
        pass


# Concrete Strategy for Linear Regression using scikit-learn
class LinearRegressionStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, params: dict = None) -> RegressorMixin:
        """
        Builds and trains a linear regression model using scikit-learn.

        Parameters:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target.
        params (dict, optional): Hyperparameter grid for tuning. Defaults to None.

        Returns:
        RegressorMixin: A scikit-learn trained Linear Regression model.
        """
        # Ensure the inputs are of the correct type
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series.")
        
        # Log that no hyperparameters are used for Linear Regression
        logging.info("Linear Regression does not use hyperparameters. Proceeding with default configuration.")

        with mlflow.start_run(run_name="Linear Regression Training") as run:
            # Creating a pipeline with standard scaling and linear regression
            model = LinearRegression()

            logging.info("Training Linear Regression model.")
            model.fit(X_train, y_train)  # Fit the model to the training data
            metrics = {
                'r2_score': model.score(X_train, y_train)
            }

            mlflow_X_train = mlflow.data.from_pandas(X_train)
            mlflow_y_train = mlflow.data.from_pandas(y_train.to_frame())
            mlflow.log_metrics(metrics)
            mlflow.log_input(mlflow_X_train, "X_train")
            mlflow.log_input(mlflow_y_train, "y_train")
            mlflow.sklearn.log_model(model, "Linear Regressor")
            mlflow.set_tag("training", "Linear Regression")
            logging.info("Model training completed.")
        return model
    

class RandomForestRegressionStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, params: dict = None) -> RegressorMixin:
        """
        Builds and trains a Random Forest model using scikit-learn.

        Parameters:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target.
        params (dict, optional): Hyperparameter grid for tuning. Defaults to None.

        Returns:
        RandomForestRegressor: A scikit-learn trained Random Forest model.
        """
        # Ensure the inputs are of the correct type
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series.")
        
        if params:
           
            # Perform hyperparameter tuning using GridSearchCV
            logging.info("Performing hyperparameter tuning for Random Forest model.")
            model = RandomForestRegressor()
            grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5, scoring='r2', n_jobs=-1, verbose=1)
            with mlflow.start_run(run_name="Random Forest Training") as run:
                grid_search.fit(X_train, y_train)

                logging.info(f"Best parameters found: {grid_search.best_params_} with r2_score: {grid_search.best_score_}")

                mlflow_X_train = mlflow.data.from_pandas(X_train)
                mlflow_y_train = mlflow.data.from_pandas(y_train.to_frame())
                mlflow.log_params(grid_search.best_params_)
                mlflow.log_metric("r2_score", grid_search.best_score_)
                mlflow.log_input(mlflow_X_train, "X_train")
                mlflow.log_input(mlflow_y_train, "y_train")
                model = grid_search.best_estimator_
                mlflow.sklearn.log_model(grid_search.best_estimator_, "Best model")
                mlflow.set_tag("training", "Random Forest Regressor")
        
        else:
            # Train the model with default parameters
            logging.info("Training Random Forest model with default parameters.")
            model = RandomForestRegressor(random_state=42)
            with mlflow.start_run(run_name="Random Forest Training") as run:
                model.fit(X_train, y_train)
                mlflow.log_params(model.get_params())
                mlflow.log_metric("r2_train_score", model.score(X_train, y_train))
                mlflow.sklearn.log_model(model, "trained model")
                mlflow_X_train = mlflow.data.from_pandas(X_train)
                mlflow_y_train = mlflow.data.from_pandas(y_train.to_frame())
                mlflow.log_input(mlflow_X_train, "X_train")
                mlflow.log_input(mlflow_y_train, "y_train")
                mlflow.set_tag("training", "Random Forest Regressor")
                logging.info(f"Random forest model with train r2_score: {model.score(X_train, y_train)}")

        logging.info("Model training completed.")
        return model


# Context Class for Model Building
class ModelBuilder:
    def __init__(self, strategy: ModelBuildingStrategy):
        """
        Initializes the ModelBuilder with a specific model building strategy.

        Parameters:
        strategy (ModelBuildingStrategy): The strategy to be used for model building.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: ModelBuildingStrategy):
        """
        Sets a new strategy for the ModelBuilder.

        Parameters:
        strategy (ModelBuildingStrategy): The new strategy to be used for model building.
        """
        logging.info("Switching model building strategy.")
        self._strategy = strategy

    def build_model(self, X_train: pd.DataFrame, y_train: pd.Series, params: dict) -> RegressorMixin:
        """
        Executes the model building and training using the current strategy.

        Parameters:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target.

        Returns:
        RegressorMixin: A trained scikit-learn model instance.
        """
        logging.info("Building and training the model using the selected strategy.")
        return self._strategy.build_and_train_model(X_train, y_train, params)
    

# Example usage
if __name__ == "__main__":
    # Example DataFrame (replace with actual data loading)
    # df = pd.read_csv('your_data.csv')
    # X_train = df.drop(columns=['target_column'])
    # y_train = df['target_column']

    # Example usage of Linear Regression Strategy
    # model_builder = ModelBuilder(LinearRegressionStrategy())
    # trained_model = model_builder.build_model(X_train, y_train)
    # print(trained_model.named_steps['model'].coef_)  # Print model coefficients

    pass