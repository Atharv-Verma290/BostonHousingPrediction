import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import mlflow
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Abstract Base Class for Model Evaluation Strategy
class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_model(self, models: list[RegressorMixin], X_test: pd.DataFrame, y_test: pd.Series) -> tuple[list[dict], RegressorMixin]:
        """
        Abstract method to evaluate a model.

        Parameters:
        models (list[RegressorMixin]): A list of trained regression models to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.Series): The testing data labels/target.

        Returns:
        evaluation_report: A list of dictionaries with evaluation metrics for each model.
        best_model: Best performing model.
        """
        pass


# Concrete Strategy for Regression Model Evaluation
class RegressionModelEvaluationStrategy(ModelEvaluationStrategy):
    def evaluate_model(self, models: list[RegressorMixin], X_test: pd.DataFrame, y_test: pd.Series) -> tuple[list[dict], RegressorMixin]:
        """
        Evaluates a list of regression models using R-squared and Mean Squared Error.

        Parameters:
        models (list[RegressorMixin]): A list of trained regression models to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.Series): The testing data labels/target.

        Returns:
        tuple: A tuple containing:
            - list[dict]: A list of dictionaries with evaluation metrics for each model.
            - RegressorMixin: The best model instance based on the highest R² score.
        """
        logging.info("Evaluating multiple models.")
        report = []
        best_model = None
        best_r2_score = float('-inf')

        for model in models:
            logging.info(f"Evaluating model: {type(model).__name__}")
            with mlflow.start_run(run_name=f"{type(model).__name__} Model Testing") as run:
                y_pred = model.predict(X_test)

                # Calculate evaluation metrics
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                # Append metrics to the report
                report.append({
                    "model": type(model).__name__,
                    "mse": mse,
                    "r2_score": r2
                })

                mlflow_X_test = mlflow.data.from_pandas(X_test)
                mlflow_y_test = mlflow.data.from_pandas(y_test.to_frame())
                mlflow_y_pred = mlflow.data.from_numpy(y_pred)

                mlflow.log_metrics({"mse": mse, "r2_score": r2})
                mlflow.log_input(mlflow_X_test, "X_test")
                mlflow.log_input(mlflow_y_test, "y_test")
                mlflow.log_input(mlflow_y_pred, "y_pred")

                mlflow.sklearn.log_model(model, f"{type(model).__name__}")
                mlflow.set_tag("testing", f"{type(model).__name__}")

                # Update the best model based on R² score
                if r2 > best_r2_score:
                    best_r2_score = r2
                    best_model = model

        logging.info(f"Evaluation report generated: {report}")
        logging.info(f"Best model: {type(best_model).__name__} with R² score: {best_r2_score}")
        return report, best_model
    

# Context Class for Model Evaluation
class ModelEvaluator:
    def __init__(self, strategy: ModelEvaluationStrategy):
        """
        Initializes the ModelEvaluator with a specific model evaluation strategy.

        Parameters:
        strategy (ModelEvaluationStrategy): The strategy to be used for model evaluation.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: ModelEvaluationStrategy):
        """
        Sets a new strategy for the ModelEvaluator.

        Parameters:
        strategy (ModelEvaluationStrategy): The new strategy to be used for model evaluation.
        """
        logging.info("Switching model evaluation strategy.")
        self._strategy = strategy

    def evaluate(self, models: list[RegressorMixin], X_test: pd.DataFrame, y_test: pd.Series) -> tuple[list[dict], RegressorMixin]:
        """
        Executes the model evaluation using the current strategy.

        Parameters:
        models (list[RegressorMixin]): A list of trained regression models to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.Series): The testing data labels/target.

        Returns:
        dict: A list of dictionaries with evaluation metrics for each model.
        best_model: Best performing model.
        """
        logging.info("Evaluating the models using the selected strategy.")
        return self._strategy.evaluate_model(models, X_test, y_test)
    

# Example usage
if __name__ == "__main__":
    # Example trained model and data (replace with actual trained model and data)
    # model = trained_sklearn_model
    # X_test = test_data_features
    # y_test = test_data_target

    # Initialize model evaluator with a specific strategy
    # model_evaluator = ModelEvaluator(RegressionModelEvaluationStrategy())
    # evaluation_metrics = model_evaluator.evaluate(model, X_test, y_test)
    # print(evaluation_metrics)

    pass