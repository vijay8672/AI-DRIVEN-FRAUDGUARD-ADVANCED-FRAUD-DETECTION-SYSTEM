import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from src.logging.logger import logger  # Assuming logger is configured in src/logging/logger.py

def main():
    # Log message for the start of model evaluation
    logger.info("Starting model evaluation process...")

    # Load the test dataset
    file_path = 'artifacts/Important_features_data.csv'  # Path to preprocessed data
    logger.info("Loading dataset from: %s", file_path)
    data = pd.read_csv(file_path)

    # Separate features and target variable
    X = data.drop(columns=['isFraud'])
    y = data['isFraud']
    logger.info("Dataset loaded with shape: %s", data.shape)

    # Load the trained ensemble model
    model_path = "artifacts/ensemble_model.pkl"
    logger.info(f"Loading trained model from: {model_path}")
    ensemble_model = joblib.load(model_path)

    # Make predictions on the test set
    logger.info("Making predictions on test data...")
    y_pred = ensemble_model.predict(X)
    y_prob = ensemble_model.predict_proba(X)[:, 1]  # Get probability for ROC AUC score

    # Evaluate model performance
    accuracy = accuracy_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_prob)

    logger.info(f"Accuracy score on test data: {accuracy}")
    logger.info(f"ROC AUC score on test data: {roc_auc}")

    # Log evaluation metrics
    logger.info("Logging evaluation metrics...")
    logger.info(f"Test Accuracy: {accuracy}")
    logger.info(f"Test ROC AUC: {roc_auc}")

    logger.info("Model evaluation completed successfully.")

if __name__ == "__main__":
    main()
