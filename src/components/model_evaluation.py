import os
import joblib
import pandas as pd
import mlflow
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)
from src.logging.logger import logger  # Assuming logger is configured in src/logging/logger.py
from src.utils.model_utils import load_model  # Assuming load_model function exists for loading saved models
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.combine import SMOTETomek


def Model_Evaluation():
    # Log message for the start of model evaluation
    logger.info("Starting model evaluation process...")

    # Start an MLflow run to log metrics
    with mlflow.start_run():
        # Load dataset
        file_path = 'artifacts/data/Important_features_data.csv'  # Path to preprocessed data
        logger.info("Loading dataset from: %s", file_path)
        data = pd.read_csv(file_path)
        logger.info("Reading data from dataset, features present are %s", data.columns)
        
        # Separate features and target variable
        X = data.drop(columns=['isFraud'])
        y = data['isFraud']
        logger.info("Dataset loaded with shape: %s", data.shape)

        # Log the feature columns used for training
        feature_columns = X.columns.tolist()  # List of feature columns
        mlflow.log_param("features_used", feature_columns)  # Log feature columns in MLflow
        logger.info(f"Features used for training: {feature_columns}")

        # Split the data into training and testing sets (stratified split for imbalanced dataset)
        logger.info("Splitting dataset into training and test sets with stratified sampling...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


        # Define the directory where the models are saved
        model_dir = "artifacts/models2/"  # Update with your model directory

        # List all model files in the model directory (assuming .pkl extension for model files)
        pickle_file = [file for file in os.listdir(model_dir) if file.endswith('.pkl')]
        logger.info(f"Found {pickle_file} models in the directory: {model_dir}")

        # Loop through each model file and evaluate
        for model_name in pickle_file:            
            # Load the model using joblib (or your load_model function)
            if os.path.exists(model_dir):
                model = load_model(model_name, model_dir)  # Adjust if you're using joblib or other loaders
                logger.info(f"Loaded model {model_name} successfully.")
            else:
                logger.error(f"Model file {model_dir} not found.")
                continue  # Skip if model is not found

            # Make predictions
            logger.info(f"Making predictions with {model_name}...")
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

            # Evaluate performance
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

            # Log metrics to MLflow
            mlflow.log_metric(f"{model_name}_accuracy", accuracy)
            mlflow.log_metric(f"{model_name}_precision", precision)
            mlflow.log_metric(f"{model_name}_recall", recall)
            mlflow.log_metric(f"{model_name}_f1", f1)
            if roc_auc is not None:
                mlflow.log_metric(f"{model_name}_roc_auc", roc_auc)

            # Log to local logs
            logger.info(f"Metrics for model {model_name}:")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1-score: {f1:.4f}")
            if roc_auc is not None:
                logger.info(f"ROC-AUC Score: {roc_auc:.4f}")

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            logger.info(f"Confusion Matrix for model {model_name}:\n{cm}")

            # Classification report
            cr = classification_report(y_test, y_pred)
            logger.info(f"Classification Report for model {model_name}:\n{cr}")

            # Save confusion matrix and classification report to text files
            cm_file = f"{model_name}_confusion_matrix.txt"
            cr_file = f"{model_name}_classification_report.txt"

            with open(cm_file, 'w') as f:
                f.write(str(cm))
            with open(cr_file, 'w') as f:
                f.write(cr)

            # Log confusion matrix and classification report to MLflow as text artifacts
            mlflow.log_artifact(cm_file)
            mlflow.log_artifact(cr_file)

            # Clean up the text files after logging
            os.remove(cm_file)
            os.remove(cr_file)

        logger.info("Model evaluation completed successfully.")

if __name__ == "__main__":
    Model_Evaluation()
