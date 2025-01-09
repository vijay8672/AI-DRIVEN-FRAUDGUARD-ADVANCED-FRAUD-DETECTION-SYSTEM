import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from src.logging.logger import logger  # Assuming logger is configured in src/logging/logger.py
from src.utils.model_utils import save_model  # Assuming save_model is already defined
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report  # For evaluating predictions


def Hyperparameter_Tuning():
    # Log message for the start of hyperparameter tuning
    logger.info("Starting hyperparameter tuning process...")

    # Load dataset
    file_path = 'artifacts/data/Important_features_data.csv'  # Path to preprocessed data
    logger.info("Loading dataset from: %s", file_path)
    data = pd.read_csv(file_path)
    logger.info("Reading data from dataset features present are %s", data.columns)
    
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

    # Apply SMOTE + Tomek Links for resampling
    logger.info("Applying SMOTE + Tomek Links for resampling...")
    smote_tomek = SMOTETomek(random_state=42)
    X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)

    # Initialize models
    models = [(XGBClassifier(eval_metric='mlogloss'), "xgboost_model")]

    # Hyperparameter grids for tuning
    param_grid = {
        'xgboost_model': {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.1, 0.2, 0.3],
        'n_estimators': [50, 100, 150],            
        'scale_pos_weight': [1, 10, 20],  
        'subsample': [0.8, 0.9, 1.0],  
        'eval_metric': ['logloss', 'auc']
        }
        }
    # Hyperparameter tuning
    for model, model_name in models:
        logger.info(f"Tuning hyperparameters for {model_name}...")
        param_grid = param_grid.get(model_name)

        search = RandomizedSearchCV(model, param_grid, n_iter=6, cv=3, scoring='recall', random_state=42, n_jobs=-1, verbose=1)
        search.fit(X_train_resampled, y_train_resampled)

        # Extract best parameters and score
        best_params = search.best_params_
        best_score = search.best_score_
        logger.info(f"Best parameters for {model_name}: {best_params}")
        logger.info(f"Best score for {model_name}: {best_score:.4f}")

        # Get best model
        best_model = search.best_estimator_

        # Predict and evaluate
        y_pred = best_model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)  # Output as dict for MLflow

        logger.info(f"Classification Report for {model_name}:\n{classification_report(y_test, y_pred)}")

        # Log to MLflow
        with mlflow.start_run(run_name=model_name):
            # Log model
            mlflow.sklearn.log_model(best_model, artifact_path=model_name)

            # Log hyperparameters
            for param, value in best_params.items():
                mlflow.log_param(param, value)

            # Log metrics
            mlflow.log_metric(f"best_recall_score", best_score)
            for label, metrics in report.items():
                if isinstance(metrics, dict):  # Ignore aggregate metrics like "accuracy"
                    for metric_name, metric_value in metrics.items():
                        mlflow.log_metric(f"{label}_{metric_name}", metric_value)

    logger.info("Hyperparameter tuning and prediction completed successfully.")


if __name__ == "__main__":
    Hyperparameter_Tuning()