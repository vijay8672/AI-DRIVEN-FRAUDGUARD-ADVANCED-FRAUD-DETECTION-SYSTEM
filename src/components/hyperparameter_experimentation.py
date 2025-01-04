import os
import joblib
import mlflow
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from src.logging.logger import logger  # Assuming logger is configured in src/logging/logger.py
from src.utils.model_utils import save_model  # Assuming save_model is already defined
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from scipy.stats import uniform

# Import your model training and evaluation functions
from src.components.model_training import Model_Training  # Replace with the correct import path
from src.components.model_evaluation import Model_Evaluation  # Replace with the correct import path

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

    # Split the data into training and testing sets (stratified split for imbalanced dataset)
    logger.info("Splitting dataset into training and test sets with stratified sampling...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Apply SMOTE for oversampling the minority class in the training data
    logger.info("Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Scale the features using StandardScaler
    logger.info("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)  # Scale test set using the same scaler

    # Initialize models
    models = {
        'xgboost': XGBClassifier(eval_metric='mlogloss', random_state=42),
        'lightgbm': lgb.LGBMClassifier(random_state=42),
        'catboost': CatBoostClassifier(learning_rate=0.2, iterations=100, depth=4, verbose=0),
        'adaboost': AdaBoostClassifier(n_estimators=100, random_state=42)
    }

    # Hyperparameter grids for tuning
    param_grids = {
        'xgboost': {
            'max_depth': [3, 4, 5],
            'learning_rate': [0.1, 0.2, 0.3],
            'n_estimators': [50, 100, 200]
        },
        'lightgbm': {
            'max_depth': [3, 4, 5],
            'learning_rate': [0.05, 0.1, 0.2],
            'n_estimators': [50, 100, 150]
        },
        'catboost': {
            'iterations': [100, 150],
            'learning_rate': [0.1, 0.2],
            'depth': [3, 4, 5]
        },
        'adaboost': {
            'learning_rate': [0.5, 1.0, 1.5],
            'n_estimators': [50, 100, 150],
            'algorithm': ['SAMME', 'SAMME.R'],  # Trying both AdaBoost algorithms
            'random_state': [42]
        }
    }

    # Choosing RandomizedSearchCV
    search_method = 'RandomizedSearchCV'

    # Perform hyperparameter tuning for each model
    logger.info("Starting hyperparameter tuning for models...")
    for model_name, model in models.items():
        logger.info(f"Tuning hyperparameters for {model_name}...")

        param_grid = param_grids.get(model_name)
        
        if search_method == 'RandomizedSearchCV':
            search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=3, scoring='accuracy', random_state=42, n_jobs=-1, verbose=1)

        # Fit the search method to the training data
        search.fit(X_train_scaled, y_train_resampled)

        # Get the best parameters and score
        best_params = search.best_params_
        best_score = search.best_score_

        logger.info(f"Best parameters for {model_name}: {best_params}")
        logger.info(f"Best score for {model_name}: {best_score:.4f}")

        # Save the best model
        best_model = search.best_estimator_
        save_model(best_model, f"best_{model_name}_model", save_dir="artifacts/models/best_models")  # Save the best model

        # Train and evaluate the best model using already existing functions
        logger.info(f"Training the best model for {model_name}...")
        trained_model = Model_Training(best_model, X_train_scaled, y_train_resampled)  # Use your model training function

        # Evaluate the trained model
        logger.info(f"Evaluating the best model for {model_name}...")
        evaluation_results = Model_Evaluation(trained_model, X_test_scaled, y_test)  # Use your model evaluation function
        logger.info(f"Evaluation Results for {model_name}: {evaluation_results}")

        # Log the best hyperparameters and score to MLflow
        mlflow.log_param(f"{model_name}_best_params", str(best_params))
        mlflow.log_metric(f"{model_name}_best_score", best_score)

    logger.info("Hyperparameter tuning completed successfully.")

if __name__ == "__main__":
    Hyperparameter_Tuning()
