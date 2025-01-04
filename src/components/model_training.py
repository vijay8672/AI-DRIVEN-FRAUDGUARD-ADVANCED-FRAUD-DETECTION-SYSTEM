import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from src.logging.logger import logger  # Assuming logger is configured in src/logging/logger.py
from src.utils.model_utils import save_model  # Import your existing save_model function
from src.utils.model_utils import save_artifact  # Import your existing save_features function

def Model_Training():
    # Log message for the start of model training
    logger.info("Starting model training process...")

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

    # Scale the features using RobustScaler
    logger.info("Scaling features using RobustScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)  # Scale test set using the same scaler

    # Save the scaler as an artifact
    save_artifact(scaler, 'artifacts/model_artifacts/scaler.pkl')

    # Initialize models
    logger.info("Initializing base models...")
    models = [
        (XGBClassifier(eval_metric='mlogloss', max_depth=4, n_estimators=100, learning_rate=0.2), "xgboost_model"),
        (lgb.LGBMClassifier(max_depth=4, n_estimators=100, learning_rate=0.2, force_col_wise=True), "lightgbm_model"), 
        (CatBoostClassifier(learning_rate=0.2, iterations=100, depth=4, verbose=0), "catboost_model"),
        (AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=42), "adaboost_model") 
    ]

    # Train and save each individual model
    logger.info("Training and saving individual models...")

    for model, model_name in models:
        logger.info(f"Training {model_name}...")

        # Train the model
        model.fit(X_train_scaled, y_train_resampled)
        
        # Save the trained model
        save_model(model, model_name, save_dir="artifacts/models/")

        logger.info(f"Saved {model_name}.")

    logger.info("Model training completed successfully.")

if __name__ == "__main__":
    Model_Training()
