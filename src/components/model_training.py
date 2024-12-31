import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from src.logging.logger import logger  # Assuming logger is configured in src/logging/logger.py
from src.utils.model_utils import save_model  # Import your existing save_model function

def Model_Training():
    # Log message for the start of model training
    logger.info("Starting model training process...")

    # Load dataset
    file_path = 'artifacts/Important_features_data.csv'  # Path to preprocessed data
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

    # Initialize models
    logger.info("Initializing base models...")
    models = [
        (XGBClassifier(eval_metric='mlogloss'), "xgboost_model"),
        (lgb.LGBMClassifier(), "lightgbm_model"),
        (CatBoostClassifier(learning_rate=0.1, iterations=80, depth=6, verbose=0), "catboost_model"),
        (SVC(kernel='linear', class_weight='balanced', probability=True, random_state=42), "svm_model")
    ]

    # Train and save each individual model
    logger.info("Training and saving individual models...")

    for model, model_name in models:
        logger.info(f"Training {model_name}...")
        # Cross-validation (reduced to 3 splits for faster training)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring='accuracy', n_jobs=-1)
        logger.info(f"Cross-validation scores for {model_name}: {cv_scores}")

        # Train the model   
        model.fit(X_train_resampled, y_train_resampled)
        
        # Save the trained model
        save_model(model, model_name, save_dir="artifacts/models/")

    # Create and train the ensemble model using VotingClassifier
    logger.info("Creating ensemble model with soft voting...")

    # Define ensemble model
    ensemble_model = VotingClassifier(estimators=[(name, model) for model, name in models], voting='soft')

    # Cross-validation for ensemble model (reduced to 3 splits for faster training)
    logger.info("Performing cross-validation for ensemble model...")
    cv_scores = cross_val_score(ensemble_model, X_train_resampled, y_train_resampled, cv=3, scoring='accuracy', n_jobs=-1)
    logger.info(f"Cross-validation scores for ensemble model: {cv_scores}")

    # Train the ensemble model
    logger.info("Training ensemble model on the full training set...")
    ensemble_model.fit(X_train_resampled, y_train_resampled)

    # Save the trained ensemble model
    save_model(ensemble_model, "ensemble_model", save_dir="artifacts/models/")

    logger.info("Model training completed successfully.")

if __name__ == "__main__":
    Model_Training()
