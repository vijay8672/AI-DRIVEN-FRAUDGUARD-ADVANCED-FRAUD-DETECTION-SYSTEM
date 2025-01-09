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
from imblearn.combine import SMOTETomek
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

    # Apply SMOTE + Tomek Links for resampling
    logger.info("Applying SMOTE + Tomek Links for resampling...")
    smote_tomek = SMOTETomek(random_state=42)
    X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)

    # Use class weights in models
    models = [
        (XGBClassifier(eval_metric='mlogloss', max_depth=4, n_estimators=100, learning_rate=0.2, scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1])), "xgboost_model"),
        (lgb.LGBMClassifier(max_depth=4, n_estimators=100, learning_rate=0.2, class_weight='balanced'), "lightgbm_model"),
        (CatBoostClassifier(learning_rate=0.2, iterations=100, depth=4, verbose=0, auto_class_weights='Balanced'), "catboost_model")
    ]

    # Initialize StratifiedKFold for cross-validation
    strat_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Use ROC-AUC or F1 score for cross-validation
    logger.info("Performing cross-validation with ROC-AUC scoring...")
    for model, model_name in models:
        model_cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=strat_kfold, scoring='roc_auc')
        logger.info(f"{model_name} ROC-AUC scores: {model_cv_scores}")
        logger.info(f"Average ROC-AUC for {model_name}: {model_cv_scores.mean()}")

        # Train the model on the entire training set
        model.fit(X_train_resampled, y_train_resampled)

        # Save the trained model (if required)
        save_model(model, model_name, save_dir="artifacts/models2/")

        logger.info("model training completed and saved.")


if __name__ == "__main__":
    Model_Training()
