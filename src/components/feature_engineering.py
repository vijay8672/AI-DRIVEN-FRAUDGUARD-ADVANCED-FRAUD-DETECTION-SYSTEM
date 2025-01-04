import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from src.logging.logger import logger
from src.utils.model_utils import save_artifact
import joblib


# Step 1: Handle missing values
def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    logger.info("Handling missing values...")
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    categorical_columns = data.select_dtypes(include=['object']).columns

    if data.isnull().sum().sum() > 0:
        logger.info("Imputing missing values...")
        num_imputer = SimpleImputer(strategy="median")
        cat_imputer = SimpleImputer(strategy="most_frequent")
        data[numerical_columns] = num_imputer.fit_transform(data[numerical_columns])
        data[categorical_columns] = cat_imputer.fit_transform(data[categorical_columns])
    else:
        logger.info("No missing values detected.")
    return data.copy()


# Step 2: Remove duplicate rows
def remove_duplicate_values(data: pd.DataFrame) -> pd.DataFrame:
    logger.info("Removing duplicate rows...")
    initial_shape = data.shape
    if data.duplicated().sum() > 0:
        data = data.drop_duplicates().reset_index(drop=True)
        logger.info(f"Removed {initial_shape[0] - data.shape[0]} duplicate rows.")
    else:
        logger.info("No duplicate rows detected.")
    return data.copy()


# Step 3: Encode categorical features
def encode_categorical_features(data: pd.DataFrame) -> pd.DataFrame:
    logger.info("Encoding categorical features...")

    # Select categorical and numerical columns
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    numerical_columns = data.select_dtypes(exclude=['object']).columns.tolist()

    if not categorical_columns:
        logger.info("No categorical columns detected. Skipping encoding.")
        return data.copy()

    logger.info(f"Categorical columns detected: {categorical_columns}. Starting encoding.")

    # OneHotEncoder with drop_first to avoid dummy variable trap
    encoder = ColumnTransformer(
        transformers=[('categorical', OneHotEncoder(drop='first', sparse_output=False), categorical_columns)],
        remainder='passthrough'
    )

    # Fit and transform the data
    encoded_data = encoder.fit_transform(data)
    save_artifact(encoder, 'artifacts/model_artifacts/encoder.pkl')

    # Retrieve the new column names in the correct order
    encoded_columns = encoder.transformers_[0][1].get_feature_names_out(categorical_columns)
    all_columns = encoded_columns.tolist() + numerical_columns

    # Create DataFrame with proper column names and order
    preprocessed_data = pd.DataFrame(encoded_data, columns=all_columns)

    # Ensure numerical columns retain their original data type
    for col in numerical_columns:
        if col in data.columns:
            preprocessed_data[col] = preprocessed_data[col].astype(data[col].dtype)
        else:
            logger.warning(f"Numerical column {col} not found in original data.")

    logger.info(f"Preprocessed data has {preprocessed_data.shape[1]} columns after encoding.")
    logger.info(f"Preprocessed data columns: {list(preprocessed_data.columns)}")

    return preprocessed_data



# Step 4: Remove highly correlated features
def remove_highly_correlated_features(data: pd.DataFrame, threshold: float = 0.87) -> pd.DataFrame:
    logger.info(f"Removing highly correlated features with correlation threshold: {threshold}...")

    corr_matrix = data.corr(numeric_only=True).abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

    if to_drop:
        logger.info(f"Features to be removed due to high correlation: {to_drop}")
        data = data.drop(columns=to_drop)
    else:
        logger.info("No highly correlated features found.")

    return data.copy()


# Main Feature Engineering Pipeline
def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting feature engineering pipeline...")

    # Step-by-step transformation ensuring data integrity
    data_step_1 = handle_missing_values(data)
    data_step_2 = remove_duplicate_values(data_step_1)
    data_step_3 = encode_categorical_features(data_step_2)
    final_data = remove_highly_correlated_features(data_step_3)

    return final_data.copy()


# Entry point for script execution
if __name__ == "__main__":
    input_path = 'data/Imported_data.csv'
    output_path = 'data/preprocessed_data.csv'

    data = pd.read_csv(input_path)

    # Separate target and drop unnecessary columns
    target = data.pop('isFraud')
    data = data.drop(columns=['step', 'nameOrig', 'nameDest', 'isFlaggedFraud'])

    # Apply feature engineering pipeline
    processed_data = feature_engineering(data)

    # Add target back to the processed data
    processed_data['isFraud'] = target.reset_index(drop=True)

    # Save the preprocessed data
    processed_data.to_csv(output_path, index=False)
    logger.info(f"Preprocessed data saved at {output_path}")
    logger.info("Feature Engineering pipeline completed successfully.")