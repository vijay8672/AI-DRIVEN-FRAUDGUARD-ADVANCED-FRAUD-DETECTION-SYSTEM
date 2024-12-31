import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from src.logging.logger import logger


# Function to handle missing values
def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    logger.info("Handling missing values...")

    # Handle missing numerical values with median imputation
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy="median")
    data[numerical_columns] = imputer.fit_transform(data[numerical_columns])

    # Handle missing categorical values with most frequent imputation
    categorical_columns = data.select_dtypes(include=['object']).columns
    cat_imputer = SimpleImputer(strategy="most_frequent")
    data[categorical_columns] = cat_imputer.fit_transform(data[categorical_columns])

    logger.info("Missing values handled successfully.")
    return data


# Function to remove duplicate rows
def remove_duplicate_values(data: pd.DataFrame) -> pd.DataFrame:
    logger.info("Removing duplicate values...")
    data = data.drop_duplicates()
    logger.info("Duplicate values removed successfully.")
    return data


# Function to encode categorical features using one-hot encoding
def encode_categorical_features(data: pd.DataFrame) -> pd.DataFrame:
    logger.info("Encoding categorical features...")

    categorical_columns = data.select_dtypes(include=['object']).columns
    encoder = ColumnTransformer(
        transformers=[('categorical', OneHotEncoder(drop='first', sparse_output=False), categorical_columns)], 
        remainder='passthrough'
    )
    data_encoded = encoder.fit_transform(data)

    # Get new column names after encoding
    encoded_columns = encoder.transformers_[0][1].get_feature_names_out(categorical_columns)
    all_columns = list(encoded_columns) + list(data.select_dtypes(exclude=['object']).columns)

    # Convert the encoded data back to a DataFrame with appropriate column names
    data_encoded_df = pd.DataFrame(data_encoded, columns=all_columns)

    logger.info("Categorical features encoded successfully.")
    return data_encoded_df


# Function to remove highly correlated features
def remove_highly_correlated_features(data: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    logger.info(f"Removing highly correlated features with correlation threshold: {threshold}...")

    # Compute the correlation matrix for numerical features
    corr_matrix = data.corr(numeric_only=True).abs()

    # Identify the upper triangle of the correlation matrix to avoid redundant comparisons
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = []

    # Iterate through the upper triangle to find correlated features
    for column in upper_triangle.columns:
        # Find features that have correlation higher than the threshold
        correlated_features = upper_triangle[column][upper_triangle[column] > threshold].index
        for feature in correlated_features:
            # Add one of the correlated features to the list to drop (either column or feature)
            if column not in to_drop:
                to_drop.append(column)

    # Remove duplicates from the list of features to drop
    to_drop = list(set(to_drop))

    logger.info(f"Features to be removed due to high correlation: {to_drop}")

    # Drop the identified highly correlated features
    data = data.drop(columns=to_drop)

    logger.info("Highly correlated features removed successfully.")

    return data


# Function to perform feature engineering
def feature_engineering(data: pd.DataFrame, original_columns: list) -> pd.DataFrame:
    logger.info("Starting feature engineering pipeline...")

    # Perform missing value handling, duplicate removal, encoding, and correlation removal
    data = handle_missing_values(data)
    data = remove_duplicate_values(data)
    data = encode_categorical_features(data)
    data = remove_highly_correlated_features(data)

    # Update original_columns after modifications
    updated_columns = [col for col in original_columns if col not in ['isFraud', 'nameOrig', 'nameDest', 'isFlaggedFraud']]

    logger.info("Feature engineering completed successfully.")
    return data, updated_columns


# Main execution
if __name__ == "__main__":
    file_path = 'data/Imported_data.csv'  # Path to your dataset
    output_path = 'data/preprocessed_data.csv'  # Path for saving preprocessed data

    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Separate target column and original column names
    target = data['isFraud']
    original_columns = data.columns.tolist()
    
    # Drop irrelevant columns and prepare the data
    data = data.drop(columns=['isFraud', 'nameOrig', 'nameDest', 'isFlaggedFraud'])

    # Ensure there are no NaN values in data or target before proceeding
    if data.isnull().any().any():
        logger.error("Data contains NaN values after preprocessing! ")
        # Handle NaN values as needed (e.g., retry imputation)
    
    if target.isnull().any():
        logger.error("Target contains NaN values! ")
        # Handle NaN values in target as needed

    # Perform feature engineering
    processed_data, updated_columns = feature_engineering(data, original_columns)

    # Add the target column back to the processed data
    processed_data['isFraud'] = target

    # Save the cleaned data to the specified output path
    if os.path.exists(output_path):
        logger.info(f"{output_path} exists. Replacing the file...")
        os.remove(output_path)  # Remove the existing file

    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Create directory if not exists
    processed_data.to_csv(output_path, index=False, header=True)
    logger.info("Preprocessed data saved successfully.")
