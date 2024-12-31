import os
import pandas as pd
from src.logging.logger import logger
from src.utils.data_utils import save_features
from src.preprocessing.feature_engineering import feature_engineering
from src.preprocessing.feature_selection import feature_selection_and_scaling

# Main data transformation pipeline
def data_transformation_pipeline(file_path: str, output_path: str, num_features: int = 8) -> None:
    """
    A pipeline that performs data transformation by combining feature engineering and feature selection with scaling.
    
    Parameters:
        file_path (str): Path to the input dataset.
        output_path (str): Path to save the transformed dataset.
        num_features (int): Number of top features to select (default is 8).
    """
    logger.info("Starting data transformation pipeline...")

    # Load the dataset
    data = pd.read_csv(file_path)
    logger.info("Data loaded successfully.")

    # Separate target column and original column names
    target = data['isFraud']
    original_columns = data.columns.tolist()

    # Drop irrelevant columns and prepare the data for feature engineering
    data = data.drop(columns=['isFraud', 'nameOrig', 'nameDest', 'isFlaggedFraud'])

    # Ensure no missing values exist in data or target
    if data.isnull().any().any():
        logger.error("Data contains NaN values after preprocessing!")
        # Handle NaN values as needed (retry imputation or skip)
    
    if target.isnull().any():
        logger.error("Target contains NaN values!")
        # Handle NaN values in target as needed

    # Step 1: Perform feature engineering (missing values handling, encoding, etc.)
    processed_data, updated_columns = feature_engineering(data, original_columns)
    logger.info("Feature engineering completed successfully.")

    # Step 2: Perform feature selection and scaling
    transformed_data = feature_selection_and_scaling(processed_data, target, num_features=num_features)
    logger.info("Feature selection and scaling completed successfully.")

    # Add the target column back to the transformed data
    transformed_data['isFraud'] = target

    # Save the transformed data to the specified output path
    if os.path.exists(output_path):
        logger.info(f"{output_path} exists. Replacing the file...")
        os.remove(output_path)  # Remove the existing file

    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Create directory if not exists
    transformed_data.to_csv(output_path, index=False, header=True)
    logger.info(f"Transformed data saved to {output_path} successfully.")

    # Optionally, save the features to the artifacts folder
    save_features(transformed_data, file_name="Transformed_features_data.csv", folder_name='artifacts')
    logger.info("Data transformation pipeline completed successfully.")

# Example usage
if __name__ == "__main__":
    input_file_path = 'data/Imported_data.csv'  # Path to your raw dataset
    output_file_path = 'data/transformed_data.csv'  # Path to save the transformed data
    
    # Run the data transformation pipeline
    data_transformation_pipeline(input_file_path, output_file_path, num_features=8)
