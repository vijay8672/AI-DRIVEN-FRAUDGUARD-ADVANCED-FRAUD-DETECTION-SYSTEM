from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
import os
from src.logging.logger import logger
from src.utils.data_utils import save_features


# Function for feature selection using Random Forest
def feature_selection(data: pd.DataFrame, target: pd.Series, num_features: int = 8) -> pd.DataFrame:
    """
    Selects the most important features using a Random Forest Classifier.
    
    Parameters:
        data (pd.DataFrame): Input feature data.
        target (pd.Series): Target variable.
        num_features (int): Number of top features to select (default is 8).
        
    Returns:
        pd.DataFrame: Data with selected top features.
    """
    logger.info("Selecting features using Random Forest Classifier...")

    # Initialize Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    
    # Fit the Random Forest model
    rf_classifier.fit(data, target)
    
    # Get feature importance from the model
    feature_importances = rf_classifier.feature_importances_
    
    # Create a DataFrame with feature names and their corresponding importance
    feature_importance_df = pd.DataFrame({
        'Feature': data.columns,
        'Importance': feature_importances
    })
    
    # Sort the features by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    # Select the most important features based on `num_features`
    top_features = feature_importance_df.head(num_features)['Feature'].values
    
    # Filter the data based on the selected features
    data_selected = data[top_features]
    
    logger.info(f"Feature selection completed. Selected columns: {list(top_features)}")
    
    return data_selected


# Function to scale numerical features using RobustScaler
def scale_numerical_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Scales numerical features using RobustScaler to handle outliers effectively.
    
    Parameters:
        data (pd.DataFrame): Input data.
        
    Returns:
        pd.DataFrame: Data with scaled numerical features.
    """
    logger.info("Scaling numerical features...")

    # Identify numerical columns
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    
    # Initialize RobustScaler
    scaler = RobustScaler()
    
    # Apply scaling to numerical columns
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    
    logger.info("Numerical features scaled successfully.")
    
    return data


# Function to combine feature selection and scaling
def feature_selection_and_scaling(data: pd.DataFrame, target: pd.Series, num_features: int = 8) -> pd.DataFrame:
    """
    Combines feature selection and scaling into a single pipeline.
    
    Parameters:
        data (pd.DataFrame): Input data.
        target (pd.Series): Target variable.
        num_features (int): Number of top features to select (default is 8).
        
    Returns:
        pd.DataFrame: Processed data with selected and scaled features.
    """
    logger.info("Starting feature selection and scaling pipeline...")

    # Perform feature selection
    data_selected = feature_selection(data, target, num_features=num_features)
    
    # Scale the selected features
    data_scaled = scale_numerical_features(data_selected)
    
    logger.info("Feature selection and scaling completed successfully.")
    
    return data_scaled




# Define the path for saving the processed data
if __name__ == "__main__":
    file_path = 'data/preprocessed_data.csv'  # Path to your dataset

    # Load the dataset
    data = pd.read_csv(file_path)

    target = data['isFraud']  # Extract the target column
    data = data.drop(columns=['isFraud'])  # Drop the target column from the features

    # Perform feature selection and scaling
    processed_data = feature_selection_and_scaling(data, target, num_features=8)

    # Add the target column back to the processed data
    processed_data['isFraud'] = target

    # Save the processed data to the artifacts folder using the save_features utility
    save_features(processed_data, file_name="Important_features_data.csv", folder_name='artifacts')


    logger.info("Feature selection and scaling pipeline completed successfully.")
