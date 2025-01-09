from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
from src.logging.logger import logger
from src.utils.data_utils import save_features

# Function for feature selection using SelectKBest with ANOVA F-value
def feature_selection(data: pd.DataFrame, target: pd.Series, num_features: int = 5) -> pd.DataFrame:
    """
    Selects the most important features using SelectKBest with ANOVA F-value.
    
    Parameters:
        data (pd.DataFrame): Input feature data.
        target (pd.Series): Target variable.
        num_features (int): Number of top features to select (default is 5).
        
    Returns:
        pd.DataFrame: Data with selected top features.
    """
    logger.info("Selecting features using SelectKBest with ANOVA F-value...")

    # Initialize SelectKBest with f_classif (ANOVA F-value) as the score function
    selector = SelectKBest(score_func=f_classif, k=num_features)

    # Fit and transform the data to select the top features
    data_selected = selector.fit_transform(data, target)
    
    # Get the selected feature names
    selected_features = data.columns[selector.get_support()]
    
    logger.info(f"Feature selection completed. Selected columns: {list(selected_features)}")
    
    # Return the data with the selected features only
    data_selected = pd.DataFrame(data_selected, columns=selected_features)
    
    return data_selected


# Define the path for saving the processed data
if __name__ == "__main__":
    file_path = 'data/preprocessed_data.csv'  # Path to your dataset

    # Load the dataset
    data = pd.read_csv(file_path)

    target = data['isFraud']  # Extract the target column
    data = data.drop(columns=['isFraud', 'type_PAYMENT'])  # Drop the target column from the features

    # Perform feature selection
    processed_data = feature_selection(data, target, num_features=5)

    # Add the target column back to the processed data
    processed_data['isFraud'] = target

    # Save the processed data to the artifacts folder using the save_features utility
    save_features(processed_data, file_name="Important_features_data.csv", folder_name='artifacts/data')

    logger.info("Feature selection completed successfully.")
