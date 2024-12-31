import os
import pandas as pd
from src.logging.logger import logger

def save_features(data: pd.DataFrame, file_name: str, folder_name: str) -> None:
    """
    Saves the given DataFrame into the specified folder as a CSV file.
    
    Parameters:
        data (pd.DataFrame): The DataFrame to save.
        file_name (str): The name of the file (e.g., "selected_features.csv").
        folder_name (str): The folder where the file will be saved (default is "artifacts").
    """
    try:
        # Ensure the folder exists
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            logger.info(f"Folder '{folder_name}' created.")  # Optional log, can be omitted

        # Define file path
        file_path = os.path.join(folder_name, file_name)
        
        # Remove the file if it already exists
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"File '{file_name}' already exists and has been removed.") 

        # Save the DataFrame as a CSV file
        data.to_csv(file_path, index=False)
        logger.info(f"File '{file_name}' saved successfully in '{folder_name}'.")
    except Exception as e:
        logger.error(f"Error saving file '{file_name}' in folder '{folder_name}': {e}")
