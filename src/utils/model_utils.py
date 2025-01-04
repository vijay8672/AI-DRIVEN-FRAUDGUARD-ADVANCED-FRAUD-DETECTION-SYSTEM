import os
import joblib
from src.logging.logger import logger

def save_model(model, model_name, save_dir="artifacts/models/"):
    """
    Saves the trained model to a specified directory with the given model name.

    Parameters:
        model (object): The trained model to be saved.
        model_name (str): The name of the model (used in the saved file name).
        save_dir (str): The directory where the model will be saved. Defaults to "artifacts/models/".

    Returns:
        str: The full path of the saved model file.
    """
    if not model_name:
        raise ValueError("Model name cannot be empty.")
    
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{model_name}.pkl")
    joblib.dump(model, file_path)
    logger.info(f"Model saved: {file_path}")
    return file_path




def load_model(model_name, model_dir="artifacts/models/"):
    """
    Loads a trained model from a specified directory using the model's name.

    Parameters:
        model_name (str): The name of the model to be loaded (should match the saved model's file name).
        model_dir (str): The directory where the model is stored. Defaults to "artifacts/models/".

    Returns:
        object: The loaded model.
    
    Raises:
        FileNotFoundError: If the model file does not exist in the specified directory.
    """
    if not model_name:
        raise ValueError("Model name cannot be empty.")
    
    # Ensure the directory and file name are properly joined
    file_path = os.path.join(model_dir, model_name)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file {file_path} not found.")
    
    # Load the model using joblib
    model = joblib.load(file_path)
    logger.info(f"Model loaded: {file_path}")
    
    return model


# Utility function to save artifacts
def save_artifact(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)
    logger.info(f"Artifact saved at {path}")
