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
