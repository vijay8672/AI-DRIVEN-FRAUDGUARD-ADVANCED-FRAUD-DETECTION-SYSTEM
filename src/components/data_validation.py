import pandas as pd
import os
from src.logging.logger import logger

def validate_data(data: pd.DataFrame):
    """
    Validate the ingested data for various checks such as missing values, duplicates, data types, etc.
    
    Parameters:
    - data (pd.DataFrame): The data to validate.
    
    Returns:
    - bool: True if the data passes validation, False otherwise.
    """
    
    logger.info("Starting data validation...")

    # 1. Check for missing values
    missing_values = data.isnull().sum()
    if missing_values.any():
        logger.warning(f"Missing values found:\n{missing_values[missing_values > 0]}")
    else:
        logger.info("No missing values found.")

    # 2. Check for duplicates
    duplicates = data.duplicated().sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate rows in the data.")
    else:
        logger.info("No duplicate rows found.")

    # 3. Validate data types (Example: Ensure 'isFraud' is a boolean, etc.)
    expected_dtypes = {
        'step':'int64',
        'type':'object',
        'amount':'float64',
        'nameOrig':'object',
        'oldbalanceOrg':'float64',
        'newbalanceOrig':'float64',
        'nameDest':'object', 
        'oldbalanceDest':'float64',
        'newbalanceDest':'float64',
        'isFraud': 'int64',
        'isFlaggedFraud': 'int64',
    }

    for column, dtype in expected_dtypes.items():
        if column in data.columns:
            actual_dtype = data[column].dtype
            if actual_dtype != dtype:
                logger.warning(f"Column '{column}' has incorrect data type. Expected {dtype}, found {actual_dtype}.")
            else:
                logger.info(f"Column '{column}' has the correct data type: {dtype}.")
        else:
            logger.warning(f"Column '{column}' is missing from the data.")

    # 4. Check for specific conditions (e.g., no negative values in 'amount')
    if 'amount' in data.columns:
        negative_values = data[data['amount'] < 0]
        if not negative_values.empty:
            logger.warning(f"Found {len(negative_values)} rows with negative values in 'amount'.")
        else:
            logger.info("No negative values found in 'amount'.")

    # 5. Validate data range (if applicable)
    if 'isFraud' in data.columns:
        invalid_isFraud = data[~data['isFraud'].isin([0, 1])]  # Corrected condition
        if not invalid_isFraud.empty:
            logger.warning(f"Found {len(invalid_isFraud)} rows with invalid 'isFraud' values.")
        else:
            logger.info("All 'isFraud' values are within a valid range.")

    # 6. Final validation check
    if missing_values.any() | duplicates > 0 | any(data.dtypes != expected_dtypes.values()):
        logger.error("Data validation failed.")
        return False
    else:
        logger.info("Data validation passed.")
        return True


def load_and_validate_data(file_path: str):
    """
    Load the data from a local CSV file and validate it.
    
    Parameters:
    - file_path (str): Path to the local CSV file.
    
    Returns:
    - bool: True if the data is valid, False otherwise.
    """
    
    if not os.path.exists(file_path):
        logger.error(f"The file does not exist at the specified path: {file_path}")
        return False

    # Load data
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully. Shape: {data.shape}")
    except Exception as e:
        logger.error(f"Error loading the data from {file_path}: {str(e)}")
        return False

    # Validate data
    return validate_data(data)


# Example of how you can call the function:
file_path = 'data/Imported_data.csv'  # Path to the ingested file
validation_result = load_and_validate_data(file_path)

if validation_result:
    logger.info("Data is ready for further processing.")
else:
    logger.error("Data validation failed. Please check the logs for details.")
