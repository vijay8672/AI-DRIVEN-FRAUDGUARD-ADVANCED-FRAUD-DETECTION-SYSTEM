import os
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import pandas as pd
from src.logging.logger import logger

# Specify the path to your .env file in the config folder (outside of the src folder)
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', '.env')

# Check if the .env file exists
if not os.path.exists(dotenv_path):
    print(f"Error: The .env file does not exist at the path: {dotenv_path}")
else:
    print(f".env file found at {dotenv_path}")

# Load environment variables from .env file
load_dotenv(dotenv_path)

# Debugging: Print out the .env path and environment variables
print(f"Loading .env from: {dotenv_path}")
print(f"AZURE_CONNECTION_STRING: {os.getenv('AZURE_CONNECTION_STRING')}")
print(f"CONTAINER_NAME: {os.getenv('CONTAINER_NAME')}")
print(f"BLOB_NAME: {os.getenv('BLOB_NAME')}")

# Step 1: Retrieve environment variables from .env file
CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("CONTAINER_NAME")
BLOB_NAME = os.getenv("BLOB_NAME")

# Ensure all necessary environment variables are present
if not CONNECTION_STRING or not CONTAINER_NAME or not BLOB_NAME:
    logger.error("One or more required environment variables are missing. Exiting.")
    exit(1)

# Step 2: Initialize BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)

# Step 3: Get Blob Client
blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=BLOB_NAME)

# Step 4: Read Blob Data Directly into a DataFrame and Save Locally
try:
    # Downloading blob as a stream and reading directly into a pandas DataFrame
    download_stream = blob_client.download_blob()
    
    # Reading the stream directly into pandas DataFrame (assuming the file is a CSV)
    data = pd.read_csv(download_stream)
    logger.info(f"Data loaded successfully. Shape: {data.shape}")

    # Step 5: Save the DataFrame to a CSV file in the 'data' folder
    folder_path = 'data'  # Folder where you want to save the file

    # Check if the folder exists, if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  # Create the folder if it doesn't exist

    local_file_path = os.path.join(folder_path, 'Imported_data.csv')  # Save file inside the 'data' folder

    # Save DataFrame to a local CSV file
    data.to_csv(local_file_path, index=False)
    logger.info(f"Data saved to {local_file_path} successfully.")

except Exception as e:
    logger.error(f"Error occurred: {str(e)}")
    raise  # Re-raise the error after logging
