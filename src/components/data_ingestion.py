import os
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import pandas as pd
from src.logging.logger import logger

# Load environment variables
load_dotenv()

class DataIngestion:
    def __init__(self, connection_string: str, container_name: str, blob_name: str):
        self.connection_string = connection_string
        self.container_name = container_name
        self.blob_name = blob_name
        
        if not all([self.connection_string, self.container_name, self.blob_name]):
            logger.error("Missing required environment variables. Please check your .env file.")
            raise ValueError("Missing required environment variables.")
        
    @classmethod
    def from_env(cls):
        """
        Creates an instance of DataIngestion using environment variables.
        """
        connection_string = os.getenv('AZURE_CONNECTION_STRING')
        container_name = os.getenv('CONTAINER_NAME')
        blob_name = os.getenv('BLOB_NAME')
        
        # Call the constructor with the environment variables
        return cls(connection_string, container_name, blob_name)
        
    def ingest_data(self):
        try:
            blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
            blob_client = blob_service_client.get_blob_client(container=self.container_name, blob=self.blob_name)

            download_stream = blob_client.download_blob()
            data = pd.read_csv(download_stream)
            logger.info(f"Data loaded successfully. Shape: {data.shape}")

            folder_path = 'dummy_data'

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            local_file_path = os.path.join(folder_path, 'dump_data.csv')
            data.to_csv(local_file_path, index=False)
            logger.info(f"Data saved to {local_file_path} successfully.")

        except Exception as e:
            logger.error(f"Error occurred: {str(e)}")
            raise


if __name__ == "__main__":
    # Automatically create the DataIngestion instance using environment variables
    ingestion = DataIngestion.from_env()

    # Call the ingest_data method
    ingestion.ingest_data()
