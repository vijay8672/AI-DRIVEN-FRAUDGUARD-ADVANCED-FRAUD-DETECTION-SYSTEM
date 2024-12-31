from src.components.data_ingestion import DataIngestion  # Import the DataIngestion class
from src.logging.logger import logger

def run_data_ingestion_pipeline():
    try:
        # Step 1: Initialize the DataIngestion class using environment variables
        ingestion = DataIngestion.from_env()
        
        # Step 2: Call the ingest_data method to load and save data
        logger.info("Starting the data ingestion process...")
        ingestion.ingest_data()
        
        # Step 3: Confirm the successful completion of the data ingestion process
        logger.info("Data ingestion completed successfully.")
    
    except Exception as e:
        logger.error(f"An error occurred in the data ingestion pipeline: {str(e)}")
        raise  # Reraise the exception after logging


if __name__ == "__main__":
    # Run the data ingestion pipeline
    run_data_ingestion_pipeline()
