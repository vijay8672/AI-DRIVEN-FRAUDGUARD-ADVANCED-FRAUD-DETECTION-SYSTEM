from src.components.model_training import Model_Training
from src.logging.logger import logger
from src.training.model_training import Model_Training

def Model_Training_Pipeline():
    try:
        logger.info("Starting the model training pipeline...")
        Model_Training()
        logger.info("Model training pipeline completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during the model training pipeline: {str(e)}", exc_info=True)

if __name__ == "__main__":
    Model_Training_Pipeline()
