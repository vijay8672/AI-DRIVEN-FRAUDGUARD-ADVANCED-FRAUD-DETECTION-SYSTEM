from src.model_evaluation import Model_Evaluation
from src.logging.logger import logger

def Model_Evaluation_Pipeline():
    try:
        # Log the start of the evaluation pipeline
        logger.info("Starting the Model Evaluation Pipeline...")

        # Start the evaluation process
        Model_Evaluation()

        # Log the completion of the pipeline
        logger.info("Model Evaluation Pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Error in Model Evaluation Pipeline: {str(e)}", exc_info=True)

if __name__ == "__main__":
    Model_Evaluation_Pipeline()
