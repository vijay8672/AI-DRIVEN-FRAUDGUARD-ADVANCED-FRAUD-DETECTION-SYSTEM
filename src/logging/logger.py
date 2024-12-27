import os
import sys
import logging
from datetime import datetime

# Step 1: Define the directory where you want to save the log file
log_dir = "logs"

# Step 2: Define the logging format
logging_str = "%(asctime)s : %(levelname)s : %(module)s : %(message)s"

# Step 3: Create a unique log file name with an incremental number and timestamp
os.makedirs(log_dir, exist_ok=True)

# Get the current log file number
existing_logs = [f for f in os.listdir(log_dir) if f.startswith("logfile_") and f.endswith(".log")]
log_number = len(existing_logs) + 1  # Increment based on existing files

# Create the log file name with a number and timestamp
timestamp = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
log_file_name = f"logfile_{log_number}_{timestamp}.log"

# Full path for the log file
log_filepath = os.path.join(log_dir, log_file_name)

# Step 4: Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("summarizerlogger")

if __name__ == "__main__":
    print("This is a test message")
    logger.info("This is a test log message")
    logger.warning("This is a test warning message")
    logger.error("This is a test error message")
    logger.critical("This is a test critical message")
    logger.debug("This is a test debug message")
