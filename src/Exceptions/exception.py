import sys
from src.logging.logger import logger  # Import the logger correctly

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        # Extracting the exception info
        _, _, exc_tb = error_detail.exc_info()

        # Get the line number and file name from the traceback object
        self.line_no = exc_tb.tb_lineno if exc_tb else None
        self.file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else None

        # Store the error message
        self.error_message = error_message

        # Log the error details
        logger.error(self.__str__())

    def __str__(self):
        # Formatting the error message
        return "Error occurred in python script name [{0}] line number [{1}] error message detail [{2}]".format(
            self.file_name, self.line_no, str(self.error_message)
        )


if __name__ == "__main__":
    try:
        logger.info("Entering the try block")  # Log an informational message
        a = 1 / 0  # This will raise a ZeroDivisionError
        print("Zero Division Error", a)

    except Exception as e:
        raise CustomException(e, sys)
