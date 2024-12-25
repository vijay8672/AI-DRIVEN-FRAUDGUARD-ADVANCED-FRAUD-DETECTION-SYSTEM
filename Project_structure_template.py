import os
from pathlib import Path
import logging

# Set up logging with a more descriptive format
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] - %(levelname)s - %(message)s',
)

# List of files and directories to create
FILES_TO_CREATE = [
    ".github/workflows/.gitkeep",
    "src/components/__init__.py",
    "src/utils/__init__.py",
    "src/utils/common.py",
    "src/logging/__init__.py",
    "src/config/__init__.py",
    "src/config/configuration.py",
    "src/pipeline/__init__.py",
    "src/entity/__init__.py",
    "src/constants/__init__.py",
    "config/config.yaml",
    "params.yaml",
    "Dockerfile",
    "setup.py",
    "research/research.ipynb"
]


def create_directory(directory: Path):
    """Create directory if it does not exist."""
    if not directory.exists():
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logging.info(f"Directory created: {directory}")
        except Exception as e:
            logging.error(f"Failed to create directory {directory}: {str(e)}")
            raise


def create_file(file_path: Path):
    """Create an empty file if it does not exist."""
    if not file_path.exists():
        try:
            file_path.touch()
            logging.info(f"Empty file created: {file_path}")
        except Exception as e:
            logging.error(f"Failed to create file {file_path}: {str(e)}")
            raise
    else:
        logging.info(f"File already exists: {file_path}")


def create_project_structure(files: list):
    """Create the project structure by creating necessary directories and files."""
    for file_path in files:
        path = Path(file_path)
        directory = path.parent

        if directory:
            create_directory(directory)

        create_file(path)


if __name__ == "__main__":
    try:
        logging.info("Starting project structure creation.")
        create_project_structure(FILES_TO_CREATE)
        logging.info("Project structure creation completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during project structure creation: {str(e)}")
