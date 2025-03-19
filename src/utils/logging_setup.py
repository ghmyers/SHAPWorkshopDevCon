
import logging
import os

def setup_logging(log_dir="logs", log_filename="project.log"):

    # Dynamically determine the project root
    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    except NameError:
        project_root = os.getcwd()  # Fallback if __file__ is not available

    # Ensure logs directory exists
    log_dir = os.path.join(project_root, log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Define log file path
    log_path = os.path.join(log_dir, log_filename)

    # Prevent duplicate handlers
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            filename=log_path,
            filemode="a",
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO
        )

    return logging.getLogger("project")

# Example usage
if __name__ == "__main__":
    logger = setup_logging()
    logger.info("Logging setup is working!")

