import logging
from pathlib import Path
from time import strftime


import logging
from pathlib import Path
from time import strftime

def create_logger(
    logger_name="main",
    date=None,
    time=None,
    date_suffix=None,
    path_logs=None,
    **kwargs,
):
    """
    Create and configure a logger with both console and file handlers.
    
    Args:
        logger_name (str): Base name for the logger.
        date (str): Date for organizing log files, defaults to current date.
        time (str): Time for organizing log files, defaults to current time.
        date_suffix (str): Optional suffix to append to the date.
        path_logs (str): Base directory for logs. If not provided, uses a relative 'logs' directory.
        
    Returns:
        tuple: Tuple containing two logger objects (general and file-specific) and the log directory path.
    """
    # Default date and time
    if date is None:
        date = strftime("%Y%m%d")
    if time is None:
        time = strftime("%Hh%Mmin%Ss")
    
    # Apply date suffix if provided
    if date_suffix:
        date += f"_{date_suffix}"
    
    # Determine the final log directory path
    if path_logs is None:
        path_logs = Path("logs") / date / time
    else:
        path_logs = Path(path_logs) / date / time
    path_logs.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = logging.getLogger(f"{logger_name}_all")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers

    logger_file = logging.getLogger(f"{logger_name}_file")
    logger_file.setLevel(logging.INFO)
    logger_file.handlers = []

    formatter = logging.Formatter("[%(levelname)s] [%(asctime)s] %(message)s")
    
    # File handler
    file_handler = logging.FileHandler(path_logs / "log.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger_file.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger, logger_file, path_logs
