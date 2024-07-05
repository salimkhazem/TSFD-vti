import logging 
from pathlib import Path 
from time import strftime

def create_logger(logger_name="main", date=strftime("%Y%m%d"),
                  time=strftime("%Hh%Mmin%Ss"), date_suffix=None, 
                  path_logs=None, **kwargs): 
    """
    create the logger object based on date and time 
    It will generate 2 loggers: 
    - one for the console 
    - one for the file 
    """
    logger = logging.getLogger(f"{logger_name}_all") 
    logger.handlers.clear() 
    logger_file = logging.getLogger(f"{logger_name}_file") 
    logger_file.handlers.clear()
    del logger, logger_file 

    if isinstance(path_logs, str): 
        date += f"_{date_suffix}" if date_suffix else ""

    final_dest = None 
    if path_logs is not None: 
        path_logs = Path(path_logs).resolve().absolute() 
        path_logs = (Path(__file__).resolve().parent[2].absolute()
                     / "logs" / date / time) 
    path_logs.mkdir(parents=True, exist_ok=True) 

    # Loggers 
    logger = logging.getLogger(f"{logger_name}_all") 
    logger.setLevel(logging.INFO) 
    logger_file = logging.getLogger(f"{logger_name}_file") 
    logger_file.setLevel(logging.INFO) 

    # Formatters 
    formatter_str = "[%(levelname)s] [%(asctime)s] %(message)s" 
    formatter = logging.Formatter(formatter_str) 

    # Handlers
    file_handler = logging.FileHandler(path_logs / "log.log", encoding="utf-8")
    logger.addHandler(file_handler)
    logger_file.addHandler(file_handler) 

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger, logger_file, path_logs, final_dest