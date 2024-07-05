from time import strftime
from .utils.logs import create_logger 

def init_logger(date_suffix=None, **kwargs): 
    date = strftime("%Y%m%d")
    time = strftime("%Hh%Mmin%Ss")
    logger, logger_file, path_logs, final_dest = create_logger(
        logger_name="main", date=date, time=time, date_suffix=date_suffix, **kwargs
    )
    return logger, date, time, path_logs, final_dest 