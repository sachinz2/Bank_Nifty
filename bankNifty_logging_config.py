"""
Centralized logging configuration for the trading application.
"""
import logging
import sys
from logging.handlers import TimedRotatingFileHandler

LOG_FILE = "logs/trading_system.log"

def setup_logging(log_level=logging.INFO):
    """
    Configures logging to both console and a rotating file.
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Prevent adding handlers multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    file_handler = TimedRotatingFileHandler(LOG_FILE, when="midnight", interval=1, backupCount=7)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    return logger