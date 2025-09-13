
import logging
import sys
from logging.handlers import TimedRotatingFileHandler
import os

def setup_logger(name='app_logger', level=logging.INFO, log_dir='logs'):
    """
    Create a logger with timestamp and level formatting that saves to both console and file
    Files are rotated daily and last 5 days are preserved
    
    Args:
        name (str): Logger name
        level: Logging level (logging.INFO, logging.WARNING, logging.CRITICAL)
        log_dir (str): Directory to save log files
    
    Returns:
        logging.Logger: Configured logger instance
    """
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        logger.handlers.clear()
    
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # File handler with daily rotation, keeping last 5 days
    log_file = os.path.join(log_dir, f'{name}.log')
    file_handler = TimedRotatingFileHandler(
        filename=log_file,
        when='midnight',
        interval=1,
        backupCount=5,  # Keep 5 days of logs
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add formatter to both handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logger()

if __name__ == "__main__":    
    # Example usage
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.critical("This is a critical message")