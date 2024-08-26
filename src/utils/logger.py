import logging
import os

class Logger:
    """
    A class to set up and manage logging for the recommendation system.

    This class provides functionality to create a logger that writes logs to a file
    and optionally to the console.

    Attributes:
        logger_name (str): The name of the logger.
        log_file (str): The path to the log file.
        log_level (str): The logging level (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
    """
    
    def __init__(self, logger_name='RecommendationSystem', log_file='logs/recommendation_system.log', log_level='INFO'):
        """
        Initializes the Logger with a specified name, log file, and log level.

        Args:
            logger_name (str): The name of the logger.
            log_file (str): The path to the log file.
            log_level (str): The logging level (default is 'INFO').
        """
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(self._get_log_level(log_level))
        
        # Create log directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(self._get_log_level(log_level))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self._get_log_level(log_level))
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _get_log_level(self, level_str):
        """
        Converts a string representation of the logging level to a logging constant.

        Args:
            level_str (str): The string representation of the logging level.

        Returns:
            int: The logging constant corresponding to the level string.
        """
        level_str = level_str.upper()
        return {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }.get(level_str, logging.INFO)

    def get_logger(self):
        """
        Returns the logger instance.

        Returns:
            logging.Logger: The logger instance.
        """
        return self.logger

# Example usage:
if __name__ == "__main__":
    # Initialize logger
    logger_instance = Logger(log_level='DEBUG').get_logger()
    
    # Example log messages
    logger_instance.debug("This is a debug message.")
    logger_instance.info("This is an info message.")
    logger_instance.warning("This is a warning message.")
    logger_instance.error("This is an error message.")
    logger_instance.critical("This is a critical message.")
