import json
import yaml

class Config:
    """
    Configuration settings for the recommendation system.

    This class stores paths to data files, model parameters, and other global settings.
    It also allows loading settings from JSON or YAML files.
    """

    # Paths to data files
    USER_DATA_PATH = '../datasets/users.csv'
    ITEM_DATA_PATH = '../datasets/movies_series_dataset.csv'

    # Number of top recommendations
    TOP_N_RECOMMENDATIONS = 100
    
    def __init__(self, config_path=None, config_type='json'):
        """
        Initializes the configuration class, potentially loading settings from a file.

        Args:
            config_path (str): Path to the configuration file (JSON or YAML).
            config_type (str): Type of configuration file ('json' or 'yaml').
        """
        # Default settings
        self.COLLABORATIVE_WEIGHT = 0.6
        self.CONTENT_WEIGHT = 0.4
        self.TEST_SIZE = 0.2
        self.RANDOM_STATE = 42
        self.RECOMMENDATION_THRESHOLD = 0.5

        # Advanced model parameters
        self.REGULARIZATION = 0.01
        self.MAX_ITER = 100
        
        # Logging settings
        self.LOGGING_LEVEL = 'INFO'
        self.LOG_FILE = 'logs/recommendation_system.log'
        
        # Load settings from file if provided
        if config_path:
            self.load_config_from_file(config_path, config_type)

    def load_config_from_file(self, config_path, config_type='json'):
        """
        Loads configuration settings from a JSON or YAML file.

        Args:
            config_path (str): Path to the configuration file.
            config_type (str): Type of configuration file ('json' or 'yaml').
        """
        with open(config_path, 'r') as file:
            if config_type == 'json':
                config = json.load(file)
            elif config_type == 'yaml' or config_type == 'yml':
                config = yaml.safe_load(file)
            else:
                raise ValueError("Unsupported config file type. Use 'json' or 'yaml'.")
        
        # Update the class attributes with values from the config file
        self.__dict__.update(config)

    def __repr__(self):
        """
        Returns a string representation of the configuration for easy inspection.
        """
        return f"Config({self.__dict__})"
