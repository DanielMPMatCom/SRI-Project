import pandas as pd

class DataLoader:
    """
    A class for loading and preprocessing user and item data.

    This class handles the loading of data from different sources and ensures that the data is cleaned
    and prepared for use by the recommendation system.

    Attributes:
        user_data_path (str): Path to the user data file.
        item_data_path (str): Path to the item data file.
        file_type (str): The file type of the data (e.g., 'csv', 'excel').
    """
    
    def __init__(self, user_data_path, item_data_path, file_type='csv'):
        """
        Initializes the DataLoader with paths to user and item data files.

        Args:
            user_data_path (str): The file path to the user data.
            item_data_path (str): The file path to the item data.
            file_type (str): The type of files to load ('csv', 'excel', etc.).
        """
        self.user_data_path = user_data_path
        self.item_data_path = item_data_path
        self.file_type = file_type.lower()
    
    def load_data(self):
        """
        Loads user and item data from the specified files.

        Returns:
            tuple: A tuple containing user data and item data as pandas DataFrames.
        """
        if self.file_type == 'csv':
            user_data = pd.read_csv(self.user_data_path)
            item_data = pd.read_csv(self.item_data_path)
        elif self.file_type == 'excel':
            user_data = pd.read_excel(self.user_data_path)
            item_data = pd.read_excel(self.item_data_path)
        else:
            raise ValueError(f"Unsupported file type: {self.file_type}")

        # Optional: Perform initial data validation
        self._validate_data(user_data, "User Data")
        self._validate_data(item_data, "Item Data")
        
        return user_data, item_data
    
    def _validate_data(self, data, data_name):
        """
        Performs basic validation checks on the loaded data.

        Args:
            data (pd.DataFrame): The data to validate.
            data_name (str): A name for the data being validated (e.g., 'User Data').

        Raises:
            ValueError: If validation checks fail.
        """
        if data.empty:
            raise ValueError(f"{data_name} is empty.")
        
        # Check for any duplicate rows
        if data.duplicated().any():
            raise ValueError(f"{data_name} contains duplicate rows.")

        # Optional: Check for expected columns (this is context-specific)
        # Example:
        # expected_columns = ['user_id', 'age', 'gender']  # Customize as needed
        # if not all(col in data.columns for col in expected_columns):
        #     raise ValueError(f"{data_name} is missing expected columns.")

        # Add any additional validation rules as needed

