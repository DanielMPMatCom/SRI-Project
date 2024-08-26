import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    """
    A class for preprocessing user and item data.

    This class handles tasks such as cleaning, feature engineering, and splitting data into
    training and test sets.

    Attributes:
        scaler (StandardScaler): An instance of the StandardScaler for normalizing data.
    """
    
    def __init__(self):
        """
        Initializes the DataPreprocessor with a scaler for data normalization.
        """
        self.scaler = StandardScaler()
    
    def clean_data(self, data):
        """
        Cleans the input data by handling missing values and outliers.

        Args:
            data (pd.DataFrame): The data to be cleaned.

        Returns:
            pd.DataFrame: The cleaned data.
        """
        # Handle missing values by dropping rows with any missing values
        data = data.dropna()

        # Outlier detection and removal (if necessary)
        # Example: Remove rows where numerical features are beyond 3 standard deviations
        for col in data.select_dtypes(include=['float64', 'int64']).columns:
            upper_bound = data[col].mean() + 3 * data[col].std()
            lower_bound = data[col].mean() - 3 * data[col].std()
            data = data[(data[col] <= upper_bound) & (data[col] >= lower_bound)]

        return data

    def feature_engineering(self, data):
        """
        Applies feature engineering techniques to enhance the dataset.

        Args:
            data (pd.DataFrame): The data to be transformed.

        Returns:
            pd.DataFrame: The transformed data with additional features.
        """
        # Example feature engineering: Add interaction terms, polynomial features, etc.
        # For categorical data, create dummy variables
        data = pd.get_dummies(data, drop_first=True)

        # Example: Create a feature for user-item interaction (if applicable)
        if 'user_id' in data.columns and 'item_id' in data.columns:
            data['user_item_interaction'] = data['user_id'].astype(str) + '_' + data['item_id'].astype(str)
        
        return data

    def split_data(self, data, target_column, test_size=0.2, random_state=42):
        """
        Splits the data into training and testing sets.

        Args:
            data (pd.DataFrame): The data to be split.
            target_column (str): The name of the target variable in the dataset.
            test_size (float): The proportion of the data to be used as test data.
            random_state (int): The seed used by the random number generator.

        Returns:
            tuple: A tuple containing the training features, testing features, training targets, and testing targets.
        """
        X = data.drop(columns=[target_column])
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

    def normalize_data(self, data):
        """
        Normalizes the input data using standard scaling.

        Args:
            data (pd.DataFrame): The data to be normalized.

        Returns:
            pd.DataFrame: The normalized data.
        """
        data_scaled = self.scaler.fit_transform(data)
        return pd.DataFrame(data_scaled, columns=data.columns)
