import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeFilteringRecommender:
    """
    A recommender system based on collaborative filtering.
    
    This class implements a collaborative filtering approach to generate recommendations
    for users based on their similarity to other users or the similarity between items.

    Attributes:
        user_data (dict): A dictionary containing user information and their interactions with items.
        item_data (dict): A dictionary containing item information.
        user_similarity_matrix (numpy.ndarray): A matrix containing similarity scores between users.
        item_similarity_matrix (numpy.ndarray): A matrix containing similarity scores between items.
    """
    
    def __init__(self, user_data, item_data):
        """
        Initializes the CollaborativeFilteringRecommender with user and item data.

        Args:
            user_data (dict): A dictionary with user interaction data.
            item_data (dict): A dictionary with item information.
        """
        self.user_data = user_data
        self.item_data = item_data
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
    
    def _build_user_item_matrix(self):
        """
        Builds a user-item interaction matrix from the user_data dictionary.

        Returns:
            numpy.ndarray: A matrix where rows represent users and columns represent items.
        """
        # Extract unique users and items
        users = list(self.user_data.keys())
        items = list(self.item_data.keys())
        
        # Initialize the matrix with zeros
        user_item_matrix = np.zeros((len(users), len(items)))
        
        # Fill the matrix with user-item interactions
        for i, user in enumerate(users):
            for item, rating in self.user_data[user].items():
                if item in items:
                    j = items.index(item)
                    user_item_matrix[i, j] = rating

        return user_item_matrix, users, items
    
    def fit(self, similarity_type='user'):
        """
        Fits the collaborative filtering model to the provided data.
        This method calculates the similarity matrix based on user or item similarities.

        Args:
            similarity_type (str): The type of similarity to use ('user' or 'item').
        """
        user_item_matrix, self.users, self.items = self._build_user_item_matrix()
        
        if similarity_type == 'user':
            # Calculate user similarity matrix using cosine similarity
            self.user_similarity_matrix = cosine_similarity(user_item_matrix)
        elif similarity_type == 'item':
            # Calculate item similarity matrix using cosine similarity
            self.item_similarity_matrix = cosine_similarity(user_item_matrix.T)
        else:
            raise ValueError("similarity_type must be either 'user' or 'item'")
    
    def _predict_user_based(self, user_id_index, user_item_matrix):
        """
        Predicts ratings for all items for a given user based on user similarity.

        Args:
            user_id_index (int): The index of the user in the user-item matrix.
            user_item_matrix (numpy.ndarray): The user-item interaction matrix.

        Returns:
            numpy.ndarray: Predicted ratings for all items for the given user.
        """
        # Weighted sum of other users' ratings
        weighted_sum = np.dot(self.user_similarity_matrix[user_id_index], user_item_matrix)
        
        # Sum of similarities (excluding the user itself)
        similarity_sum = np.sum(self.user_similarity_matrix[user_id_index]) - 1
        
        # Avoid division by zero
        if similarity_sum == 0:
            return weighted_sum / 1
        else:
            return weighted_sum / similarity_sum
    
    def recommend(self, user_id, top_n=10):
        """
        Generates recommendations for a given user based on the collaborative filtering model.

        Args:
            user_id (str): The ID of the user to generate recommendations for.
            top_n (int): The number of top recommendations to return. Default is 10.

        Returns:
            list: A list of recommended item IDs.
        """
        if self.user_similarity_matrix is None:
            raise ValueError("The model has not been fitted yet. Call 'fit' first.")
        
        if user_id not in self.user_data:
            raise ValueError(f"User {user_id} not found in the user data.")
        
        user_id_index = self.users.index(user_id)
        user_item_matrix, _, _ = self._build_user_item_matrix()
        
        # Predict ratings for all items for this user
        predicted_ratings = self._predict_user_based(user_id_index, user_item_matrix)
        
        # Get the list of items the user hasn't interacted with yet
        user_rated_items = set(self.user_data[user_id].keys())
        unrated_items = [self.items[i] for i in range(len(self.items)) if self.items[i] not in user_rated_items]
        
        # Sort the predicted ratings for unrated items in descending order
        sorted_items = sorted(unrated_items, key=lambda x: predicted_ratings[self.items.index(x)], reverse=True)
        
        return sorted_items[:top_n]
