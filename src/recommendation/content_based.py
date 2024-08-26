import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    """
    A content-based recommender system.

    This class implements a content-based recommendation approach to generate 
    recommendations for users by analyzing the content features of items.
    
    Attributes:
        item_data (dict): A dictionary containing item information.
        user_profiles (dict): A dictionary containing user profiles based on their preferences.
        item_profiles (numpy.ndarray): A matrix where rows represent items and columns represent content features.
    """
    
    def __init__(self, item_data, user_profiles):
        """
        Initializes the ContentBasedRecommender with item data and user profiles.

        Args:
            item_data (dict): A dictionary with item information.
            user_profiles (dict): A dictionary with user profiles.
        """
        self.item_data = item_data
        self.user_profiles = user_profiles
        self.item_profiles = None
    
    def _build_item_profiles(self):
        """
        Builds item profiles based on their content features.

        Returns:
            numpy.ndarray: A matrix where rows represent items and columns represent content features.
        """
        # Extract item descriptions or relevant content features
        item_descriptions = [self.item_data[item]['description'] for item in self.item_data]
        
        # Convert descriptions to a TF-IDF matrix
        vectorizer = TfidfVectorizer()
        self.item_profiles = vectorizer.fit_transform(item_descriptions)
    
    def _build_user_profile(self, user_id):
        """
        Builds a user profile based on their interaction history and preferences.

        Args:
            user_id (int): The ID of the user for whom the profile is being built.

        Returns:
            numpy.ndarray: A vector representing the user's profile.
        """
        # Initialize user profile with zeros
        user_profile = np.zeros(self.item_profiles.shape[1])
        
        # Get the user's interaction history
        user_history = self.user_profiles.get(user_id, {})
        
        # Accumulate the weighted sum of item profiles based on user ratings
        for item, rating in user_history.items():
            item_index = list(self.item_data.keys()).index(item)
            user_profile += rating * self.item_profiles[item_index].toarray()[0]
        
        # Normalize the user profile
        user_profile /= np.linalg.norm(user_profile)
        
        return user_profile
    
    def fit(self):
        """
        Fits the content-based model to the provided data.
        This method generates item profiles from item content and optionally updates user profiles.
        """
        self._build_item_profiles()
    
    def recommend(self, user_id, top_n=10):
        """
        Generates recommendations for a given user based on the content-based model.

        Args:
            user_id (str): The ID of the user to generate recommendations for.
            top_n (int): The number of top recommendations to return. Default is 10.

        Returns:
            list: A list of recommended item IDs.
        """
        # Build the user's profile
        user_profile = self._build_user_profile(user_id)
        
        # Calculate similarity between user profile and item profiles
        similarity_scores = cosine_similarity([user_profile], self.item_profiles)
        
        # Get top N items with highest similarity scores
        top_item_indices = similarity_scores.argsort()[0][-top_n:][::-1]
        recommended_items = [list(self.item_data.keys())[i] for i in top_item_indices]
        
        return recommended_items
