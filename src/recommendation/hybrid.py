class HybridRecommender:
    """
    A hybrid recommender system that combines collaborative filtering and content-based methods.
    """

    def __init__(self, collaborative_recommender, content_recommender, user_activity, item_metadata):
        """
        Initializes the HybridRecommender.

        Args:
            collaborative_recommender (CollaborativeFilteringRecommender): Collaborative filtering recommender instance.
            content_recommender (ContentBasedRecommender): Content-based recommender instance.
            user_activity (dict): A dictionary of user activities.
            item_metadata (pd.DataFrame): A dataframe containing metadata for items.
        """
        self.collaborative_recommender = collaborative_recommender
        self.content_recommender = content_recommender
        self.user_activity = user_activity
        self.item_metadata = item_metadata

    def recommend(self, user_id, top_n=10):
        """
        Generates hybrid recommendations for a user.

        Args:
            user_id (int): The ID of the user for whom to generate recommendations.
            top_n (int): The number of top recommendations to return.

        Returns:
            list: A list of recommended items.
        """
        collab_recs = self.collaborative_recommender.recommend(user_id, top_n=top_n)
        content_recs = self.content_recommender.recommend(user_id, top_n=top_n)

        # Combine recommendations and adjust based on user activity and item metadata
        combined_recs = self._combine_recommendations(collab_recs, content_recs)

        # Sort by relevance considering user activity and metadata
        sorted_recs = self._sort_by_relevance(combined_recs, user_id)

        return sorted_recs[:top_n]

    def _combine_recommendations(self, collab_recs, content_recs):
        """
        Combines collaborative and content-based recommendations.

        Args:
            collab_recs (list): Collaborative filtering recommendations.
            content_recs (list): Content-based recommendations.

        Returns:
            list: Combined recommendations.
        """
        combined = {**{rec['movie_id']: rec for rec in collab_recs},
                    **{rec['movie_id']: rec for rec in content_recs}}
        return list(combined.values())

    def _sort_by_relevance(self, recommendations, user_id):
        """
        Sorts recommendations by relevance, considering user activity and item metadata.

        Args:
            recommendations (list): The list of combined recommendations.
            user_id (int): The ID of the user for whom to generate recommendations.

        Returns:
            list: Sorted list of recommendations by relevance.
        """
        # Example of sorting logic considering user activity and metadata
        recommendations.sort(key=lambda x: (self.user_activity.get(user_id, {}).get(x['movie_id'], 0),
                                            self.item_metadata.loc[x['movie_id'], 'rating']),
                             reverse=True)
        return recommendations
