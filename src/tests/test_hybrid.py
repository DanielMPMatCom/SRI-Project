import unittest
from src.recommendation.collaborative_filtering import CollaborativeFilteringRecommender
from src.recommendation.content_based import ContentBasedRecommender
from src.recommendation.hybrid import HybridRecommender

class TestHybridRecommender(unittest.TestCase):
    def setUp(self):
        # Example data for testing
        self.user_data = {'user1': {'item1': 5}}
        self.item_data = {'item1': {'feature1': 0.9}}
        self.collab_recommender = CollaborativeFilteringRecommender(self.user_data)
        self.content_recommender = ContentBasedRecommender(self.item_data)
        self.recommender = HybridRecommender(self.collab_recommender, self.content_recommender)

    def test_recommend(self):
        recommendations = self.recommender.recommend(user_id='user1', top_n=1)
        self.assertIsInstance(recommendations, list)
        self.assertGreaterEqual(len(recommendations), 0)

if __name__ == '__main__':
    unittest.main()
