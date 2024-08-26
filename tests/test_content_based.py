import unittest
from src.recommendation.content_based import ContentBasedRecommender

class TestContentBasedRecommender(unittest.TestCase):
    def setUp(self):
        # Example item data for testing
        self.item_data = {
            'item1': {'feature1': 0.9, 'feature2': 0.1},
            'item2': {'feature1': 0.4, 'feature2': 0.6},
        }
        self.recommender = ContentBasedRecommender(self.item_data)

    def test_recommend(self):
        recommendations = self.recommender.recommend(user_id='user1', top_n=1)
        self.assertIsInstance(recommendations, list)
        self.assertGreaterEqual(len(recommendations), 0)

if __name__ == '__main__':
    unittest.main()
