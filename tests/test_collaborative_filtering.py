import unittest
from src.recommendation.collaborative_filtering import CollaborativeFilteringRecommender

class TestCollaborativeFiltering(unittest.TestCase):
    def setUp(self):
        # Example user data for testing
        self.user_data = {
            'user1': {'item1': 5, 'item2': 3},
            'user2': {'item1': 4, 'item3': 2},
        }
        self.recommender = CollaborativeFilteringRecommender(self.user_data)

    def test_recommend(self):
        recommendations = self.recommender.recommend(user_id='user1', top_n=1)
        self.assertIsInstance(recommendations, list)
        self.assertGreaterEqual(len(recommendations), 0)

if __name__ == '__main__':
    unittest.main()
