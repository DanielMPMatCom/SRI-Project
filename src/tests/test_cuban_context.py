import unittest
from src.recommendation.cuban_context import CubanContextAdapter

class TestCubanContextAdapter(unittest.TestCase):
    def setUp(self):
        # Example recommendations for testing
        self.recommendations = ['item1', 'item2']
        self.adapter = CubanContextAdapter(popular_content=['item2', 'item3'])

    def test_adjust_recommendations(self):
        adjusted_recommendations = self.adapter.adjust_recommendations(self.recommendations)
        self.assertIn('item2', adjusted_recommendations)

if __name__ == '__main__':
    unittest.main()
