import unittest
from src.evaluation.metrics import EvaluationMetrics

class TestEvaluationMetrics(unittest.TestCase):
    def test_calculate_rmse(self):
        true_ratings = [4, 5, 3, 2, 1]
        predicted_ratings = [4.2, 4.8, 3.1, 2.0, 1.3]
        rmse = EvaluationMetrics.calculate_rmse(true_ratings, predicted_ratings)
        self.assertAlmostEqual(rmse, 0.18, places=2)

    def test_calculate_precision(self):
        true_labels = [1, 0, 1, 0, 1]
        predicted_labels = [0.9, 0.4, 0.8, 0.3, 0.9]
        precision = EvaluationMetrics.calculate_precision(true_labels, predicted_labels, threshold=0.5)
        self.assertAlmostEqual(precision, 1.0)

    def test_calculate_recall(self):
        true_labels = [1, 0, 1, 0, 1]
        predicted_labels = [0.9, 0.4, 0.8, 0.3, 0.9]
        recall = EvaluationMetrics.calculate_recall(true_labels, predicted_labels, threshold=0.5)
        self.assertAlmostEqual(recall, 1.0)

if __name__ == '__main__':
    unittest.main()
