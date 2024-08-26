from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score, roc_auc_score

class EvaluationMetrics:
    """
    A class for evaluating the performance of the recommendation system.

    This class provides methods to calculate common metrics such as RMSE, precision, recall, F1-score, and AUC-ROC.

    Attributes:
        None
    """
    
    @staticmethod
    def calculate_rmse(true_ratings, predicted_ratings):
        """
        Calculates the Root Mean Squared Error (RMSE) between true and predicted ratings.

        Args:
            true_ratings (list): The list of true user ratings.
            predicted_ratings (list): The list of predicted ratings by the recommender system.

        Returns:
            float: The RMSE value.
        """
        return mean_squared_error(true_ratings, predicted_ratings, squared=False)
    
    @staticmethod
    def calculate_precision(true_labels, predicted_labels, threshold=0.5):
        """
        Calculates the precision score based on true and predicted labels.

        Args:
            true_labels (list): The list of true binary labels.
            predicted_labels (list): The list of predicted probabilities or binary labels.
            threshold (float): The threshold to convert probabilities to binary labels if needed.

        Returns:
            float: The precision score.
        """
        binary_predictions = [1 if p >= threshold else 0 for p in predicted_labels]
        return precision_score(true_labels, binary_predictions)
    
    @staticmethod
    def calculate_recall(true_labels, predicted_labels, threshold=0.5):
        """
        Calculates the recall score based on true and predicted labels.

        Args:
            true_labels (list): The list of true binary labels.
            predicted_labels (list): The list of predicted probabilities or binary labels.
            threshold (float): The threshold to convert probabilities to binary labels if needed.

        Returns:
            float: The recall score.
        """
        binary_predictions = [1 if p >= threshold else 0 for p in predicted_labels]
        return recall_score(true_labels, binary_predictions)
    
    @staticmethod
    def calculate_f1_score(true_labels, predicted_labels, threshold=0.5):
        """
        Calculates the F1-score based on true and predicted labels.

        Args:
            true_labels (list): The list of true binary labels.
            predicted_labels (list): The list of predicted probabilities or binary labels.
            threshold (float): The threshold to convert probabilities to binary labels if needed.

        Returns:
            float: The F1-score.
        """
        binary_predictions = [1 if p >= threshold else 0 for p in predicted_labels]
        return f1_score(true_labels, binary_predictions)
    
    @staticmethod
    def calculate_auc_roc(true_labels, predicted_probabilities):
        """
        Calculates the Area Under the Receiver Operating Characteristic Curve (AUC-ROC).

        Args:
            true_labels (list): The list of true binary labels.
            predicted_probabilities (list): The list of predicted probabilities.

        Returns:
            float: The AUC-ROC score.
        """
        return roc_auc_score(true_labels, predicted_probabilities)
    
    @staticmethod
    def log_metrics(true_labels, predicted_labels, predicted_probabilities=None, threshold=0.5):
        """
        Logs the main evaluation metrics including RMSE, Precision, Recall, F1-score, and AUC-ROC.

        Args:
            true_labels (list): The list of true binary labels.
            predicted_labels (list): The list of predicted labels or probabilities.
            predicted_probabilities (list, optional): The list of predicted probabilities for AUC-ROC. Defaults to None.
            threshold (float): The threshold to convert probabilities to binary labels if needed. Defaults to 0.5.
        """
        precision = EvaluationMetrics.calculate_precision(true_labels, predicted_labels, threshold)
        recall = EvaluationMetrics.calculate_recall(true_labels, predicted_labels, threshold)
        f1 = EvaluationMetrics.calculate_f1_score(true_labels, predicted_labels, threshold)
        auc_roc = EvaluationMetrics.calculate_auc_roc(true_labels, predicted_probabilities) if predicted_probabilities is not None else 'N/A'
        
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")
