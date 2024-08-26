from textblob import TextBlob

class EmotionAnalyzer:
    """
    A class for analyzing the sentiment of user comments and adjusting recommendations accordingly.
    """

    def analyze_emotion(self, comment):
        """
        Analyzes the sentiment of a user's comment.

        Args:
            comment (str): The comment text.

        Returns:
            str: The detected emotion ('happy', 'love', 'excited', 'sad', 'neutral').
        """
        analysis = TextBlob(comment)
        polarity = analysis.sentiment.polarity

        if polarity > 0.5:
            return 'happy'
        elif 0.1 < polarity <= 0.5:
            return 'love'
        elif -0.1 < polarity <= 0.1:
            return 'neutral'
        elif -0.5 < polarity <= -0.1:
            return 'sad'
        else:
            return 'excited'

class EmotionBasedRecommender:
    """
    A class to adjust recommendations based on the user's detected emotion.
    """

    def __init__(self, emotion_analyzer, recommendations):
        """
        Initializes the EmotionBasedRecommender.

        Args:
            emotion_analyzer (EmotionAnalyzer): An instance of the EmotionAnalyzer.
            recommendations (list): The initial list of recommendations.
        """
        self.emotion_analyzer = emotion_analyzer
        self.recommendations = recommendations

    def adjust_recommendations(self, emotion):
        """
        Adjusts the recommendations based on the detected emotion.

        Args:
            emotion (str): The detected emotion.

        Returns:
            list: The adjusted list of recommendations.
        """
        if emotion == 'happy':
            return [rec for rec in self.recommendations if 'comedy' in rec['genre']]
        elif emotion == 'love':
            return [rec for rec in self.recommendations if 'romance' in rec['genre']]
        elif emotion == 'excited':
            return [rec for rec in self.recommendations if 'action' in rec['genre']]
        elif emotion == 'sad':
            return [rec for rec in self.recommendations if 'drama' in rec['genre'] or 'comedy' in rec['genre']]
        else:
            return self.recommendations
