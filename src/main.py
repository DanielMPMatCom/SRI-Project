import os
import pandas as pd
from utils.config import Config
from data_processing.data_loader import DataLoader
from data_processing.preprocessing import DataPreprocessor
from recommendation.collaborative_filtering import CollaborativeFilteringRecommender
from recommendation.content_based import ContentBasedRecommender
from recommendation.hybrid import HybridRecommender
from recommendation.cuban_context import CubanContextAdapter
from recommendation.emotion_based import EmotionAnalyzer, EmotionBasedRecommender
from recommendation.watch_history import WatchHistory
from evaluation.metrics import EvaluationMetrics

def main():
    # Load data
    data_loader = DataLoader(Config.USER_DATA_PATH, Config.ITEM_DATA_PATH)
    user_data, item_data = data_loader.load_data()

    # Preprocess data
    preprocessor = DataPreprocessor()
    user_data = preprocessor.clean_data(user_data)
    item_data = preprocessor.clean_data(item_data)

    # Ensure DataFrame indexes are reset
    user_data.reset_index(drop=True, inplace=True)
    item_data.reset_index(drop=True, inplace=True)

    # Create recommenders
    collaborative_recommender = CollaborativeFilteringRecommender(user_data, item_data)
    content_recommender = ContentBasedRecommender(item_data, user_data)

    # Fit the collaborative filtering recommender
    collaborative_recommender.fit()

    # Prepare to save recommendations
    recommendations_list = []

    # Iterate over each user
    for user_id in user_data['user_id']:
        try:
            # Debugging: Verify user_id is present in the DataFrame
            if user_id not in user_data['user_id'].values:
                print(f"User {user_id} not found in the data!")
                continue

            # Hybrid recommendation
            hybrid_recommender = HybridRecommender(
                collaborative_recommender, content_recommender, user_data, item_data
            )
            recommendations = hybrid_recommender.recommend(
                user_id=user_id, top_n=Config.TOP_N_RECOMMENDATIONS
            )

            # Contextual adaptation for Cuban users
            cuban_context_adapter = CubanContextAdapter(popular_content=['item1', 'item2', 'item3'])
            adapted_recommendations = cuban_context_adapter.adjust_recommendations(recommendations)

            # Emotion-based recommendation adjustment
            comment = user_data.loc[user_data['user_id'] == user_id, 'mood_comment'].values[0] 
            emotion_analyzer = EmotionAnalyzer()
            detected_emotion = emotion_analyzer.analyze_emotion(comment)

            emotion_based_recommender = EmotionBasedRecommender(emotion_analyzer, adapted_recommendations)
            emotion_adjusted_recommendations = emotion_based_recommender.adjust_recommendations(detected_emotion)

            # Watch history filtering
            watch_history = WatchHistory(user_id=user_id)
            final_recommendations = watch_history.filter_watched_movies(emotion_adjusted_recommendations)

            # Store recommendations in the list
            for item in final_recommendations:
                recommendations_list.append({"user_id": user_id, "recommended_item": item})

        except Exception as e:
            print(f"Error processing user {user_id}: {e}")

    # Save all recommendations to a CSV file
    recommendations_df = pd.DataFrame(recommendations_list)
    output_path = os.path.join("../datasets", "recommendations.csv")
    recommendations_df.to_csv(output_path, index=False)

    print(f"Recommendations saved to {output_path}")

    # Evaluate system performance (example evaluation)
    true_ratings = [4, 5, 3, 2, 1]
    predicted_ratings = [4.2, 4.8, 3.1, 2.0, 1.3]
    
    rmse = EvaluationMetrics.calculate_rmse(true_ratings, predicted_ratings)
    print("RMSE:", rmse)

if __name__ == "__main__":
    main()
