import os
import numpy as np

from extended_naive_bayes.nbp import group_prediction
from nbcf.nbcf_opt import nbcf, predict_hybrid
from ml_processing.ml_procesing import MovieLensProcessing

def main():
    # Load data
    ml_processing = MovieLensProcessing(rating_path="./datasets/ml-1m/ratings.dat")
    rating = ml_processing.numpy_user_movie_matrix()

    # Create recommenders
    alpha = 0.01
    r = 5
    
    user_prediction, item_prediction = nbcf(rating=rating, alpha=alpha, r=r)
    
    hybrid_prediction = predict_hybrid(rating=rating, r=r, predict_item=item_prediction, predict_user=user_prediction)

    # Group recommender
    groups = []
    group_predict = np.ndarray((len(groups), rating.shape[1], r))

    for i in len(groups):
        prediction = group_prediction(rating=rating, group=groups[i], hp=hybrid_prediction, r=r)
        for j in range(rating.shape[1]):
            for y in range(r):
                group_predict[j][y] = prediction[j][y]


    prediction = np.full(rating.shape, -1)

    users, movies = rating.shape
    for user in users:
        for movie in movies:
            if rating[user][movie] == -1:
                current_y = 0
                current_trust = 0
                for y in range(r):
                    if hybrid_prediction[user][movie][y] > current_trust:
                        current_trust = hybrid_prediction[user][movie][y]
                        current_y = y
                prediction[user][movie] = current_y
            
    

if __name__ == "__main__":
    main()
