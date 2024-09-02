import os
import numpy as np

from extended_naive_bayes.nbp import group_prediction
from nbcf.nbcf import MemorizationNBCF as nbcf
from ml_processing.ml_procesing import MovieLensProcessing


def main():
    # Load data
    time = os.times()
    ml_processing = MovieLensProcessing(rating_path="./datasets/ml-1m/ratings.dat")
    rating = ml_processing.numpy_user_movie_matrix()

    # Create recommenders
    alpha = 0.01
    r = 5

    memo_nbcf = nbcf()

    prior_user, prior_item, item_likelihood, user_likelihood = memo_nbcf.nbcf(
        rating=rating, alpha=alpha, r=r
    )
    user_prediction = memo_nbcf.predict_user(
        rating=rating, pu=prior_user, user_likelihood=user_likelihood, r=r
    )
    item_prediction = memo_nbcf.predict_item(
        rating=rating, pi=prior_item, item_likelihood=item_likelihood, r=r
    )
    hybrid_prediction = memo_nbcf.predict_hybrid(
        rating=rating, r=r, predict_item=item_prediction, predict_user=user_prediction
    )

    print("group recommender")
    # Group recommender
    groups = []
    group_predict = np.ndarray((len(groups), rating.shape[1], r))

    for i in len(groups):
        prediction = group_prediction(
            rating=rating, group=groups[i], hp=hybrid_prediction, r=r
        )
        for j in range(rating.shape[1]):
            for y in range(r):
                group_predict[j][y] = prediction[j][y]

    print("final prediction")
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

    print("Time:", os.times() - time)


if __name__ == "__main__":
    main()
