import numpy as np
import time


def nbcf(rating: np.ndarray, alpha: float, r: int):

    users, movies = rating.shape
    r_alpha = r * alpha

    pu = np.zeros((users, r))
    pi = np.zeros((movies, r))

    movie_users = {}
    m_collaborative_filtering = {}

    user_movies = {}
    u_collaborative_filtering = {}

    user_map = [set() for i in range(users)]
    movie_map = [set() for i in range(movies)]

    d = time.time()
    print("0 Mapping movies and users ...")

    # Map movies and users for faster navegation
    for user in range(users):
        user_map[user] = set()
        for movie in range(movies):
            if rating[user][movie] != -1:
                user_map[user].add(movie)
                movie_map[movie].add(user)

    print(time.time() - d, " Mapped movies and users ...")
    d = time.time()
    # Prior probability user based
    for movie in range(movies):
        movie_users[movie] = len(movie_map[movie]) + r_alpha
        for qualified in range(r):
            m_collaborative_filtering[(movie, qualified)] = [
                i for i in range(users) if rating[i, movie] == qualified + 1
            ]
            pi[movie, qualified] = (
                len(m_collaborative_filtering[(movie, qualified)]) + alpha
            ) / movie_users[movie]

    print(time.time() - d, " Prior probability user based End...")
    d = time.time()
    # Prior probability item based
    for user in range(users):
        user_movies[user] = len(user_map[user]) + r_alpha
        for qualified in range(r):
            u_collaborative_filtering[(user, qualified)] = [
                i for i in range(movies) if rating[user, i] == qualified + 1
            ]
            pu[user, qualified] = (
                len(u_collaborative_filtering[(user, qualified)]) + alpha
            ) / user_movies[user]

    print(time.time() - d, " Prior probability item based End ...")
    d = time.time()
    print("Start Calculating Predictions ...")
    # Calculate predictions
    item_prediction = np.full((users, movies, r), -1, float)
    user_prediction = np.full((users, movies, r), -1, float)

    for user in range(users):
        for movie in range(movies):

            if rating[user][movie] != -1:
                continue

            for qualified in range(r):

                # Item based prediction
                tmp = pi[movie, qualified]

                for j_movie in user_map[user]:
                    numerator = 0
                    denominator = 0
                    for u in m_collaborative_filtering[(movie, qualified)]:
                        if rating[u, j_movie] == -1:
                            continue
                        denominator += 1
                        if rating[u, j_movie] == rating[user, j_movie]:
                            numerator += 1

                    tmp *= (numerator + alpha) / (denominator + r_alpha)

                item_prediction[user, movie, qualified] = tmp

                # User based prediction
                tmp = pu[user, qualified]

                for j_user in movie_map[movie]:
                    numerator = 0
                    denominator = 0
                    for m in u_collaborative_filtering[(user, qualified)]:
                        if rating[j_user, m] == -1:
                            continue
                        denominator += 1
                        if rating[j_user, m] == rating[j_user, movie]:
                            numerator += 1

                    tmp *= (numerator + alpha) / (denominator + r_alpha)

                user_prediction[user, movie, qualified] = tmp

    print(time.time() - d, " Calculated Predictions ...")

    return user_prediction, item_prediction, user_map, movie_map


def predict_hybrid(
    rating: np.ndarray, r, predict_item, predict_user, user_map, movie_map
):
    users, movies = rating.shape
    prediction = np.zeros((users, movies, r))
    for user in range(users):
        for movie in range(movies):
            if rating[user][movie] != -1:
                continue
            ui = len(movie_map[movie])
            iu = len(user_map[user])
            for qualified in range(r):
                prediction[user, movie, qualified] = predict_user[
                    user, movie, qualified
                ] ** (1 / (1 + ui)) * predict_item[user, movie, qualified] ** (
                    1 / (1 + iu)
                )
    return prediction


## Test

# rating = np.array(
#     [
#         [-1, 1, 2, 2, 5, -1, 4, 3, 5],
#         [1, 5, 3, -1, 2, 4, 4, 3, -1],
#         [1, 1, 2, -1, 2, 4, 4, 5, -1],
#         [3, 2, 2, 3, -1, 1, 3, 2, -1],
#         [5, 1, 5, 5, 4, 4, 5, 2, -1],
#     ]
# )

# ALPHA = 0.01
# R = 5

# pu, pi = nbcf(rating=rating, alpha=ALPHA, r=R)

# print(" =========================================")

# print(pi[0, 0])

# print(" =========================================")

# print(pu[0, 0])

# ph = predict_hybrid(rating=rating, r=R, predict_item=pi, predict_user=pu)

# print(ph[0, 0])
