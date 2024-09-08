import cupy as cp
import time


def nbcf(rating, alpha: float, r: int):

    users, movies = rating.shape
    r_alpha = r * alpha

    pu = cp.zeros((users, r))
    pi = cp.zeros((movies, r))

    m_collaborative_filtering = {}

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
        ci = cp.zeros(r)
        for user in movie_map[movie]:
            ci[rating[user, movie] - 1] += 1
        m_collaborative_filtering[movie] = ci

        pi[movie] = (ci + alpha) / (len(movie_map[movie]) + r_alpha)

    print(time.time() - d, " Prior probability user based End...")
    d = time.time()
    # Prior probability item based
    for user in range(users):
        cu = cp.zeros(r)
        for movie in user_map[user]:
            cu[rating[user, movie] - 1] += 1
        u_collaborative_filtering[user] = cu

        pu[user] = (cu + alpha) / (len(user_map[user]) + r_alpha)

    print(time.time() - d, " Prior probability item based End ...")
    d = time.time()
    print("Start Calculating Predictions ...")
    # Calculate predictions
    item_prediction = cp.zeros((users, movies, r))
    user_prediction = cp.zeros((users, movies, r))

    for user in range(users):
        for movie in range(movies):

            if rating[user][movie] != -1:
                continue

            denominator = m_collaborative_filtering[movie] + r_alpha
            for j_movie in user_map[user]:

                user_masks = cp.array(
                    [rating[:, j_movie] == (qualified + 1) for qualified in range(r)]
                )
                user_movie_counts_same_rating = cp.sum(user_masks, axis=1) + alpha

                item_prediction[user, movie] = (
                    cp.dot(pi[movie], user_movie_counts_same_rating) / denominator
                )

            denominator = u_collaborative_filtering[user] + r_alpha
            for j_user in movie_map[movie]:

                item_masks = cp.array(
                    [rating[j_user, :] == (qualified + 1) for qualified in range(r)]
                )
                item_user_counts_same_rating = cp.sum(item_masks, axis=1) + alpha

                user_prediction[user, movie] = (
                    cp.dot(pu[user], item_user_counts_same_rating) / denominator
                )

    print(time.time() - d, " Calculated Predictions ...")

    return user_prediction, item_prediction, user_map, movie_map


def predict_hybrid(rating, r, predict_item, predict_user, user_map, movie_map):
    users, movies = rating.shape

    prediction = cp.zeros((users, movies, r))

    for user in range(users):
        for movie in range(movies):
            if rating[user][movie] != -1:
                continue

            ui = len(movie_map[movie])
            iu = len(user_map[user])

            prediction[user, movie] = predict_user[user, movie] ** (
                1 / (1 + ui)
            ) * predict_item[user, movie] ** (1 / (1 + iu))

    return prediction


## Test

# rating = cp.array(
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
