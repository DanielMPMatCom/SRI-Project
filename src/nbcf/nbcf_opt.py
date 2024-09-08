import time
import torch


def nbcf(rating, alpha: float, r: int):

    users, movies = rating.shape
    r_alpha = r * alpha

    pu = torch.zeros((users, r), device=rating.device)
    pi = torch.zeros((movies, r), device=rating.device)

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
        dm = len(movie_map[movie]) + r_alpha
        ci = torch.zeros(r, device=rating.device)
        for user in movie_map[movie]:
            ci[rating[user, movie] - 1] += 1
        m_collaborative_filtering[movie] = ci
        pi[movie] = (m_collaborative_filtering[movie] + alpha) / dm

    print(time.time() - d, " Prior probability user based End...")
    d = time.time()
    # Prior probability item based
    for user in range(users):
        du = len(user_map[user]) + r_alpha
        cu = torch.zeros(r, device=rating.device)
        for movie in user_map[user]:
            cu[rating[user, movie] - 1] += 1
        u_collaborative_filtering[user] = cu
        pu[user] = (u_collaborative_filtering[user] + alpha) / du

    print(time.time() - d, " Prior probability item based End ...")

    d = time.time()
    print("Start Calculating Predictions ...")
    # Calculate predictions
    item_prediction = torch.zeros((users, movies, r), device=rating.device)
    user_prediction = torch.zeros((users, movies, r), device=rating.device)

    for user in range(users):
        for movie in range(movies):

            if rating[user][movie] != -1:
                continue

            for qualified in range(r):

                # Item based prediction
                tmp = pi[movie, qualified]
                denominator = m_collaborative_filtering[movie][qualified]
                for j_movie in user_map[user]:
                    numerator = 0
                    numerator = torch.sum(
                        rating[torch.tensor(list(movie_map[j_movie])), j_movie]
                        == rating[torch.tensor(list(movie_map[j_movie])), movie]
                    )
                    tmp *= (numerator + alpha) / (denominator + r_alpha)
                item_prediction[user, movie, qualified] = tmp

                # User based prediction
                tmp = pu[user, qualified]
                denominator = u_collaborative_filtering[user][qualified]

                for j_user in movie_map[movie]:
                    numerator = torch.sum(
                        rating[j_user, torch.tensor(list(user_map[j_user]))]
                        == rating[user, torch.tensor(list(user_map[j_user]))]
                    )
                    tmp *= (numerator + alpha) / (denominator + r_alpha)
                user_prediction[user, movie, qualified] = tmp

    print(time.time() - d, " Calculated Predictions ...")

    return user_prediction, item_prediction, user_map, movie_map


def predict_hybrid(rating, r, predict_item, predict_user, user_map, movie_map):
    users, movies = rating.shape
    prediction = torch.zeros((users, movies, r))
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

# rating = torch.array(
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
