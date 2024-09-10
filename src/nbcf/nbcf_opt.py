import numpy as np
import time


def nbcf(rating: np.ndarray, alpha: float, r: int, qualified_array: np.ndarray):

    users, movies = rating.shape
    r_alpha = r * alpha

    pu = np.zeros((users, r), float)
    pi = np.zeros((movies, r), float)

    m_collaborative_filtering = {}

    u_collaborative_filtering = {}

    user_map = [set() for i in range(users)]
    movie_map = [set() for i in range(movies)]

    d = time.time()
    print("0 ⏰ Mapping movies and users ...")

    # Map movies and users for faster navegation
    for user in range(users):
        user_map[user] = set()
        for movie in range(movies):
            if rating[user][movie] != -1:
                user_map[user].add(movie)
                movie_map[movie].add(user)

    print(time.time() - d, "⏰ Mapped movies and users ...")
    d = time.time()

    # Prior probability user based
    for movie in range(movies):
        d = len(movie_map[movie]) + r_alpha
        for qualified in qualified_array:
            m_collaborative_filtering[(movie, qualified)] = [
                i for i in range(users) if rating[i, movie] == qualified + 1
            ]
            pi[movie, qualified] = (
                len(m_collaborative_filtering[(movie, qualified)]) + alpha
            ) / d

    print(time.time() - d, "⏰ Prior probability user based End...")
    d = time.time()
    # Prior probability item based
    for user in range(users):
        d = len(user_map[user]) + r_alpha
        for qualified in qualified_array:
            u_collaborative_filtering[(user, qualified)] = [
                i for i in range(movies) if rating[user, i] == qualified + 1
            ]
            pu[user, qualified] = (
                len(u_collaborative_filtering[(user, qualified)]) + alpha
            ) / d

    print(time.time() - d, "⏰ Prior probability item based End ...")
    d = time.time()
    print("Start Calculating Predictions ...")
    # Calculate predictions
    item_prediction = np.zeros((users, movies, r), float)
    user_prediction = np.zeros((users, movies, r), float)

    for user in range(users):
        print(user, " of ", users)
        for movie in range(movies):
            
            if rating[user][movie] != -1:
                continue

            for qualified in qualified_array:

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

    print(time.time() - d, "⏰ Calculated Predictions ...")

    return user_prediction, item_prediction, user_map, movie_map


def predict_hybrid(
    rating: np.ndarray,
    r,
    user_prediction,
    item_prediction,
    user_map,
    movie_map,
    qualified_array,
):
    users, movies = rating.shape
    prediction = np.zeros((users, movies, r), float)
    for user in range(users):
        for movie in range(movies):
            # if rating[user][movie] != -1:
            #     continue
            ui = len(movie_map[movie])
            iu = len(user_map[user])
            for qualified in qualified_array:
                if item_prediction[user, movie, qualified] == 0:
                    print("Error:", user, movie, qualified)
                    print("el usuario ha calificado la película?", rating[user, movie])

                prediction[user, movie, qualified] = user_prediction[
                    user, movie, qualified
                ] ** (1 / (1 + ui)) * item_prediction[user, movie, qualified] ** (
                    1 / (1 + iu)
                )

    return prediction


def attempt(value, expected, test_name=""):
    try:
        assert value == expected
        print("\033[92m" + f"✅ Test {test_name} passed!" + "\033[0m")
    except AssertionError as e:
        raise AssertionError(
            f"❌ Test failed! Expected {expected}, but got {value}"
        ) from e


# Test

rating = np.array(
    [
        [-1, 1, 2, 2, 5, -1, 4, 3, 5],
        [1, 5, 3, -1, 2, 4, 4, 3, -1],
        [1, 1, 2, -1, 2, 4, 4, 5, -1],
        [3, 2, 2, 3, -1, 1, 3, 2, -1],
        [5, 1, 5, 5, 4, 4, 5, 2, -1],
    ]
)

ALPHA = 0.01
R = 5

user_prediction, item_prediction, um, mm = nbcf(
    rating=rating, alpha=ALPHA, r=R, qualified_array=[i for i in range(R)]
)


# print(pi[0, 0])
print(
    item_prediction[0, 0]
    - np.array(
        [1.13551245e-05, 3.16049383e-08, 7.89407449e-11, 3.16049383e-08, 3.75908309e-12]
    )
)

attempt(
    np.allclose(
        item_prediction[0, 0],
        [
            1.13551245e-05,
            3.16049383e-08,
            7.89407449e-11,
            3.16049383e-08,
            3.75908309e-12,
        ],
    ),
    True,
    "item prediction",
)


print(user_prediction[0, 0])
attempt(
    np.allclose(
        user_prediction[0, 0],
        [
            1.19040964e-07,
            1.24921748e-05,
            1.17862340e-09,
            1.20231373e-05,
            4.92571226e-08,
        ],
    ),
    True,
    "user prediction",
)


ph = predict_hybrid(
    rating=rating,
    r=R,
    user_prediction=user_prediction,
    item_prediction=item_prediction,
    user_map=um,
    movie_map=mm,
    qualified_array=[i for i in range(R)],
)

attempt(
    np.allclose(
        ph[0, 0],
        [0.00993204, 0.01207249, 0.00089421, 0.01198044, 0.00128937],
    ),
    True,
    "hybrid test",
)


print(ph[0, 0])
