import numpy as np


def unpopular(rating: np.ndarray, movie: int):  # pi
    return 1 - np.sum(rating[:, movie] != -1) / rating.shape[0]

def all_group_movies(rating: np.ndarray, group: np.ndarray):
    group_movies = np.full(rating.shape[1], False)
    for u in group:
        group_movies = np.logical_or(group_movies, rating[u, :] != -1)
    return group_movies

def interception_movies_group_user(rating: np.ndarray, group: np.ndarray, user: int):
    group_movies = all_group_movies(rating=rating, group=group)
    user_movies = rating[user, :] != -1
    return np.logical_and(user_movies, group_movies)

def user_group_similarity(
    rating: np.ndarray, group: np.ndarray, user: int
):  # Xgu Jacard index
    group_movies = all_group_movies(rating=rating, group=group)
    user_movies = rating[user, :] != -1

    interception = interception_movies_group_user(rating=rating, group=group, user=user)
    union = np.logical_or(user_movies, group_movies)

    return np.sum(
        [
            unpopular(rating=rating, movie=i)
            for i in range(rating.shape[1])
            if interception[i]
        ]
    ) / np.sum(
        [unpopular(rating=rating, movie=i) for i in range(rating.shape[1]) if union[i]]
    )

def user_singularity(rating: np.ndarray, user: int, movie: int):
    movie_users = rating[:, movie] != -1
    return np.sum(
        np.logical_and(movie_users, rating[:, movie] != rating[user, movie])
    ) / np.sum(movie_users)

def item_group_singularity(rating: np.ndarray, group: np.ndarray, movie: int):
    movie_users_in_group = [user for user in group if rating[user][movie] != -1]

    return (
        np.power(
            np.prod(
                [
                    user_singularity(rating=rating, user=u, movie=movie)
                    for u in movie_users_in_group
                ]
            ),
            (1 / len(movie_users_in_group)),
        )
        if len(movie_users_in_group) > 0
        else 0
    )

def normalized_rating(rating: np.ndarray):
    rating_min = np.min(rating[rating != -1])
    rating_max = np.max(rating[rating != -1])
    normalized = (rating - rating_min) / (rating_max - rating_min)
    normalized[rating == -1] = -1  # Restaurar los valores -1
    return normalized


rating = np.array(
    [
        [-1, 1, -1, 1, 1, -1, 0, -1, 0, -1, -1, 0],
        [0, 1, -1, 0, 1, -1, 1, -1, 0, -1, -1, -1],
        [-1, 1, -1, 0, 0, -1, 0, 0, -1, -1, -1, 0],
        [-1, 1, 1, 1, 1, 1, 1, -1, 1, 0, 1, 1],
        [1, 0, -1, 1, -1, 1, -1, -1, 1, 0, -1, 1],
        [0, 1, -1, 0, 1, -1, 0, 0, 0, -1, -1, 0],
        [1, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1],
        [1, 1, -1, 0, 0, 0, 0, -1, 0, -1, 1, 0],
        [-1, 1, -1, 1, 0, -1, 1, -1, -1, 1, 1, -1],
    ]
)

groups = np.array([[0, 1, 2]])
group = groups[0]

# # Test 1 Unpopularity
# print([unpopular(rating, i) for i in range(rating.shape[1])])

# # Test 2 User group similarity
# print(
#     [
#         user_group_similarity(rating=rating, group=group, user=u)
#         for u in range(rating.shape[0])
#         if u not in group
#     ]
# )

# # Test 3 User singularity
# for u in range(rating.shape[0]):
#     print(
#         [
#             (i + 1, user_singularity(rating=rating, user=u, movie=i))
#             for i in range(rating.shape[1])
#             if rating[u, i] != -1
#         ]
#     )

# # Test 4 Item group singularity
# print(
#     [
#         item_group_singularity(rating=rating, group=group, movie=i)
#         for i in range(rating.shape[1])
#     ]
# )

# # Test 5 Normalized rating
# print(normalized_rating(rating))

# # Required for following tests
nr = normalized_rating(rating)

