import numpy as np


def nbcf(rating: np.ndarray, alpha, r):
    # user based approach
    users, movies = rating.shape
    r_alpha = r * alpha

    pu = np.zeros((users, r + 1))
    pi = np.zeros((movies, r + 1))

    for movie in range(movies):
        movie_users = np.sum(rating[:, movie] != -1) + r_alpha
        for qualified in range(r + 1):
            pi[movie, qualified] = (
                np.sum(rating[:, movie] == qualified) + alpha
            ) / movie_users

    for user in range(users):
        user_movie = np.sum(rating[user, :] != -1) + r_alpha
        for qualified in range(r + 1):
            pu[user, qualified] = (
                np.sum(rating[user, :] == qualified) + alpha
            ) / user_movie

    item_likehood = np.zeros((movies, r + 1, movies, r + 1), 0)

    for imovie in range(movie):
        for ir in range(r + 1):
            count_user_ir = np.sum(rating[:, imovie] == ir)
            for jmovie in range(movie):
                for jr in range(r + 1):
                    count_user_jr = np.sum(rating[:, jmovie] == jr)
                    

    return pu, pi


rating = np.array(
    [
        [-1, 1, 2, 2, 5, -1, 4, 3, 5],
        [1, 5, 3, -1, 2, 4, 4, 5, -1],
        [1, 1, 2, -1, 2, 4, 4, 5, -1],
        [3, 2, 2, 3, -1, 1, 3, 2, -1],
        [5, 1, 5, 5, 4, 4, 5, 2, -1],
    ]
)

ALPHA = 0.01
R = 5

a, b = nbcf(rating=rating, alpha=ALPHA, r=R)
print(a)
print(" - - - - - ")
print(b)

print(" =========================================")

a, b = nbcf_paper(rating=rating, alpha=ALPHA, r=R)
print(a)
print(" - - - -- ")
print(b)
