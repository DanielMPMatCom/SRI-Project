import numpy as np
from nbcf.Serialized import Serialized


class MemorizationNBCF:

    def __init__(self):
        self.cache_ui = {}
        self.cache_iu = {}
        self.cache_gui = {}
        self.cache_movie_rating = {}
        self.cache_user_rating = {}
        self.serialized = Serialized()

    def hash_group(self, group: list):
        return "$".join(group)

    def hash_user_item(self, user: int, qualified: int):
        return f"{user}${qualified}"

    def hash_movie_rating(self, movie: int, qualified: int):
        return f"{movie}${qualified}"

    def get_user_rating(self, rating: np.ndarray, user: int, qualified: int):
        key = self.hash_user_item(user, qualified)
        if key in self.cache_user_rating:
            return self.cache_user_rating[key]
        else:
            user_rating = rating[user, :] == qualified + 1
            self.cache_user_rating[key] = user_rating
            return user_rating

    def get_movie_rating(self, rating: np.ndarray, movie: int, qualified: int):
        key = self.hash_movie_rating(movie, qualified)
        if key in self.cache_movie_rating:
            return self.cache_movie_rating[key]
        else:
            movie_rating = rating[:, movie] == qualified + 1
            self.cache_movie_rating[key] = movie_rating
            return movie_rating

    def get_ui(self, rating: np.ndarray, movie: int):
        if movie in self.cache_ui:
            return self.cache_ui[movie]
        else:
            UI = rating[:, movie] != -1
            self.cache_ui[movie] = UI
            return UI

    def get_iu(self, rating: np.ndarray, user: int):
        if user in self.cache_iu:
            return self.cache_iu[user]
        else:
            IU = rating[user, :] != -1
            self.cache_iu[user] = IU
            return IU

    def nbcf(self, rating: np.ndarray, alpha: float, r: int):
        """
        NBCF (Naive Bayes Collaborative Filtering) algorithm implementation.

        Args:
            rating (np.ndarray): 2D array representing the user-item rating matrix.
            alpha (float): Smoothing parameter for Laplace smoothing.
            r (int): Number of rating levels.

        Returns:
            pu (np.ndarray): 2D array representing the user latent factors.
            pi (np.ndarray): 2D array representing the item latent factors.
            item_likelihood (np.ndarray): 4D array representing the item-item likelihoods.
            user_likelihood (np.ndarray): 4D array representing the user-user likelihoods.
        """

        users, movies = rating.shape
        r_alpha = r * alpha

        pu = np.zeros((users, r))
        pi = np.zeros((movies, r))

        for movie in range(movies):
            movie_users = np.sum(self.get_ui(rating=rating, movie=movie)) + r_alpha
            for qualified in range(r):
                pi[movie, qualified] = (
                    np.sum(
                        self.get_movie_rating(
                            rating=rating, movie=movie, qualified=qualified + 1
                        )
                    )
                    + alpha
                ) / movie_users

        for user in range(users):
            user_movie = np.sum(self.get_iu(rating=rating, user=user)) + r_alpha
            for qualified in range(r):
                pu[user, qualified] = (
                    np.sum(
                        self.get_user_rating(
                            rating=rating, user=user, qualified=qualified + 1
                        )
                    )
                    + alpha
                ) / user_movie

        item_likelihood = {}

        try:
            item_likelihood = self.serialized.load_object("item_likelihood")
            print("loading item likelihood success")
            
        except FileNotFoundError:
            print("calculating, item likelihood not found")
            for user in range(users):
                for movie in range(movies):
                    if rating[user, movie] == -1:
                        for qualified in range(r):
                            collaborative_filtering = self.get_movie_rating(
                                rating=rating, movie=movie, qualified=qualified + 1
                            )

                            for j_movie in range(movies):
                                if movie == j_movie or rating[user, j_movie] == -1:
                                    continue

                                item_likelihood[user, movie, qualified, j_movie] = (
                                    np.sum(
                                        np.logical_and(
                                            self.get_movie_rating(
                                                rating=rating,
                                                movie=j_movie,
                                                qualified=rating[user, j_movie],
                                            ),
                                            collaborative_filtering,
                                        )
                                    )
                                    + alpha
                                ) / (
                                    np.sum(
                                        np.logical_and(
                                            self.get_ui(rating=rating, movie=j_movie),
                                            collaborative_filtering,
                                        )
                                    )
                                    + r_alpha
                                )
            print("calculating item likelihood success")
            self.serialized.save_object(item_likelihood, "item_likelihood")

        user_likelihood = {}

        try:
            user_likelihood = self.serialized.load_object("user_likelihood")
            print("loading user likelihood success")
        except FileNotFoundError:
            print("calculating, user likelihood not found")

            for movie in range(movies):
                for user in range(users):
                    if rating[user, movie] == -1:
                        for qualified in range(r):
                            collaborative_filtering = self.get_user_rating(
                                rating=rating, user=user, qualified=qualified + 1
                            )

                            for j_user in range(users):
                                if user == j_user or rating[j_user, movie] == -1:
                                    continue

                                user_likelihood[movie, user, qualified, j_user] = (
                                    np.sum(
                                        np.logical_and(
                                            self.get_user_rating(
                                                rating=rating,
                                                user=j_user,
                                                qualified=rating[j_user, movie],
                                            ),
                                            collaborative_filtering,
                                        )
                                    )
                                    + alpha
                                ) / (
                                    np.sum(
                                        np.logical_and(
                                            self.get_iu(rating=rating, user=j_user),
                                            collaborative_filtering,
                                        )
                                    )
                                    + r_alpha
                                )
            print("calculating user likelihood success")
            self.serialized.save_object(user_likelihood, "user_likelihood")

        return pu, pi, item_likelihood, user_likelihood

    def predict_item(self, rating: np.ndarray, pi: np.ndarray, item_likelihood, r: int):
        """
        Predicts the rating of items based on user ratings and item likelihood.

        Args:
            rating (np.ndarray): Array of user ratings.
            pi (np.ndarray): Array of item probabilities.
            item_likelihood (np.ndarray): The likelihood of items.
            r (int): The number of qualified items.

        Returns:
            np.ndarray: Array of predicted ratings.
        """
        print("predicting item")
        users, movies = rating.shape
        prediction = np.zeros((users, movies, r))
        for user in range(users):
            for movie in range(movies):
                for qualified in range(r):
                    tmp = pi[movie, qualified]
                    for j_movie in range(movies):
                        if movie == j_movie or rating[user, j_movie] == -1:
                            continue
                        tmp *= item_likelihood[
                            user, movie, qualified, j_movie
                        ]  # review
                    prediction[user, movie, qualified] = tmp
        return prediction

    def predict_user(self, rating: np.ndarray, pu: np.ndarray, user_likelihood, r: int):
        """
        Predicts the rating of users based on item ratings and user likelihood.

        Args:
            rating (np.ndarray): Array of item ratings.
            pu (np.ndarray): Array of user probabilities.
            user_likelihood (np.ndarray): The likelihood of users.
            r (int): The number of qualified users.

        Returns:
            np.ndarray: Array of predicted ratings.
        """
        users, movies = rating.shape
        prediction = np.zeros((users, movies, r))
        print("predicting user")
        for user in range(users):
            for movie in range(movies):
                for qualified in range(r):
                    tmp = pu[user, qualified]
                    for j_user in range(users):
                        if user == j_user or rating[j_user, movie] == -1:
                            continue
                        tmp *= user_likelihood[movie, user, qualified, j_user]  # review
                    prediction[user, movie, qualified] = tmp
        return prediction

    def predict_hybrid(self, rating: np.ndarray, r, predict_item, predict_user):
        print("predicting hybrid")
        users, movies = rating.shape
        prediction = np.zeros((users, movies, r))
        for user in range(users):
            for movie in range(movies):
                ui = np.sum(rating[:, movie] != -1)
                iu = np.sum(rating[user, :] != -1)
                for qualified in range(r):
                    prediction[user, movie, qualified] = predict_user[
                        user, movie, qualified
                    ] ** (1 / (1 + ui)) * predict_item[user, movie, qualified] ** (
                        1 / (1 + iu)
                    )
        return prediction


# rating = np.array(
#     [
#         [-1, 1, 2, 2, 5, -1, 4, 3, 5],
#         [1, 5, 3, -1, 2, 4, 4, 3, -1],
#         [1, 1, 2, -1, 2, 4, 4, 5, -1],
#         [3, 2, 2, 3, -1, 1, 3, 2, -1],
#         [5, 1, 5, 5, 4, 4, 5, 2, -1],
#     ]
# )

# # ALPHA = 0.01
# # R = 5

# # a, b, c, d = nbcf(rating=rating, alpha=ALPHA, r=R)
# # print(a)
# # print(" - - - - - ")
# # print(b)

# # print(" =========================================")

# # print(c[0, 0, 0])

# # print(" =========================================")

# # print(d[0, 0, 0])


# # print(" =========================================")

# # pi = predict_item(rating, b, c, R)
# # print(pi[0, 0])

# # print(" =========================================")

# # pu = predict_user(rating, a, d, R)
# # print(pu[0, 0])

# # print(" =========================================")

# # ph = predict_hybrid(rating, R, pi, pu)
# # print(ph[0, 0])
