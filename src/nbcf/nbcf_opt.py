import numpy as np
import time


class NBCF:
    """
    NBCF (Naive Bayes Collaborative Filtering) class for recommendation system.

    Args:
        rating (np.ndarray): Matrix of user ratings.
        alpha (float): Smoothing parameter for prior probabilities.
        r (float): Regularization parameter.
        qualified_array (list[int]): List of qualified ratings.
        load (bool, optional): Whether to load pre-trained model. Defaults to False.

    Attributes:
        rating (np.ndarray): Matrix of user ratings.
        alpha (float): Smoothing parameter for prior probabilities.
        r (float): Regularization parameter.
        qualified_array (list[int]): List of qualified ratings.
        users (int): Number of users.
        movies (int): Number of movies.
        pu (np.ndarray): User-based prior probability matrix.
        pi (np.ndarray): Item-based prior probability matrix.
        m_collaborative_filtering (dict): Collaborative filtering dictionary for movies.
        u_collaborative_filtering (dict): Collaborative filtering dictionary for users.
        user_map (list[set]): Mapping of movies for each user.
        movie_map (list[set]): Mapping of users for each movie.
        r_alpha (float): Regularization parameter multiplied by alpha.
        item_prediction (np.ndarray): Item-based prediction matrix.
        user_prediction (np.ndarray): User-based prediction matrix.
        prediction (np.ndarray): Hybrid prediction matrix.

    Methods:
        nbcf(): Performs NBCF algorithm.
        predict_hybrid(): Performs hybrid prediction using NBCF.

    """

    EPSILON = np.finfo(float).eps

    def __init__(self, rating, alpha, r, qualified_array, load=False) -> None:
        """
        Initialize the NBCF class.

        Args:
            rating (np.ndarray): Matrix of user ratings.
            alpha (float): Smoothing parameter for prior probabilities.
            r (float): Regularization parameter.
            qualified_array (list[int]): List of qualified ratings.
            load (bool, optional): Whether to load pre-trained model. Defaults to False.
        """
        self.rating: np.ndarray = rating
        self.alpha: float = alpha
        self.r: float = r
        self.qualified_array: list[int] = qualified_array
        self.users, self.movies = self.rating.shape

        self.pu = np.zeros((self.users, len(self.qualified_array)), float)
        self.pi = np.zeros((self.movies, len(self.qualified_array)), float)

        self.m_collaborative_filtering = {}
        self.u_collaborative_filtering = {}

        self.user_map = [set() for _ in range(self.users)]
        self.movie_map = [set() for _ in range(self.movies)]

        self.r_alpha = self.r * self.alpha

        self.item_prediction = np.zeros(
            (self.users, self.movies, len(self.qualified_array)), float
        )
        self.user_prediction = np.zeros(
            (self.users, self.movies, len(self.qualified_array)), float
        )

        self.prediction = np.zeros((self.users, self.movies, r), float)

        self.nbcf()

        print("Initialize Hybrid")
        self.predict_hybrid()

    def nbcf(self):
        """
        Performs the NBCF algorithm.
        """

        d = time.time()
        print("0 ⏰ Mapping movies and users ...")

        # Map movies and users for faster navigation
        for user in range(self.users):
            self.user_map[user] = set()
            for movie in range(self.movies):
                if self.rating[user][movie] != -1:
                    self.user_map[user].add(movie)
                    self.movie_map[movie].add(user)

        print(time.time() - d, "⏰ Mapped movies and users ...")
        d = time.time()

        # Prior probability user based
        for movie in range(self.movies):

            d = len(self.movie_map[movie]) + self.r_alpha
            for qualified in self.qualified_array:
                self.m_collaborative_filtering[(movie, qualified)] = [
                    i
                    for i in range(self.users)
                    if self.rating[i, movie] == qualified + 1
                ]
                self.pi[movie, qualified] = (
                    len(self.m_collaborative_filtering[(movie, qualified)]) + self.alpha
                ) / d

        print(time.time() - d, "⏰ Prior probability user based End...")
        d = time.time()
        # Prior probability item based
        for user in range(self.users):
            d = len(self.user_map[user]) + self.r_alpha
            for qualified in self.qualified_array:
                self.u_collaborative_filtering[(user, qualified)] = [
                    i
                    for i in range(self.movies)
                    if self.rating[user, i] == qualified + 1
                ]
                self.pu[user, qualified] = (
                    len(self.u_collaborative_filtering[(user, qualified)]) + self.alpha
                ) / d

        print(time.time() - d, "⏰ Prior probability item based End ...")
        d = time.time()
        print("Start Calculating Predictions ...")
        # Calculate predictions

        for user in range(self.users):
            print(user, " of ", self.users)
            for movie in range(self.movies):

                if self.rating[user][movie] != -1:
                    continue

                for qualified in self.qualified_array:

                    # Item based prediction
                    tmp = self.pi[movie, qualified]

                    for j_movie in self.user_map[user]:
                        numerator = 0
                        denominator = 0
                        for u in self.m_collaborative_filtering[(movie, qualified)]:
                            if self.rating[u, j_movie] == -1:
                                continue
                            denominator += 1
                            if self.rating[u, j_movie] == self.rating[user, j_movie]:
                                numerator += 1

                        tmp *= (numerator + self.alpha) / (denominator + self.r_alpha)

                    self.item_prediction[user, movie, qualified] = tmp

                    # User based prediction
                    tmp = self.pu[user, qualified]

                    for j_user in self.movie_map[movie]:
                        numerator = 0
                        denominator = 0
                        for m in self.u_collaborative_filtering[(user, qualified)]:
                            if self.rating[j_user, m] == -1:
                                continue
                            denominator += 1
                            if self.rating[j_user, m] == self.rating[j_user, movie]:
                                numerator += 1

                        tmp *= (numerator + self.alpha) / (denominator + self.r_alpha)

                    self.user_prediction[user, movie, qualified] = tmp

        print(time.time() - d, "⏰ Calculated Predictions ...")

    def predict_hybrid(self):
        """
        Predicts the ratings for unrated movies using a hybrid approach.

        This method iterates over each user and each movie in the dataset. If a rating for a particular user-movie pair is not available (indicated by -1), it calculates the prediction using a hybrid approach. The hybrid approach combines user-based and item-based predictions.

        Parameters:
            None

        Returns:
            None
        """
        for user in range(self.users):
            for movie in range(self.movies):
                if self.rating[user][movie] != -1:
                    continue
                ui = len(self.movie_map[movie])
                iu = len(self.user_map[user])
                for qualified in self.qualified_array:

                    left = (
                        self.user_prediction[user, movie, qualified] ** (1 / (1 + ui))
                        + self.EPSILON
                    )
                    right = (
                        self.item_prediction[user, movie, qualified] ** (1 / (1 + iu))
                        + self.EPSILON
                    )

                    self.prediction[user, movie, qualified] = left * right
                    if self.prediction[user, movie, qualified] == 0:
                        print("user", user, "movie", movie, "qualified", qualified)


# Test function for the NBCF class


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

nbcfTest = NBCF(rating=rating, alpha=ALPHA, r=R, qualified_array=range(R))

item_prediction = nbcfTest.item_prediction


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

user_prediction = nbcfTest.user_prediction


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

ph = nbcfTest.prediction


attempt(
    np.allclose(
        ph[0, 0],
        [0.00993204, 0.01207249, 0.00089421, 0.01198044, 0.00128937],
    ),
    True,
    "hybrid test",
)


print(ph[0, 0])
