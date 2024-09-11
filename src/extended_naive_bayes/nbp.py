import numpy as np


def group_prediction(
    rating: np.ndarray,
    group: np.ndarray,
    hp: np.ndarray,
    qualified_array: "list[int]",
    movie: int,
):
    """
    Calculate the group prediction for a given movie based on ratings, group information, and hybrid probabilities.
    Parameters:
        rating (np.ndarray): Array of ratings.
        group (np.ndarray): Array of group information.
        hp (np.ndarray): Array of hybrid probabilities.
        qualified_array (list[int]): List of qualified ratings.
        movie (int): Index of the movie.
    Returns:
        np.ndarray: Array of group predictions for the given movie.
    """

    _, movies = rating.shape
    group_prediction_hf = np.zeros((movies, len(qualified_array)))

    for item in [movie]:

        for qualified in qualified_array:

            hybrid_based = 0
            inverted_hybrid_based = 0

            for g_user in group:
                if rating[g_user][item] != -1:
                    continue

                hybrid_based += np.log(hp[g_user, item, qualified])

                inverted_hybrid_based += np.log(
                    np.sum(
                        [hp[g_user, item, q] for q in qualified_array if q != qualified]
                    )
                )

            group_prediction_hf[item, qualified] = (hybrid_based) - (
                hybrid_based + inverted_hybrid_based
            )

    return group_prediction_hf
