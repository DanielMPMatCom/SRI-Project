import numpy as np

# This class is under development and is not yet used in the project.
# The purpose of this class is check result of nbcf paper.


def user_mae(hybrid_prediction: np.ndarray, test, user):
    """
    Calculate the Mean Absolute Error (MAE) for a specific user.

    Parameters:
    hybrid_prediction (np.ndarray): The hybrid prediction matrix.
    test (list): The test dataset.
    user (int): The user ID.

    Returns:
    float: The MAE for the specific user.
    """
    mae = 0
    user_test = [x for x in test if x[0] == user]
    for u, m, r in user_test:
        mae += abs(hybrid_prediction[u, m].argmax() + 1 - r)
    return mae / len(user_test)


def mae(hybrid_prediction: np.ndarray, test):
    """
    Calculate the Mean Absolute Error (MAE) for all users.

    Parameters:
    hybrid_prediction (np.ndarray): The hybrid prediction matrix.
    test (list): The test dataset.

    Returns:
    float: The MAE for all users.
    """
    mae = 0
    users = np.unique([x[0] for x in test])
    for user in users:
        mae += user_mae(hybrid_prediction, test, user)
    return mae / len(users)


def user_precision(hybrid_prediction: np.ndarray, test, user, troubleshoot_value=3):
    """
    Calculate the precision for a specific user.

    Parameters:
    hybrid_prediction (np.ndarray): The hybrid prediction matrix.
    test (list): The test dataset.
    user (int): The user ID.
    troubleshoot_value (int, optional): The threshold value for troubleshooting. Defaults to 3.

    Returns:
    float: The precision for the specific user.
    """
    precision = 0
    user_test = [x for x in test if x[0] == user]
    N = 0
    for u, m, r in user_test:
        if hybrid_prediction[u, m].argmax() + 1 >= troubleshoot_value:
            N += 1
            if r >= troubleshoot_value:
                precision += 1

    return precision / N


def precision(hybrid_prediction: np.ndarray, test, troubleshoot_value=3):
    """
    Calculate the precision of a hybrid prediction model.

    Parameters:
    - hybrid_prediction (np.ndarray): The hybrid prediction array.
    - test: The test data.
    - troubleshoot_value (int): The value used for troubleshooting.

    Returns:
    - float: The precision value.
    """
    precision = 0
    users = np.unique([x[0] for x in test])
    for user in users:
        precision += user_precision(hybrid_prediction, test, user, troubleshoot_value)
    return precision / len(users)


def user_recall(hybrid_prediction: np.ndarray, test, user, troubleshoot_value=3):
    """
    Calculate the recall metric for a given user in a recommendation system.

    Parameters:
    - hybrid_prediction (np.ndarray): The hybrid prediction matrix.
    - test: The test dataset.
    - user: The user for whom to calculate the recall.
    - troubleshoot_value (int): The minimum rating value to consider for recall calculation.

    Returns:
    - float: The recall metric for the given user.
    """

    recall = 0
    N = 0
    user_test = [x for x in test if x[0] == user]
    for u, m, r in user_test:
        if (
            hybrid_prediction[u, m].argmax() + 1 >= troubleshoot_value
            and r >= troubleshoot_value
        ):
            recall += 1
        if r >= troubleshoot_value:
            N += 1

    return recall / N


def DCGu(hybrid_prediction: np.ndarray, test, user):
    """
    Calculate the Discounted Cumulative Gain (DCG) for a specific user.

    Parameters:
    hybrid_prediction (np.ndarray): The hybrid prediction matrix.
    test (list): The test dataset.
    user (int): The user ID.

    Returns:
    float: The DCG for the specific user.
    """
    user_hybrid_movie_scores = []
    for u, m, r in test:
        if u == user:
            user_hybrid_movie_scores.append(hybrid_prediction[u, m].argmax() + 1)

    user_hybrid_movie_scores = sorted(
        user_hybrid_movie_scores, key=lambda x: x[0], reverse=True
    )

    dcgu = 0

    for i, r in enumerate(user_hybrid_movie_scores):
        dcgu += 2 ** (r - 1) - 1 / np.log2(i + 2)

    return dcgu


def IDCGu(test, user):
    """
    Calculate the Ideal Discounted Cumulative Gain (IDCG) for a specific user.

    Parameters:
    test (list): The test dataset.
    user (int): The user ID.

    Returns:
    float: The IDCG for the specific user.
    """
    real_scores = []
    for u, _, r in test:
        if u == user:
            real_scores.append(r)

    user_hybrid_movie_scores = sorted(
        user_hybrid_movie_scores, key=lambda x: x[1], reverse=True
    )

    idcgu = 0

    for i, r in enumerate(user_hybrid_movie_scores):
        idcgu += 2 ** (r - 1) - 1 / np.log2(i + 2)

    return idcgu


def nDCGu(hybrid_prediction: np.ndarray, test, troubleshoot_value=3):
    """
    Calculate the normalized Discounted Cumulative Gain (nDCG) for all users.

    Parameters:
    hybrid_prediction (np.ndarray): The hybrid prediction matrix.
    test (list): The test dataset.
    troubleshoot_value (int, optional): The threshold value for troubleshooting. Defaults to 3.

    Returns:
    float: The nDCG for all users.
    """
    nDCGu = 0
    users = np.unique([x[0] for x in test])
    for user in users:
        nDCGu += DCGu(hybrid_prediction, test, user, troubleshoot_value) / IDCGu(
            test, user
        )
    return nDCGu


def NDCG(hybrid_prediction: np.ndarray, test, troubleshoot_value=3):
    """
    Calculate the normalized Discounted Cumulative Gain (nDCG) for all users.

    Parameters:
    hybrid_prediction (np.ndarray): The hybrid prediction matrix.
    test (list): The test dataset.
    troubleshoot_value (int, optional): The threshold value for troubleshooting. Defaults to 3.

    Returns:
    float: The nDCG for all users.
    """
    nDCG = 0
    users = np.unique([x[0] for x in test])
    for user in users:
        nDCG += DCGu(hybrid_prediction, test, user, troubleshoot_value) / IDCGu(
            test, user
        )
    return nDCG / len(users)
