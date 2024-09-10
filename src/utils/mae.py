import numpy as np


def user_mae(hybrid_prediction: np.ndarray, test, user):
    mae = 0
    user_test = [x for x in test if x[0] == user]
    for u, m, r in user_test:
        mae += abs(hybrid_prediction[u, m].argmax() + 1 - r)
    return mae / len(user_test)


def mae(hybrid_prediction: np.ndarray, test):  # revisar esto
    mae = 0
    users = np.unique([x[0] for x in test])
    for user in users:
        mae += user_mae(hybrid_prediction, test, user)
    return mae / len(users)


def user_precision(hybrid_prediction: np.ndarray, test, user, troubleshoot_value=3):
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
    precision = 0
    users = np.unique([x[0] for x in test])
    for user in users:
        precision += user_precision(hybrid_prediction, test, user, troubleshoot_value)
    return precision / len(users)


def user_recall(hybrid_prediction: np.ndarray, test, user, troubleshoot_value=3):
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
    nDCGu = 0
    users = np.unique([x[0] for x in test])
    for user in users:
        nDCGu += DCGu(hybrid_prediction, test, user, troubleshoot_value) / IDCGu(
            test, user
        )
    return nDCGu


def NDCG(hybrid_prediction: np.ndarray, test, troubleshoot_value=3):
    nDCG = 0
    users = np.unique([x[0] for x in test])
    for user in users:
        nDCG += DCGu(hybrid_prediction, test, user, troubleshoot_value) / IDCGu(
            test, user
        )
    return nDCG / len(users)
