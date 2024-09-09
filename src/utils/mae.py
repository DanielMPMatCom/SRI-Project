import numpy as np


def user_mae(hybrid_prediction: np.ndarray, test, user):
    mae = 0
    user_test = [x for x in test if x[0] == user]
    for u, m, r in user_test:
        mae += abs(hybrid_prediction[u, m].argmax() - r)
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
        if hybrid_prediction[u, m].argmax() >= troubleshoot_value:
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
            hybrid_prediction[u, m].argmax() >= troubleshoot_value
            and r >= troubleshoot_value
        ):
            recall += 1
        if r >= troubleshoot_value:
            N += 1

    return recall / N


def nDCG(hybrid_prediction: np.ndarray, test, user, troubleshoot_value=3):
    nDCG = 0
    user_test = [x for x in test if x[0] == user]
    for u, m, r in user_test:
        if hybrid_prediction[u, m].argmax() >= troubleshoot_value:
            nDCG += 1 / np.log2(2 + hybrid_prediction[u, m].argmax())
    return nDCG
