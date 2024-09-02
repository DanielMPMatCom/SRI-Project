import numpy as np


def nbcf(rating: np.ndarray, alpha: float):
    if alpha < 0 or 1 < alpha:
        raise Exception("Parameter alpha must be in [0, 1]")

    pup = {}
    pip = {}
    cup = {}
    cip = {}
    r_alpha = np.max(rating) * alpha
    count_user, count_item = rating.shape

    uc = {}
    ic = {}
    ijc = {}
    uvc = {}

    for user in range(count_user):
        for item in range(count_item):
            if rating[user, item] != -1:
                y = rating[user, item]
                pup[user, y] = (
                    pup.get((user, y), alpha) * uc.get(user, r_alpha) + 1
                ) / (uc.get(user, r_alpha) + 1)
                uc[user] = uc.get(user, r_alpha) + 1
                pip[item, y] = (
                    pip.get((item, y), alpha) * ic.get(item, r_alpha) + 1
                ) / (ic.get(item, r_alpha) + 1)
                ic[item] = ic.get(item, r_alpha) + 1

                for j_item in range(count_item):
                    if rating[user, j_item] != -1:
                        k = rating[user, j_item]
                        cip[j_item, k, item, y] = (
                            ijc.get((j_item, item, y), r_alpha)
                            * cip.get((j_item, k, item, y), alpha)
                            + 1
                        ) / (ijc.get((j_item, item, y), r_alpha) + 1)
                        ijc[j_item, item, y] = ijc.get((j_item, item, y), r_alpha) + 1

                for v_user in range(count_user):
                    if rating[v_user, item] != -1:
                        k = rating[v_user, item]
                        cup[v_user, k, user, y] = (
                            uvc.get((v_user, user, y), r_alpha)
                            * cup.get((v_user, k, user, y), alpha)
                            + 1
                        ) / (uvc.get((v_user, user, y), r_alpha) + 1)
                        uvc[v_user, user, y] = uvc.get((v_user, user, y), r_alpha) + 1

    return pup, pip, cip, cup

def hybrid(rating, pup, pip, cip, cup, user, movie, max_rating):
    iu = np.sum(rating[user, :] != -1)
    ui = np.sum(rating[:, movie] != -1)

    max_y = -1
    max_value = -1
    for y in range(max_rating + 1):
        productory = [cup[v_user, rating[user,movie], user, y] for v_user in range(rating.shape[0]) if rating[v_user, movie] == k]
        for i_user in range(rating.shape[0]):
            if rating[i_user, movie] != -1:
                
            
