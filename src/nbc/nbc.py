import numpy as np

ranking_ui: dict = {  # key -> user, value -> { key -> item, value -> rating }
    0: {
        1: 1,
        2: 2,
        3: 2,
        4: 5,
        6: 4,
        7: 3,
        8: 5,
    },
    1: {
        0: 1,
        1: 5,
        2: 3,
        4: 2,
        5: 3,
        6: 4,
        7: 3,
    },
    2: {
        0: 1,
        1: 1,
        2: 2,
        4: 2,
        5: 4,
        6: 4,
        7: 5,
    },
    3: {
        0: 3,
        1: 2,
        2: 2,
        3: 3,
        5: 1,
        6: 3,
        7: 2,
    },
    4: {
        0: 5,
        1: 1,
        2: 5,
        3: 5,
        4: 4,
        5: 4,
        6: 5,
        7: 2,
    },
}


def nbcf(ranking_ui: dict, alpha: float, r: float):
    # initialize
    pup = {}
    pip = {}
    cup = {}
    cip = {}

    # tmp
    uc = {}
    ic = {}
    ijc = {}
    uvc = {}

    # R = max([max(ranking_ui[u].values()) for u in ranking_ui])  # max rating

    ranking_ui_inverse = {}

    for user in ranking_ui:
        for item in ranking_ui[user]:
            if item not in ranking_ui_inverse:
                ranking_ui_inverse[item] = set()
            ranking_ui_inverse[item].add(user)

    r_alpha = r * alpha

    pup = np.full((len(ranking_ui), r + 1), alpha)
    uc = np.full(len(ranking_ui), r_alpha)
    pip = np.full((len(ranking_ui_inverse), r + 1), alpha)
    ic = np.full(len(ranking_ui_inverse), r_alpha)

    cip = np.full(
        (len(ranking_ui_inverse), r + 1, len(ranking_ui_inverse), r + 1), alpha
    )
    ijc = np.full((len(ranking_ui_inverse), len(ranking_ui_inverse), r + 1), r_alpha)
    cup = np.full((len(ranking_ui), r + 1, len(ranking_ui), r + 1), alpha)
    uvc = np.full((len(ranking_ui), len(ranking_ui), r + 1), r_alpha)

    for user in ranking_ui:
        for item in ranking_ui[user]:
            y = ranking_ui[user][item]

            pup[user][y] = (uc[user] * pup[user][y] + 1) / (uc[user] + 1)
            uc[user] = uc[user] + 1
            pip[item][y] = (ic[item] * pip[item][y] + 1) / (ic[item] + 1)
            ic[item] = ic[item] + 1

            for j_item in ranking_ui[user]:
                # if j_item == item:  # check if the item is the same
                #     continue

                k = ranking_ui[user][j_item]

                cip[j_item][k][item][y] = (
                    ijc[j_item][item][y] * cip[j_item][k][item][y] + 1
                ) / (ijc[j_item][item][y] + 1)

                ijc[j_item][item][y] = ijc[j_item][item][y] + 1

            for v_user in ranking_ui_inverse[item]:
                # if v_user == user:  # check if the user is the same
                #     continue

                k = ranking_ui[v_user][item]

                cup[v_user][k][user][y] = (
                    uvc[v_user][user][y] * cup[v_user][k][user][y] + 1
                ) / (uvc[v_user][user][y] + 1)
                uvc[v_user][user][y] = uvc[v_user][user][y] + 1

    return pup, pip, cup, cip


a, b, c, d = nbcf(ranking_ui, 0.01, 5)

print("PUP\n", a)
print("PIP\n", b)
# print("CUP\n", c)
# print("CIP\n", d)
