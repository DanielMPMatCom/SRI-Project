import numpy as np


def nbcf(
    rating: dict,
    inverse_rating: dict,
    alpha: float,
    r: int,
) -> tuple:
    """
    Performs Naive Bayes Collaborative Filtering (NBCF) algorithm.

    Args:
        rating (dict): Dictionary containing user-item ratings.
        inverse_rating (dict): Dictionary containing item-user ratings.
        alpha (float): Parameter alpha for smoothing.
        r (int): Parameter r for smoothing.

    Returns:
        tuple: A tuple containing the following dictionaries:
            - pup: Dictionary of user-item probabilities.
            - pip: Dictionary of item-item probabilities.
            - cip: Dictionary of conditional item-item probabilities.
            - cup: Dictionary of conditional user-user probabilities.
    """

    if alpha < 0 or 1 < alpha:
        raise ValueError("Parameter alpha must be in [0, 1]")

    # Use get for default value of the dictionaries with alpha
    pup = {}
    pip = {}
    cup = {}
    cip = {}

    r_alpha = r * alpha

    # Use get for default value of the dictionaries with r_alpha
    uc = {}
    ic = {}
    ijc = {}
    uvc = {}

    for user in rating.keys():  # For each user in the rating
        for item in rating[user]:  # For each item rated by the user

            y = rating[user][item]  # Rating given by the user to the item

            pup[user, y] = (pup.get((user, y), alpha) * uc.get(user, r_alpha) + 1) / (
                uc.get(user, r_alpha) + 1
            )

            uc[user] = uc.get(user, r_alpha) + 1

            pip[item, y] = (pip.get((item, y), alpha) * ic.get(item, r_alpha) + 1) / (
                ic.get(item, r_alpha) + 1
            )

            ic[item] = ic.get(item, r_alpha) + 1

            for j_item in rating[user]:  # For each item rated by the user

                k = rating[user][j_item]

                cip[j_item, k, item, y] = (
                    ijc.get((j_item, item, y), r_alpha)
                    * cip.get((j_item, k, item, y), alpha)
                    + 1
                ) / (ijc.get((j_item, item, y), r_alpha) + 1)

                ijc[j_item, item, y] = ijc.get((j_item, item, y), r_alpha) + 1

            for v_user in inverse_rating[item]:  # For each user that rated the item

                k = inverse_rating[item][v_user]

                cup[v_user, k, user, y] = (
                    uvc.get((v_user, user, y), r_alpha)
                    * cup.get((v_user, k, user, y), alpha)
                    + 1
                ) / (uvc.get((v_user, user, y), r_alpha) + 1)

                uvc[v_user, user, y] = uvc.get((v_user, user, y), r_alpha) + 1

    return pup, pip, cip, cup  # Return the probabilities


rating = {
    1: {2: 1, 3: 2, 4: 2, 5: 5, 7: 4, 8: 3, 9: 5},
    2: {1: 2, 2: 5, 3: 3, 5: 2, 6: 3, 7: 4, 8: 3},
    3: {1: 1, 2: 1, 3: 2, 5: 2, 6: 4, 7: 4, 8: 5},
    4: {1: 3, 2: 2, 3: 2, 4: 3, 6: 1, 7: 3, 8: 2},
    5: {1: 5, 2: 1, 3: 5, 4: 5, 5: 4, 6: 4, 7: 5, 8: 2},
}  # Example of rating, the keys are the users, and the values are the items rated by the user with the rating given

inverse_rating = (
    {}
)  # Inverse rating, the keys are the items, and the values are the users that rated the item with the rating given

for user in rating:
    for item in rating[user]:
        inverse_rating.setdefault(item, {})
        inverse_rating[item][user] = rating[user][item]

pup, pip, cip, cup = nbcf(
    rating=rating,
    inverse_rating=inverse_rating,
    alpha=0.01,
    r=5,
)

for i in range(1, 6):  # For each rating
    print(pup.get((1, i), 0))


# print(hybrid(rating, inverse_rating, pup, pip, cip, cup, 1, 2, 5))


# def hybrid(rating, inverse_rating, pup, pip, cip, cup, user, movie, max_rating):

#     max_y = -1
#     max_value = -1
#     for y in range(1, max_rating + 1):
#         productory_user = [
#             cup[v_user, rating[v_user][movie], user, y]
#             for v_user in inverse_rating[movie]
#         ]
#         productory_user_value = np.prod(productory_user)

#         productory_item = [
#             cip[item, rating[user][item], movie, y] for item in rating[user]
#         ]
#         productory_item_value = np.prod(productory_item)

#         value = (pup[user, y] * productory_user_value) ** (
#             1 / (1 + len(productory_user))
#         ) + (pip[movie, y] * productory_item_value) ** (1 / (1 + len(productory_item)))

#         if value > max_value:
#             max_value = value
#             max_y = y

#     return max_y
