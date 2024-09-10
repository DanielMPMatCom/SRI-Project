import numpy as np

def generate_groups(rating, threshold, rated):
    
    groups = {}

    for u in range(rating.shape[0]):
        for m in range(rating.shape[1]):
            if rating[u][m] in rated:
                y = rating[u][m]
                groups.setdefault((m, y), []).append(u)

    ans = {}

    for key, users in groups.items():
        if len(users) > threshold:
            ans[key] = users

    

    return ans


