import numpy as np

def generate_groups(rating, threshold):
    
    groups = {}

    for u in range(rating.shape[0]):
        for m in range(rating.shape[1]):
            if rating[u][m] == 5:
                y = rating[u][m]
                groups.setdefault((m, y), []).append(u)

    ans = {}

    for key, users in groups.items():
        if len(users) > threshold:
            ans[key] = users

    

    return ans


