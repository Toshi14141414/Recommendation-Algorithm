import random

def RandomSelectNegativeSample(self, items):
    items_pool = range(270)
    ret = dict()
    for i in items.keys():
        ret[i] = 1
    n = 0
    for i in range(0, len(items) * 3):
        item = items_pool[random.randint(0, len(items_pool)-1)]
        if item in ret:
            continue
        ret[item] = 0
        n += 1
        if n > len(items):
            break
    return ret

def LatentFactorModel(user_items, F=100, N, alpha=0.02, lamb=0.01):
    P, Q = InitModel(user_items, F)#InitModel
    for step in range(0, N):
        for user, items in user_items.items():
            samples = RandomSelectNegativeSample(items)
            for item, rui in samples.items():
                eui = rui - Predict(user, item, P, Q, F) # Predict
                for f in range(F):
                    P[user][f] += alpha*(eui*Q[item][f] - lamb*P[user][f])
                    Q[item][f] += alpha*(eui*P[user][f] - lamb*Q[item][f])
        alpha *= 0.9
        print cost
    return P, Q

def Predict(user, item, P, Q, F):
    ret = 0
    for f in range(F):
        ret += P[user][f] * Q[item][f]
    return ret

def InitModel(user_items, F):
    P = dict()
    Q = dict()
    for u, items in user_items.items():
        if u not in P:
            P[u] = dict()
        for f in range(F):
            P[u][f] = 1

        for i in items[0]:
            if i not in Q:
                Q[i] = dict()
            for f in range(F):
                Q[i][f] = 1
    return P, Q



def Recommend(user, P, Q):
    rank = dict()
    for f, puf in P[user].items():
        for i, qfi in P[f].items():
            if i not in rank:
                rank[i] += puf * qfi
    return rank