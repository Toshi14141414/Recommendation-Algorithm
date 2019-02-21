import numpy as np
import math
#import matplotlib

_NUMofITEM = 269
_NUMofUSER = 335225

#ItemCF Class
class ItemSimilarity(object):
    def __init__(self, numOfItems, numOfUsers):
        self.itemSimilarity = dict()
        self.num_items_user = dict()
        for i in range(numOfItems):
            self.itemSimilarity[i] = dict()
            for j in range(numOfItems):
                self.itemSimilarity[i][j] = 0
        self.co_rated = self.itemSimilarity
        for i in range(numOfItems):
            self.num_items_user[i] = 0

    def itemCF(self, user_items_table):
        #calculate co-rated users between items
        for user, items in user_items_table.items():
            for i in items:
                self.num_items_user[i] += 1
                for j in items:
                    if i == j:
                        continue
                    self.co_rated[i][j] += 1

    def itemCF_IUF(self, user_items_table):
        #calculate co-rated users between items
        for user, items in user_items_table.items():
            for i in items:
                self.num_items_user[i] += 1
                for j in items:
                    if i == j:
                        continue
                    self.co_rated[i][j] += 1 / math.log(1 + len(items) * 1.0)

    def itemCF_IUF_TD(self, user_items_table, alpha):
        #calculate co-rated users between items
        for user, items in user_items_table.items():
            for i in items:
                self.num_items_user[i] += 1
                for j in items:
                    if i == j:
                        continue
                    self.co_rated[i][j] += 1 / math.log(1 + len(items) * 1.0)

    def cal_itemSimilarity(self):
        #calculate final similarity matrix
        for i in range(len(self.co_rated)):
            for j in range(len(self.co_rated[i])):
                self.itemSimilarity[i][j] = self.co_rated[i][j] / math.sqrt(self.num_items_user[i] * self.num_items_user[j])

    def normolized_itemSimilarity(self):
        #normalize similarity matrix
        for i in range(len(self.itemSimilarity)):
            max_sim = sorted(self.itemSimilarity[i].iteritems(), key=lambda d:d[1], reverse=True)[0][1]
            for j in range(len(self.itemSimilarity[i])):
                self.itemSimilarity[i][j] = self.itemSimilarity[i][j] / max_sim * 1.0

# Recommendation function
def Recommendation(test_set, itemSimilarity, K, N):
    ret = dict()
    tar = dict()
    for user, items in test_set.items():
        rank = dict()
        n = int(len(items)/3)
        target = items[-n:]
        ru = items[:-n]
        for i in ru:
            for j in sorted(itemSimilarity[i].iteritems(), key=lambda d:d[1], reverse=True)[0:K]:# sorted topK
                if j[0] in ru:
                    continue
                if not rank.has_key(j[0]):
                    rank[j[0]] = 0
                rank[j[0]] += itemSimilarity[i][j[0]] * 1.0
        ret[user] = sorted(rank.iteritems(), key=lambda d:d[1], reverse=True)[0:N]
        tar[user] = target
    return ret, tar

#Recommendation function with Location Information
def Recommendation_LD(test_set, itemSimilarity, bigram, skipbigram, K, N):
    ret = dict()
    tar = dict()
    for user, items in test_set.items():
        rank = dict()
        n = int(len(items)/3)
        target = items[-n:]
        ru = items[:-n]
        for i in ru:
            for j in sorted(itemSimilarity[i].iteritems(), key=lambda d:d[1], reverse=True)[0:K]:# sorted topK
                if j[0] in ru:
                    continue
                if not rank.has_key(j[0]):
                    rank[j[0]] = 0
                rank[j[0]] += itemSimilarity[i][j[0]] * (bigram[i][j[0]] + skipbigram[i][j[0]])
        ret[user] = sorted(rank.iteritems(), key=lambda d:d[1], reverse=True)[0:N]
        tar[user] = target
    return ret, tar

# calculate the recall
def Recall(rank, target):
    hit = 0
    total = 0
    for user, items in rank.items():
        for item in items:
            if item[0] in target[user]:
                hit += 1
        total += len(target[user])
    return float(hit) / total

# calculate the precision
def Precision(rank, target):
    hit = 0
    total = 0
    for user, items in rank.items():
        for item in items:
            if item[0] in target[user]:
                hit += 1
        total += len(rank[user])
    return float(hit) / total

# calculate the coverage
def Coverage(rank):
    recommend_items = set()
    for items in rank.values():
        for item in items:
            recommend_items.add(item[0])
    return float(len(recommend_items))/_NUMofITEM

# evaluate ItemCF 
def ItemCF(user_items_train, user_items_test):
    IS = ItemSimilarity(_NUMofITEM, _NUMofUSER)
    IS.itemCF(user_items_train)
    IS.cal_itemSimilarity()
    itemSimilarity = IS.itemSimilarity
    for K in [5,10,20,40,80,160]:
        rank, target = Recommendation(user_items_test, itemSimilarity, K, 10)
        recall = Recall(rank, target)
        precision = Precision(rank, target)
        F = 2.0 * recall * precision / (recall + precision)
        coverage = Coverage(rank)
        print recall, precision, F, coverage

# evaluate ItemCF_IUF
def ItemCF_IUF(user_items_train, user_items_test):
    IS = ItemSimilarity(_NUMofITEM, _NUMofUSER)
    IS.itemCF_IUF(user_items_train)
    IS.cal_itemSimilarity()
    itemSimilarity = IS.itemSimilarity
    for K in [5,10,20,40,80,160]:
        rank, target = Recommendation(user_items_test, itemSimilarity, K, 10)
        recall = Recall(rank, target)
        precision = Precision(rank, target)
        F = 2.0 * recall * precision / (recall + precision)
        coverage = Coverage(rank)
        print recall, precision, F, coverage

# evaluate ItemCF_IUF_Norm
def ItemCF_IUF_Norm(user_items_train, user_items_test):
    IS = ItemSimilarity(_NUMofITEM, _NUMofUSER)
    IS.itemCF_IUF(user_items_train)
    IS.cal_itemSimilarity()
    IS.normolized_itemSimilarity()
    itemSimilarity = IS.itemSimilarity
    for K in [5,10,20,40,80,160]:
        rank, target = Recommendation(user_items_test, itemSimilarity, K, 10)
        recall = Recall(rank, target)
        precision = Precision(rank, target)
        F = 2.0 * recall * precision / (recall + precision)
        coverage = Coverage(rank)
        print recall, precision, F, coverage

# evaluate ItemCF_IUF_TD_Norm
def ItemCF_IUF_TD_Norm(user_items_train, user_items_test):
    IS = ItemSimilarity(_NUMofITEM, _NUMofUSER)
    IS.itemCF_IUF_TD(user_items_train, alpha)
    IS.cal_itemSimilarity()
    IS.normolized_itemSimilarity()
    itemSimilarity = IS.itemSimilarity
    for K in [5,10,20,40,80,160]:
        rank, target = Recommendation(user_items_test, itemSimilarity, K, 10)
        recall = Recall(rank, target)
        precision = Precision(rank, target)
        F = 2.0 * recall * precision / (recall + precision)
        coverage = Coverage(rank)
        print recall, precision, F, coverage

# evaluate ItemCF_IUF_LD_Norm
def ItemCF_IUF_LD_Norm(user_items_train, user_items_test):
    IS = ItemSimilarity(_NUMofITEM, _NUMofUSER)
    IS.itemCF_IUF(user_items_train)
    IS.cal_itemSimilarity()
    IS.normolized_itemSimilarity()
    itemSimilarity = IS.itemSimilarity
    bigram = tools.BiGram(user_items_train)
    skipbigram = tools.SkipBiGram(user_items_train)
    for K in [5,10,20,40,80,160]:
        rank, target = Recommendation_LD(user_items_test, itemSimilarity, bigram, skipbigram, K, 10)
        recall = Recall(rank, target)
        precision = Precision(rank, target)
        F = 2.0 * recall * precision / (recall + precision)
        coverage = Coverage(rank)
        print recall, precision, F, coverage

# evaluate ItemCF_IUF_TLD_Norm
def ItemCF_IUF_TLD_Norm(user_items_train, user_items_test):
    IS = ItemSimilarity(_NUMofITEM, _NUMofUSER)
    IS.itemCF_IUF_TD(user_items_train, alpha)
    IS.cal_itemSimilarity()
    IS.normolized_itemSimilarity()
    itemSimilarity = IS.itemSimilarity
    bigram = tools.BiGram(user_items_train)
    skipbigram = tools.SkipBiGram(user_items_train)
    for K in [5,10,20,40,80,160]:
        rank, target = Recommendation_LD(user_items_test, itemSimilarity, bigram, skipbigram, K, 10)
        recall = Recall(rank, target)
        precision = Precision(rank, target)
        F = 2.0 * recall * precision / (recall + precision)
        coverage = Coverage(rank)
        print recall, precision, F, coverage

if __name__ == '__main__':
    user_items_train, user_items_test = tools.split_data('/home/ydeng/bishe/data/user_items_each')
    ItemCF(user_items_train, user_items_test)
    ItemCF_IUF(user_items_train, user_items_test)
    ItemCF_IUF_Norm(user_items_train, user_items_test)