import numpy as np
import math

_NUMofITEM = 269
_NUMofUSER = 335225

class UserSimilarity(object):
    def __init__(self, numOfItems, numOfUsers):
        self.UserSimilarity = dict()
        self.num_users_item = dict()
        for i in range(numOfUsers):
            self.UserSimilarity[i] = dict()
            for j in range(numOfUsers):
                self.UserSimilarity[i][j] = 0
        self.co_rated = self.UserSimilarity
        for i in range(numOfUsers):
            self.num_users_item[i] = 0

    def userCF(self, user_items_table):
        #build inverse table for item_users
        item_users = dict()
        for u, items in user_items_table.items():
            for i in items.keys():
                if i not in item_users:
                    item_users[i] = set()
                item_users[i].add(u)

        #calculate co-rated users between items
        for i, users in item_users.items():
            for u in users:
                #self.num_items_user[i] = (1 if not self.num_items_user.has_key(i) else self.num_items_user[i]+1)
                self.num_users_item[u] += 1
                for v in users:
                    if u == v:
                        continue
                    self.co_rated[u][v] += 1

    def userCF_IUF(self, user_items_table):
        #calculate co-rated users between items
        for i, users in item_users.items():
            for u in users:
                self.num_users_item[u] += 1
                for v in users:
                    if u == v:
                        continue
                    self.co_rated[u][v] += 1 / math.log(1 + len(users) * 1.0)

    def cal_userSimilarity(self):
        #calculate final similarity matrix
        for u, related_users in self.co_rated.items():
            for v, cuv in related_users.items():
                self.itemSimilarity[u][v] = cuv / math.sqrt(self.num_users_item[u] * self.num_users_item[v])

def UserCF(user_items_train, user_items_test):
    US = UserSimilarity(_NUMofITEM, _NUMofUSER)
    US.userCF(user_items_table)
    US.cal_userSimilarity()
    userSimilarity = US.userSimilarity
    for K in [5,10,20,40,80,160]:
        rank, target = Recommendation(user_items_test, itemSimilarity, K, 10)
        recall = Recall(rank, target)
        precision = Precision(rank, target)
        F = 2.0 * recall * precision / (recall + precision)
        coverage = Coverage(rank)
        print recall, precision, F, coverage

def UserCF_IUF(user_items_train, user_items_test):
    US = UserSimilarity(_NUMofITEM, _NUMofUSER)
    US.userCF_IUF(user_items_table)
    US.cal_userSimilarity()
    userSimilarity = US.userSimilarity
    for K in [5,10,20,40,80,160]:
        rank, target = Recommendation(user_items_test, itemSimilarity, K, 10)
        recall = Recall(rank, target)
        precision = Precision(rank, target)
        F = 2.0 * recall * precision / (recall + precision)
        coverage = Coverage(rank)
        print recall, precision, F, coverage

def UserCF_IUF_Norm(user_items_train, user_items_test):
    US = UserSimilarity(_NUMofITEM, _NUMofUSER)
    US.userCF_IUF(user_items_table)
    US.cal_userSimilarity()
    US.normolized_userSimilarity()
    userSimilarity = US.userSimilarity
    for K in [5,10,20,40,80,160]:
        rank, target = Recommendation(user_items_test, itemSimilarity, K, 10)
        recall = Recall(rank, target)
        precision = Precision(rank, target)
        F = 2.0 * recall * precision / (recall + precision)
        coverage = Coverage(rank)
        print recall, precision, F, coverage

if __name__ == '__main__':
    user_items_train, user_items_test = tools.split_data('/home/ydeng/bishe/data/user_items_each')
    UserCF(user_items_train, user_items_test)
    #UserCF_IUF(user_items_train, user_items_test)
    #UserCF_IUF_Norm(user_items_train, user_items_test)

