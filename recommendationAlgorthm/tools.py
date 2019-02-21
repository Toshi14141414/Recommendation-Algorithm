import numpy as np

_NUMofITEM = 271

def BiGram(user_items):
    bigram = np.ones((_NUMofITEM, _NUMofITEM))
    for items in user_items.values():
        for i in range(len(items)-1):
            bigram[items[i]][items[i+1]] += 1
    for i in range(len(bigram)):
        s = sum(bigram[i])
        for j in range(bigram[i]):
            if i == j:
                bigram[i][j] = 0
            else:
                bigram[i][j] = float(bigram[i][j])/s
    return bigram

def SkipBiGram(user_items):
    skipbigram = np.ones((_NUMofITEM, _NUMofITEM))
    for items in user_items.values():
        for i in range(len(items)-2):
            skipbigram[items[i]][items[i+2]] += 1
    for i in range(len(skipbigram)):
        s = sum(skipbigram[i])
        for j in range(skipbigram[i]):
            if i == j:
                skipbigram[i][j] = 0
            else:
                skipbigram[i][j] = float(skipbigram[i][j])/s
    return skipbigram

def split_data(file_dir):
    user_items_train = dict()
    user_items_test = dict()
    with open('/home/ydeng/bishe/data/user_items_each', 'r') as fin:
        for i in range(80000):
            line = fin.readline()
            items = line.strip().split('\t')
            exec('user_items_train[items[0]] = map(int,' + items[1] + ')')
        for line in fin:
            items = line.strip().split('\t')
            exec('user_items_test[items[0]] = map(int,' + items[1] + ')')
    return user_items_train, user_items_test