# Recommendation-Algorithm

Collaborative filtering algorithm based on users and iterms

Data :
The raw data which this algorithm based on comes from the indoor positioning data of Xidan Joy City from April 1, 2016 to April 7, 2016. The information included is: time, user Mac, building id, floor id, store name, date.

First, the user behavior data set is randomly divided into M shares according to a uniform distribution, one is selected as a test set, and the remaining M-1 shares are used as a training set. Then build a model on the training set, and predict the user behavior on the test set, and count the corresponding evaluation indicators. In the test set, the first 2/3 of the user's behavior trajectory is taken as the user's historical trajectory, and the last 1/3 is used for prediction.

