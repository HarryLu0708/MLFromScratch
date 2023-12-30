import numpy as np
from collections import Counter

class KNN:

    def distance(self,x1,x2):
        return np.sqrt(np.sum(x1-x2)**2)

    def __init__(self, k=3):
        self.k = k
    
    def train(self, X, y):
        self.X = X
        self.y = y
    
    def predict(self, X_test):
        # predict for each value in the test array
        y_pred = [self.single_predict(x) for x in X_test]
        return np.array(y_pred)

    def single_predict(self, x):
        # find distances for all training data
        distances = [self.distance(x,x_train) for x_train in self.X]
        # sort the distances and return an array with sorted indices, extract first k elements
        k_indices = np.argsort(distances)[:self.k]
        # get the label based on the indices in the array and lables
        k_nearest_labels = [self.y[i] for i in k_indices]
        # majority vote
        return Counter(k_nearest_labels).most_common(1)[0][0]

    
