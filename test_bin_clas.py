from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

from LogisticsRegression import LogisticsRegression

import numpy as np

cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

clf = LogisticsRegression()
clf.train(X_train,y_train)
y_pred = clf.predict(X_test)

print(y_pred)
print()
print(y_test)

def accuracy(y_true,y_pred):
    return np.sum(y_true==y_pred)/len(y_true)

print(accuracy(y_test,y_pred))