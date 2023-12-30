from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

from LinearRegression import LinearRegression

import numpy as np

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

clf = LinearRegression(lr=0.01)
clf.train(X_train,y_train)
y_pred = clf.predict(X_test)

print(y_pred)
print()
print(y_test)

def mse(y_true,y_pred):
    return np.mean((y_pred-y_true)**2)

print(mse(y_test,y_pred))