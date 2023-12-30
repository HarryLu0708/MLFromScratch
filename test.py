from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

from KNN import KNN

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

clf = KNN(k=3)
clf.train(X_train,y_train)
y_pred = clf.predict(X_test)

print(y_pred)
print()
print(y_test)

