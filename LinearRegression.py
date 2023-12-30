import numpy as np

class LinearRegression:
    def __init__(self,lr=0.01,n_itres=1000):
        self.lr = lr
        self.n_itres = n_itres
        self.bias = None
        self.weight = None
    
    def train(self,X,y):
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_itres):
            y_pred = np.dot(X,self.weight)+self.bias
            dw = (1/n_samples)*np.dot(X.T,(y_pred-y))
            db = (1/n_samples)*np.sum(y_pred-y)

            self.weight -= self.lr*dw
            self.bias -= self.lr*db
    
    def predict(self,X):
        return np.dot(X,self.weight)+self.bias