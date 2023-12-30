import numpy as np

class LogisticsRegression:
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
            linear_model = np.dot(X,self.weight)+self.bias
            y_pred = self.__sigmoid(linear_model)
            dw = (1/n_samples)*np.dot(X.T,(y_pred-y))
            db = (1/n_samples)*np.sum(y_pred-y)

            self.weight -= self.lr*dw
            self.bias -= self.lr*db
    
    def predict(self,X):
        linear_model = np.dot(X,self.weight)+self.bias
        y_val = self.__sigmoid(linear_model)
        y_pred = [1 if i>0.5 else 0 for i in y_val]
        return y_pred
    
    def __sigmoid(self,x):
        return 1/(1+np.exp(-x))