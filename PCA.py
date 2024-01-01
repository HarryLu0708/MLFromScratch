import numpy as np

class PCA:
    def __init__(self,n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def train(self,X,y):
        # mean
        self.mean = np.mean(X,axis=0)
        X = X-self.mean
        # covariance
        # row=1 sample, column = feature
        cov = np.cov(X.T)
        # eigenvectors, eigenvalues
        eigenvalues,eigenvectors=np.linalg.eig(cov)
        # sort eigenvectors
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        # store first n eigenvectors
        self.components = eigenvectors[0:self.n_components]

    def transform(self,X):
        # project data
        X = X-self.mean
        return np.dot(X,self.components.T)
