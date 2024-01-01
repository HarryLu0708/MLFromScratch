import numpy as np
from collections import Counter

class Node:
    def __init__(self,feature=None,threhold=None,left=None,right=None,*,value=None):
        self.feature = feature
        self.threhold = threhold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __entropy(self,y):
        hist = np.bincount(y)
        ps = hist/len(y)
        return -np.sum([p*np.log2(p) for p in ps if p>0])
    
    def __grow_tree(self,X,y,depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        # stopping criteria
        if (depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value = self.__most_common_label(y)
            return Node(value=leaf_value)
        feat_idx = np.random.choice(n_features,self.n_features,replace=False)
        best_feat,best_thresh = self.__best_criteria(X,y,feat_idx)
        left_idxs,right_idxs = self.__split(X[:,best_feat],best_thresh)
        left = self.__grow_tree(X[left_idxs,:],y[left_idxs],depth+1)
        right = self.__grow_tree(X[right_idxs,:],y[right_idxs],depth+1)
        return Node(best_feat,best_thresh,left,right)


    def __best_criteria(self,X,y,feat_idxs):
        best_gain = -1
        split_idx,split_threh = None,None
        for feat_idx in feat_idxs:
            X_column = X[:,feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self.__information_gain(y,X_column,threshold)
                if gain>best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threh = threshold
        return split_idx,split_threh
    
    def __information_gain(self,y,X_column,split_threh):
        # parent Entropy
        parent_entropy = self.__entropy(y)
        # generate split
        left_idxs, right_idxs = self.__split(X_column,split_threh)
        if len(left_idxs)==0 or len(right_idxs)==0:
            return 0
        # weight avg child Entropy
        n = len(y)
        n_l,n_r = len(left_idxs),len(right_idxs)
        e_l,e_r = self.__entropy(y[left_idxs]), self.__entropy(y[right_idxs])
        child_entropy = (n_l/n)*e_l+(n_r/n)*e_r
        # return ig
        ig = parent_entropy-child_entropy
        return ig
    
    def __split(self,X_column,split_threh):
        left_idxs = np.argwhere(X_column<=split_threh).flatten()
        right_idxs = np.argwhere(X_column>split_threh).flatten()
        return left_idxs,right_idxs
    
    def __most_common_label(self,y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def __init__(self,min_samples_split=2,max_depth=100,n_features=None):
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.n_features=n_features
        self.root = None

    def train(self,X,y):
        # grow tree
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features,X.shape[1])
        self.root = self.__grow_tree(X,y)
    
    def predict(self, X):
        return np.array([self.__traverse_tree(x, self.root) for x in X])

    
    def __traverse_tree(self,x,node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature]<=node.threhold:
            return self.__traverse_tree(x,node.left)
        return self.__traverse_tree(x,node.right)

