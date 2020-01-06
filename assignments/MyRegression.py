import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
a = pd.read_csv("../text/data/X_train.csv", sep=";")
b = pd.read_csv("../text/data/y_train.csv", sep=";")

class MyRegression():
    def __init__(self, lam = 0):
        
        self.a = a
        self.b = b
        self.lam = lam
        
    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        if self.lam != 0:
            pass
        else:
            pass
        
        self.a_ =  np.linalg.solve(X, y)
        self.b_ =  np.linalg.solve(X,y)
        return self
        
    def predict(self, X):
        check_is_fitted(self, "a_", "b_") # 学習済みかチェックする(推奨)
        X = check_array(X)
        return y
        
if __name__=="__main__":
    clf = MyRegression()

