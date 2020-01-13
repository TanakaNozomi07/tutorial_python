from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
a = pd.read_csv("tutorial_python/text/data/X_train.csv", sep=",")
b = pd.read_csv("tutorial_python/text/data/y_train.csv", sep=",")

class MyRegression():
    def __init__(self,lam = 0,a=None,b=None):
        
        self.a = a
        self.b = b
        self.lam = lam
        
    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        
        if self.lam != 0:
            pass
        else:
            pass
        
        one = np.ones(X.shape[0]).reshape(-1,1)
        X_ = np.concatenate((one,X),axis=1)
        Lam = self.lam * np.eye(X_.shape[1])
        A = Lam + np.dot(X_.T,X_)
        X_daggar = np.dot(np.linalg.inv(A),X_.T)
        w = np.dot(X_daggar,y)
        
        self.a_ =  w[1:]
        self.b_ =  w[0] #a(スカラー)X(行列) = y(ベクトル) の意。カラム数２以上のデータが扱えない。データを減らすかこの書き方を変えるべし。
        return self
        
    def score(self, X, y):
        return 1
    
    def get_params(self, deep=True):
        return {'a': self.a, 'b': self.b}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self,parameter, value)
        return self
        
    def predict(self, X):
        X = check_array(X)
        y = np.dot(X, self.a_)+self.b_
        
        check_is_fitted(self, "a_", "b_") # 学習済みかチェックする(推奨)
        X = check_array(X)
        return y
        
if __name__=="__main__":
    clf = MyRegression()

a["最多風向"][a["最多風向"] == "北北東"] = 0.125
a["最多風向"][a["最多風向"] == "北東"] = 0.25
a["最多風向"][a["最多風向"] == "東北東"] = 0.375
a["最多風向"][a["最多風向"] == "東"] = 0.5
a["最多風向"][a["最多風向"] == "東南東"] = 0.625
a["最多風向"][a["最多風向"] == "南東"] = 0.75
a["最多風向"][a["最多風向"] == "南南東"] = 0.875
a["最多風向"][a["最多風向"] == "北"] = 0
a["最多風向"][a["最多風向"] == "北北西"] = 0.125
a["最多風向"][a["最多風向"] == "北西"] = 0.25
a["最多風向"][a["最多風向"] == "西北西"] = 0.375
a["最多風向"][a["最多風向"] == "西"] = 0.5
a["最多風向"][a["最多風向"] == "西南西"] = 0.625
a["最多風向"][a["最多風向"] == "南西"] = 0.75
a["最多風向"][a["最多風向"] == "南南西"] = 0.875
a["最多風向"][a["最多風向"] == "南"] = 1

X = a[["最高気温"]].values
T = b.values

X_train = X[:13]
T_train = T[:13]
X_test = X[13:]
T_test = T[13:]

from sklearn.model_selection import GridSearchCV

np.random.seed(0)

# Grid search
parameters = {'lam':np.exp([i for i in range(-30,1)])}
reg = GridSearchCV(MyRegression(),parameters,cv=5)
reg.fit(X_train,T_train)
best = reg.best_estimator_

# 決定係数
print("決定係数: ", best.score(X_train, T_train)) # BaseEstimatorを継承しているため使える
# lambda
print("lam: ", best.lam)

fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

axes[0].scatter(X_train, T_train, marker=".")
axes[0].plot(X_train, best.predict(X_train), color="red")
axes[0].set_title("train")

axes[1].scatter(X_test, T_test, marker=".")
axes[1].plot(X_test, best.predict(X_test), color="red")
axes[1].set_title("test")
fig.show()
