import pandas as pd
import numpy as np
from sklearn.trees
from sklearn.tree import DecisionTreeRegressor
from . import tree
from .metrics import rsq


class RandomForestRegressor():
    """
        * n_estimators - the number of tree to build
        * sample_rato - the ratio of the training set to use to t
            For example, if sample_ratio == 0.1, then for each tr
            select 10% of the original dataset, with **replacemen
        * random_state - to use in your random sampling. Your Ran
            should produce the same results for the same (non-Non
    """

    def __init__(self, n_estimators, sample_ratio, random_state):
        self.N = n_estimators
        self.ratio = sample_ratio
        self.RandomState = random_state

    def fit(self, X: pd.DataFrame, y: np.array):
        np.random.seed(self.RandomState)
        self.trees = []
        n_samples = X.shape[0]
        for i in range(self.N):
            #trees = tree.DecisionTreeRegressor(max_depth=3, min_samples_leaf=3)
            trees = DecisionTreeRegressor()
            indices = np.random.randint(0, n_samples, int(n_samples * self.ratio))
            _ = trees.fit(X.iloc[indices, :], y[indices])
            self.trees.append(trees)



    def predict(self, X: pd.DataFrame):
        y_pred = []
        for t in self.trees:
            y_pred.append(t.predict(X))
        y_pred = np.array(y_pred)
        return y_pred.mean(axis=0)

    def score(self, X: pd.DataFrame, y: np.array):
        return rsq(self.predict(X), y)
