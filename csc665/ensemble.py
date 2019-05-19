import pandas as pd
import numpy as np

from .metrics import rsq
# TODO: replace Scikit-Learn's tree with the self-implemented one once it's fully functional
from sklearn.tree import DecisionTreeRegressor


class RandomForestRegressor:

    def __init__(self, n_estimators, sample_ratio, random_state=None):

        # the number of trees to build
        self.n_estimators = n_estimators

        # the ratio of the training set to use
        self.sample_ratio = sample_ratio

        # the random state to use for sampling
        self.random_state = random_state

        # list for trees
        self.trees = []

    def fit(self, x: pd.DataFrame, y: np.array):

        # set the random seed for randint
        np.random.seed(self.random_state)

        # clear out all of the current trees
        self.trees = []

        # set the amount of items that will be in the sample set
        n_samples = x.shape[0]

        # fit each individual tree
        for i in range(self.n_estimators):
            tree = DecisionTreeRegressor()
            indices = np.random.randint(0, n_samples, int(n_samples * self.sample_ratio))
            tree.fit(x.iloc[indices, :], y[indices])
            self.trees.append(tree)

    def predict(self, x: pd.DataFrame):

        # create a list for predictions
        y_prediction_list = []

        # get predictions from each tree
        for tree in self.trees:
            y_prediction_list.append(tree.predict(x))

        # return the means of all of the predicted values, index by index
        y_prediction_list = np.array(y_prediction_list)
        return np.mean(y_prediction_list, axis=0)

    def score(self, x: pd.DataFrame, y: np.array):
        return rsq(self.predict(x), y)
