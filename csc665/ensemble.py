import pandas as pd
import numpy as np
from . import metrics as met
import math
from . import features as fet
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score as r2SC

class RandomForestRegressor:
    def __init__(self, n_estimators, sample_ratio, random_state = None):
        self.n_estimators = n_estimators
        self.sample_ratio = ((100 * sample_ratio) /100)
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: np.array):
        np.random.seed(self.random_state)
        self.trees = []
        n_samples = X.shape[0]
        n = self.n_estimators

        for i in range(n):
            tree = DecisionTreeRegressor()
            indices = np.random.randint(0, int(n_samples), int(math.ceil(n_samples*self.sample_ratio)))
            _ = tree.fit(X.iloc[indices, :], y[indices])
            self.trees.append(tree)


    def predict(self, X:pd.DataFrame):
        y_pred_list = []
        for t in self.trees:
            y_pred_list.append(t.predict(X))
        total_list = np.array(y_pred_list)
        b = np.mean(total_list, axis = 0)
        print("B: ", b)
        # print("y_pred_list: ", y_pred_list.shape)
        return b

    def score(self, X:pd.DataFrame, y: np.array):
        y_pred_sampled = X.mean(axis=0)
        #return r2SC(y_pred_sampled, y)
        return met.rsq(y_pred_sampled, y)

# csv_df = pd.read_csv("Melbourne_housing_FULL.csv")
# X, y = fet.preprocess_ver_1(csv_df, 'Price')

# RANDOM_STATE = 10
# X_train, X_test, y_train, y_test = fet.train_test_split(X, y, test_size=0.2,shuffle=True, random_state=RANDOM_STATE)

# print("ytest: ", y_test.shape)
# Z = RandomForestRegressor(5, 1.0)
# Z.fit(X_train, y_train)
# a = Z.predict(X_test)
# # y_pred_sampled = a.mean(axis = 0)
# # b = r2SC(y_pred_sampled, y_test)
# b = Z.score(a, y_test)
# c = Z.predict([[20,10]])
# print("Predict: ",c)
# print("Score: ", b)