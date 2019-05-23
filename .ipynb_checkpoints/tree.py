import pandas as pd
import numpy as np
from . import metrics
#-----------------------------------------------------------------------------------------
class DecisionTreeRegressor:
    def __init__(self, max_depth, min_samples_leaf):
        assert min_samples_leaf > 0

        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

        # Will be assigned in fit()
        self.depth = 0
        self.left = None
        self.right = None

        # self.N = None
        self.mse =None
        self.val=None
        self.X =None
        self.y=None
        self.X = None
        self.y = None

        self.split_col = None
        # Split value for the split_col
        self.split_val = None
        # The mse for the current or best split
        self.split_mse = None
    #-------------------------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: np.array):
        # self.indices = range(X.shape[0])

        self.internal_fit(
            X,
            y,
            # Indices of rows to be used in the tree
            # All rows will be used at the top level; then this array will depend
            # on the split
            np.array(range(X.shape[0])),
            # Initial depth.
            0)
    #-------------------------------------------------------------------------------------
    def internal_fit(self, X, y, indices: np.array, depth):
        self.X = X
        self.y = y
        self.indices = indices
        self.depth = depth

        # Calculate value
        # self.value = y[indices].mean()
        self.value = y[indices].mean()
        self.mse = ((y[indices]-self.value)**2).mean()
        self.N = indices.shape[0]
        # self.N = indices.shape[0]

        # The following values will be set during training/splitting
        # Index of a column on which we split
        self.split_col = None
        # Split value for the split_col
        self.split_val = None
        # The mse for the current or best split
        self.split_mse = None

        # Left and right subtrees, if not leaf
        self.left = None
        self.right = None

        if depth < self.max_depth:
            self.split(indices)
    #-------------------------------------------------------------------------------------
    def split(self, indices: np.array):
        # Iterate over every column in X and try to split
        for col_index in range(self.X.shape[1]):
            self.find_best_split(col_index, indices)

        # We may fail to find a split even if the max_depth permits, due to
        # min_samples_leaf. In this case we create no branches.
        if self.split_mse is not None:
            # print("Best split: ", self.depth, self.split_col, self.split_val, self.split_mse)

            # Once done with finding the split, actually split and
            # create two subtrees
            X_col = self.X.values[indices, self.split_col]

            # Left indices
            left_indices_bool = X_col <= self.split_val
            left_indices = indices[left_indices_bool]
            # print("left", left_indices)
            assert isinstance(left_indices, np.ndarray)

            self.left = DecisionTreeRegressor(self.max_depth, self.min_samples_leaf)
            self.left.internal_fit(self.X, self.y, left_indices, self.depth + 1)

            # Right indices
            right_indices_bool = X_col > self.split_val
            right_indices = indices[right_indices_bool]
            assert isinstance(right_indices, np.ndarray)

            self.right = DecisionTreeRegressor(self.max_depth, self.min_samples_leaf)
            self.right.internal_fit(self.X, self.y, right_indices, self.depth + 1)
    #-------------------------------------------------------------------------------------
    def find_best_split(self, col_index, indices):
        X_col = self.X.values[indices, col_index]
        y = self.y[indices]

        for row_index in range(indices.shape[0]):
            left = X_col <= X_col[row_index]
            right = X_col > X_col[row_index]

            assert isinstance(left, np.ndarray)
            assert isinstance(right, np.ndarray)

            # Calculate MSE values and decide if this the best split
            # so far. If yes, set the object values: self.split_col,
            # self.split_val, etc.

            # If one of the branches has NO samples, then this is an invalid split. Skip.
            if left.any() and right.any():
                cur_mse = self.calc_mse(y[left]) + self.calc_mse(y[right])
                # print(X_col[row_index], cur_mse, self.calc_mse(y[left]) / left.sum())
                if self.split_mse is None or cur_mse < self.split_mse:
                    if left.sum() >= self.min_samples_leaf and right.sum() >= self.min_samples_leaf:
                        # best split
                       # Split value for the split_col
                        self.split_col = col_index
                        self.split_val = np.mean([np.max(X_col[left]), np.min(X_col[right])])
                        # The mse for the current or best split
                        self.split_mse = cur_mse
                        # print("New Best: ", X_col[row_index], cur_mse, self.calc_mse(y[left]) / left.sum(),
                        #       X_col[left].max(), X_col[right].min())
    #-------------------------------------------------------------------------------------
    def predict(self, X: pd.DataFrame):
        result = []
        for row_index in range(X.shape[0]):
            result.append(self.internal_predict(X.values[row_index]))
        return np.array(result)

    #-------------------------------------------------------------------------------------
     def internal_predict(self, X_row: np.array):
        if self._is_leaf():
            self.value = [12, 1, 3]
            np.argmax(self.value) #-> 0
            return self.value[0] / np.sum(self.value) #-> 12/15
        else:
            if X_row[self.split_col] <= self.split_val:
                next_branch = self.left
            else:
                next_branch = self.right
            return next_branch.internal_predict(X_row)
    #-------------------------------------------------------------------------------------
        def score(self, X: pd.DataFrame, y: np.ndarray):
        return metrics.rsq(self.predict(X), y)

    #-------------------------------------------------------------------------------------
    
    def _is_leaf(self):
        assert self.left is None and self.right is None \
            or self.right is None and self.right is not None
        return self.left is None

    #-------------------------------------------------------------------------------------
    # @staticmethod
    # noinspection PyMethodMayBeStatic
    def calc_mse(self, y):
        # value = y.mean()
        # return ((y - value) ** 2).sum()
        return np.var(y) * y.shape[0]
    #-------------------------------------------------------------------------------------
    def __repr__(self):
        # The number of tabs equal to the level * 4, for formatting
        # Print the tree for debugging purposes, e.g.
        # 0: [value, mse, samples, split_col <= split_val, if any]
        #     1: [value, mse, samples, split_col <= split_val, if any]
        #     1: [value, mse, samples, split_col > split_val, if any]
        #           2: 1: [value, mse, samples, split_col <= split_val, if any]
        # etc..
        # The number of tabs equal to the level * 4, for formatting

        tabs = "".join([" " for _ in range(self.depth * 4)])

        if not self._is_leaf():
            attr_name = self.X.columns[self.split_col]
            split_expression = attr_name + " <= " + "{}".format(self.split_val)
    else:
            split_expression = ''

        self_repr = "{}:{}[{}, val: {:0.2f}, mse: {:0.2f}, samples: {:2.0f}]\n"\
            .format(self.depth, tabs, split_expression, self.value, self.mse,self.indices.shape[0])
        if self.left:
            self_repr = self.left.__repr__()
            self_repr += self.right.__repr__()
        raise self_repr
