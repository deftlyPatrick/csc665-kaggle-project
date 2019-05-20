import pandas as pd
import numpy as np
import metrics
import features
# from . import metrics
# from . import features
from collections import Counter
from sklearn import datasets
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
        self.value = None
        self.mse = None
        self.indices = None

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
        self.value = y[indices].mean()
        self.mse = ((y[indices] - self.value) ** 2).mean()
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

        if self.max_depth is None or depth < self.max_depth:
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
            # print("self.left: ", self.left)
            # Right indices
            right_indices_bool = X_col > self.split_val
            right_indices = indices[right_indices_bool]
            assert isinstance(right_indices, np.ndarray)
            # print("self.right: ", self.right)
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
                        self.split_mse = cur_mse
                        self.split_col = col_index
                        self.split_val = np.mean([np.max(X_col[left]), np.min(X_col[right])])
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
            return self.value
        else:
            if X_row[self.split_col] <= self.split_val:
                next_branch = self.left
            else:
                next_branch = self.right
            return next_branch.internal_predict(X_row)
    #-------------------------------------------------------------------------------------
    def score(self, X: pd.DataFrame, y: np.array):
        return metrics.rsq(self.predict(X), y)
    #-------------------------------------------------------------------------------------
    def _is_leaf(self):
        assert self.left is None and self.right is None \
            or self.left is not None and self.right is not None
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
        # 0: [value, mse, samples, split_col <= split_val, if any]91
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

        self_repr = "{}:{}[{}, val: {:0.2f}, mse: {:0.2f}, samples: {:0.2f}]\n".format(
            self.depth,
            tabs,
            split_expression,
            self.value,
            self.mse,
            self.indices.shape[0]
        )

        if self.left:
            self_repr += self.left.__repr__()
            self_repr += self.right.__repr__()

        return self_repr
#-----------------------------------------------------------------------------------------
class DecisionTreeClassifier:
        def __init__(self, max_depth, min_samples_leaf):
            assert min_samples_leaf > 0

            self.max_depth = max_depth
            self.min_samples_leaf = min_samples_leaf

            # Will be assigned in fit()
            self.depth = 0
            self.left = None
            self.right = None

            # self.N = None
            self.value = None
            self.gini = None
            # self.mse = None
            self.indices = None

            self.X = None
            self.y = None

            self.split_col = None
            # Split value for the split_col
            self.split_val = None

            # Looking for smallest gini weight
            self.split_gini = None

            # # The mse for the current or best split
            # self.split_mse = None
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
            # self.value = [12, 1, 3]
            self.value = self.map_classes(y, indices)
            # print("value: ", self.value)
            self.gini = self.calc_gini((y[self.indices]))
            # self.mse = ((y[indices] - self.value) ** 2).mean()
            # self.N = indices.shape[0]

            # The following values will be set during training/splitting
            # Index of a column on which we split
            self.split_col = None
            # Split value for the split_col
            self.split_val = None

            # Looking for smallest gini weight
            self.split_gini = None

            # The mse for the current or best split
            # self.split_mse = None

            # Left and right subtrees, if not leaf
            self.left = None
            self.right = None

            # if self.max_depth is None or depth < self.max_depth:
            if self.max_depth is None or depth < self.max_depth :
                self.split(indices)
        #-------------------------------------------------------------------------------------
        def split(self, indices: np.array):
            # Iterate over every column in X and try to split
            for col_index in range(self.X.shape[1]):
                self.find_best_split(col_index, indices)

            # We may fail to find a split even if the max_depth permits, due to
            # min_samples_leaf. In this case we create no branches.
            # if self.split_gini != 0.0 or np.sum(self.value) > 2 :
            # print(np.sum(self.value))
            if self.split_gini != 0.0 and self.split_gini is not None:
                # print("Best split:")
                # print("Depth: ", self.depth, "Split_col: " ,self.split_col, "Split_val: ", self.split_val, "Split_gini: ",self.split_gini, "\n")

                # Once done with finding the split, actually split and
                # create two subtrees
                X_col = self.X.values[indices, self.split_col]

                # Left indices
                left_indices_bool = X_col <= self.split_val
                left_indices = indices[left_indices_bool]
                # print("left", left_indices)
                assert isinstance(left_indices, np.ndarray)

                self.left = DecisionTreeClassifier(self.max_depth, self.min_samples_leaf)
                # print("Left: ")
                self.left.internal_fit(self.X, self.y, left_indices, self.depth + 1)

                # Right indices
                right_indices_bool = X_col > self.split_val
                right_indices = indices[right_indices_bool]
                assert isinstance(right_indices, np.ndarray)

                self.right = DecisionTreeClassifier(self.max_depth, self.min_samples_leaf)
                # print("Right: ")
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
                    left_gini = self.calc_gini(y[left])
                    l_gini_weight = left_gini * len(y[left])/len(y)

                    right_gini = self.calc_gini(y[right])
                    r_gini_weight = right_gini * len(y[right]) / len(y)

                    cur_gini_weight = l_gini_weight + r_gini_weight
                    # cur_mse = self.calc_mse(y[left]) + self.calc_mse(y[right])
                    # print(X_col[row_index], cur_mse, self.calc_mse(y[left]) / left.sum())
                    # if self.split_mse is None or cur_mse < self.split_mse:
                    if self.split_gini is None or cur_gini_weight < self.split_gini:
                        if left.sum() >= self.min_samples_leaf and right.sum() >= self.min_samples_leaf:
                            # best split
                            # self.split_mse = cur_mse
                            self.split_gini = cur_gini_weight
                            self.split_col = col_index
                            self.split_val = np.mean([np.max(X_col[left]), np.min(X_col[right])])
                            # print("New Best: ", X_col[row_index], cur_mse, self.calc_mse(y[left]) / left.sum(),
                            #       X_col[left].max(), X_col[right].min())
                # elif:



        #-------------------------------------------------------------------------------------
        def predict(self, X: pd.DataFrame):
            result = []
            for row_index in range(X.shape[0]):
                result.append(self.internal_predict(X.values[row_index]))
            return np.array(result)
        #-------------------------------------------------------------------------------------
        def internal_predict(self, X_row: np.array):
            if self._is_leaf():
                # self.value = [12, 1, 2]
                # max_val = np.argmax(self.value) #-> 0
                # return self.value[np.argmax(self.value)] / np.sum(self.value)
                #-> 12 / 15 , 0.80

                return np.argmax(self.value)
            else:
                if X_row[self.split_col] <= self.split_val:
                    next_branch = self.left
                else:
                    next_branch = self.right
                return next_branch.internal_predict(X_row)
        #-------------------------------------------------------------------------------------
        def score(self, X: pd.DataFrame, y: np.array):
            return metrics.rsq(self.predict(X), y)
        #-------------------------------------------------------------------------------------
        def _is_leaf(self):
            assert self.left is None and self.right is None \
                or self.left is not None and self.right is not None
            return self.left is None
        #-------------------------------------------------------------------------------------
        # @staticmethod
        # noinspection PyMethodMayBeStatic
        def calc_mse(self, y):
            # value = y.mean()
            # return ((y - value) ** 2).sum()
            return np.var(y) * y.shape[0]

        # -------------------------------------------------------------------------------------
        def calc_gini(self, y):
            classes_array =[]
            numType = np.unique(y)
            type_cnt = Counter(y)
            y_count = len(y)
            for i in range(len(numType)):
                dup_num = type_cnt[numType[i]]
                classes_array.append(np.square(dup_num /y_count))
            total_array = np.sum(classes_array)
            gini = 1- total_array
            return gini

        # -------------------------------------------------------------------------------------
        def map_classes(self, y, indices):
            y2 = y[indices]
            map_class, map_val = np.unique(y, return_counts=True)
            classes, classes2_val = np.unique(y2, return_counts=True)
            c = np.in1d(map_class, classes)
            split_value = []
            inc = 0
            for i in range(len(c)):
                if c[i]:
                    split_value.append(classes2_val[inc])
                    inc += 1
                else:
                    split_value.append(0)
            return split_value
        #-------------------------------------------------------------------------------------
        def __repr__(self):
            # The number of tabs equal to the level * 4, for formatting
            # Print the tree for debugging purposes, e.g.
            # 0: [value, mse, samples, split_col <= split_val, if any]91
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

            self_repr = "{}:{}[{}, val: {:0.2f}, gini: {:0.2f}, samples: {:0.2f}]\n".format(
                self.depth,
                tabs,
                split_expression,
                self.value,
                self.gini,
                self.indices.shape[0]
            )

            if self.left:
                self_repr += self.left.__repr__()
                self_repr += self.right.__repr__()

            return self_repr





# iris = datasets.load_iris()
# csv_df = pd.DataFrame(data = iris.data  , columns = iris.feature_names)
# y = iris.target
# X = csv_df
#


#
# csv_df = pd.read_csv("iris.csv")
# X = csv_df.drop('virginica', axis=1)
# y = csv_df['virginica']


# # csv_df = pd.read_csv("bestThreePlayers.csv")
# # X = csv_df.drop('Three Pointers Made', axis = 1)
# # y = csv_df['Three Pointers Made']



# X_train, X_test, y_train, y_test = features.train_test_split(X, y, test_size=0.20, shuffle=True, random_state=None)
# Z = DecisionTreeClassifier(100, 10)
# Z.fit(X, y)

# a = Z.predict(X)
# print(a)

# b = metrics.visualize_tree(Z)
# print(b)