import pandas as pd
import numpy as np
from . import metrics

# TODO: replace with a modified version of the original regressor once it's working acceptably

class DecisionTreeClassifier:
    def __init__(self, max_depth, min_samples_leaf):
        assert min_samples_leaf > 0

        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

        # Will be assigned in fit()
        self.depth = 0
        self.left = None
        self.right = None

        # Used for determining value
        self.class_count = None

        # self.N = None
        self.value = None
        self.gini = None
        self.indices = None

        self.X = None
        self.y = None

        self.split_col = None
        # Split value for the split_col
        self.split_val = None
        # The gini for the current or best split
        self.split_gini = None

    def fit(self, X: pd.DataFrame, y: np.array):
        # self.indices = range(X.shape[0])

        # Get the number of times that each class appears in the root
        self.class_count = dict()
        self.class_count = self.get_class_count(y)

        self.internal_fit(
            X,
            y,
            # Indices of rows to be used in the tree
            # All rows will be used at the top level; then this array will depend
            # on the split
            np.array(range(X.shape[0])),
            # Count of classes
            self.class_count,
            # Initial depth.
            0)

    def internal_fit(self, X, y, indices: np.array, class_count: dict, depth):
        self.X = X
        self.y = y
        self.indices = indices
        self.depth = depth

        # Use a reset copy of class count to preserve all classes in the tree
        self.class_count = class_count
        self.class_count = self.reset_class_count()
        self.class_count = self.get_class_count(y[self.indices])

        # Calculate value
        # self.value = y[indices].mean()
        self.value = self.get_value()
        self.gini = self.calc_gini(y[self.indices])
        # self.N = indices.shape[0]

        # The following values will be set during training/splitting
        # Index of a column on which we split
        self.split_col = None
        # Split value for the split_col
        self.split_val = None
        # The gini for the current or best split
        self.split_gini = None

        # Left and right subtrees, if not leaf
        self.left = None
        self.right = None

        if self.max_depth is None or depth < self.max_depth:
            self.split(indices)

    def split(self, indices: np.array):
        # Iterate over every column in X and try to split
        for col_index in range(self.X.shape[1]):
            self.find_best_split(col_index, indices)

        # We may fail to find a split even if the max_depth permits, due to
        # min_samples_leaf. In this case we create no branches.
        if self.split_gini is not None:
            # print("Best split: ", self.depth, self.split_col, self.split_val, self.split_gini)

            # Once done with finding the split, actually split and
            # create two subtrees
            X_col = self.X.values[indices, self.split_col]

            # Left indices
            left_indices_bool = X_col <= self.split_val
            left_indices = indices[left_indices_bool]
            assert isinstance(left_indices, np.ndarray)

            self.left = DecisionTreeClassifier(self.max_depth, self.min_samples_leaf)
            self.left.internal_fit(self.X, self.y, left_indices, self.class_count, self.depth + 1)

            # Right indices
            right_indices_bool = X_col > self.split_val
            right_indices = indices[right_indices_bool]
            assert isinstance(right_indices, np.ndarray)

            self.right = DecisionTreeClassifier(self.max_depth, self.min_samples_leaf)
            self.right.internal_fit(self.X, self.y, right_indices, self.class_count, self.depth + 1)

    def find_best_split(self, col_index, indices):
        X_col = self.X.values[indices, col_index]
        y = self.y[indices]

        for row_index in range(indices.shape[0]):
            left = X_col <= X_col[row_index]
            right = X_col > X_col[row_index]

            assert isinstance(left, np.ndarray)
            assert isinstance(right, np.ndarray)

            # Calculate gini values and decide if this the best split
            # so far. If yes, set the object values: self.split_col,
            # self.split_val, etc.

            # If one of the branches has NO samples, then this is an invalid split. Skip.
            if left.any() and right.any():
                left_gini = self.calc_gini(y[left]) * (left.size / indices.size)
                right_gini = self.calc_gini(y[right]) * (right.size / indices.size)
                cur_gini = left_gini + right_gini
                # print(X_col[row_index], cur_mse, self.calc_mse(y[left]) / left.sum())
                if self.split_gini is None or cur_gini < self.split_gini:
                    if left.sum() >= self.min_samples_leaf and right.sum() >= self.min_samples_leaf:
                        # best split
                        self.split_gini = cur_gini
                        self.split_col = col_index
                        self.split_val = np.mean([np.max(X_col[left]), np.min(X_col[right])])

    def predict(self, X: pd.DataFrame):
        result = []
        for row_index in range(X.shape[0]):
            result.append(self.internal_predict(X.values[row_index]))
        return np.array(result)

    def internal_predict(self, X_row: np.array):
        if self._is_leaf():
            return np.argmax(self.value)
        else:
            if X_row[self.split_col] <= self.split_val:
                next_branch = self.left
            else:
                next_branch = self.right
            return next_branch.internal_predict(X_row)

    def score(self, X: pd.DataFrame, y: np.array):
        return metrics.rsq(self.predict(X), y)

    def _is_leaf(self):
        assert self.left is None and self.right is None \
            or self.left is not None and self.right is not None
        return self.left is None

    def get_class_count(self, class_list: np.array):
        # use a reset copy of class count to preserve all possible classes (including ones not in this node)
        class_count = self.reset_class_count()

        for sample in class_list:
            # count the number of times of each class appears
            class_count[sample] = class_count.get(sample, 0) + 1

        return class_count

    def reset_class_count(self):
        reset_count = self.class_count

        # set the count of each class to 0
        for class_type in reset_count:
            reset_count[class_type] = 0

        return reset_count

    def get_value(self):
        classes = []
        counts = []

        # get lists of the classes and counts
        for class_type in self.class_count:
            classes.append(class_type)
            counts.append(self.class_count[class_type])

        # sort the counts according to the classes
        value = [count for _, count in sorted(zip(classes, counts))]

        return value

    def calc_gini(self, class_list: np.array):
        gini = 1;
        class_count = self.get_class_count(class_list)

        for class_type in class_list:
            # divide the number of appearances by the total number of samples
            class_count[class_type] = class_count[class_type] / class_list.size

            # square the previous value
            class_count[class_type] = (class_count[class_type])**2

            # subtract the result
            gini = gini - class_count[class_type]

        return gini


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
