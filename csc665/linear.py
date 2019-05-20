#linear.py

import numpy as np

class LinearRegression:
    def __init__(self, learning_rate, max_iterations):
        """
        @max_iterations: the maximum number of updating iterations
        to perform before stopping
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = 0
        self.bias = 0
        self.cost_list = []

    def fit(self, X, y):
        """
        X is an array of input features, dimensions [n_samples, n_features], e.g.
        [[1, 2, 3], [2, 3, 4], [5, 6, 7]]

        y is targets, a single-dim array, [n_samples], e.g.
        [4, 5, 8]
        """
        # rows, columns
        n_samples, n_features = X.shape
        self.weights = np.zeros(shape=(n_features, 1))


        for i in range(self.max_iterations):
            # h = prediction
            h = self.predict(X)

            #cost
            cost = self.cost_function(h, y)
            # self.cost_list.append(cost)
            # print("self.cost: ", self.cost_list)


            # prediction - actual
            self.weights = self.weights - (cost * self.learning_rate)
            self.bias = self.bias - (cost * self.learning_rate)
            # self.weights = self.weights - self.learning_rate / n_samples * ((h - n_samples) * X).sum()
            # self.bias = self.bias - self.learning_rate / n_samples * (h - n_samples).sum()

    def predict(self, X):
            """
            X is an array of input features, dimensions [n_samples, n_features, e.g.
            Returns an Numpy array of real-valued predictions, one for each input, e.g.
            [3.45, 1334.5, 0.94]

            """
            prediction = np.dot(X, self.weights) + self.bias

            return prediction

    def cost_function(self, h, y):
        return 1 / (2 * len(y)) * pow(h-y, 2).sum()

class LogisticRegression:
    def __init__(self, learning_rate, max_iterations):
        """
        @ max_iterations: the maximum number of updating iterations
        to perform before stopping
        """

        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = 0
        self.bias = 0
        self.cost_list = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(shape=(n_features, 1))

        for i in range(self.max_iterations):
            h = self.predict(X)

            cost = (1 / n_samples) * np.sum(-y*(np.log(h)) - (1 - y) * (np.log(1 - h)))
            # self.cost_list.append(cost)


            # prediction - actual
            self.weights = self.weights - (cost * self.learning_rate)
            self.bias = self.bias - (cost* self.learning_rate)
                        # - self.learning_rate / n_samples * (h - n_samples).sum()


    def predict(self, X):
        """
        X is an array of multiple inputs. Each input is (x1, x2, . . . , xN).

        [[1, 1], [1, 0], [0, 1]

        Returns an Numpy array of classes predicted, e.g. [1, 0, 1, 1].
        Return 1 if probability is >= 0.5, otherwise 0.

        E.g. [1, 1, 0]
        """
        h = self.sigmoid_function(np.dot(X, self.weights) + self.bias)
        predictions = [1 if i > 0.5 else 0 for i in h]

        return predictions


    def predict_proba(self, X):
        """
        X is an array of input features, dimensions [n_samples, n_features], e.g.

        [[1, 1,], [1, 0], [0, 1]]

        Returns probabilities, e.g. [0.78, 0.68, 0.11]
        """
        self.predict = np.vectorize(self.predict)
        return self.predict(X).flatten()

    def sigmoid_function(self, X):
        return 1 / (1 + np.exp(-X))
