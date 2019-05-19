import numpy as np


class LinearRegression:
    def __init__(self, learning_rate, max_iterations):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    def fit(self, x, y):
        n_samples = x.shape[0]
        n_features = x.shape[1]

        bias = np.ones(shape=(n_samples, 1))
        x_bias = np.append(bias, x, axis=1)
        n_features += 1

        self.weights = np.zeros(n_features)

        for i in range(self.max_iterations):
            y_prediction = self.predict(x)

            loss = (1/(2*n_samples))*np.sum(np.square(y_prediction-y))
            if i % 100000 == 0:
                print("Loss - ", loss)

            error = y_prediction - y

            gradient = np.dot(x_bias.T, error)
            gradient /= n_samples
            gradient *= self.learning_rate

            self.weights -= gradient

    def predict(self, x):
        bias = np.ones(shape=(x.shape[0], 1))
        x_bias = np.append(bias, x, axis=1)
        predictions = np.dot(x_bias, self.weights)
        return predictions.flatten()


class LogisticRegression:
    def __init__(self, learning_rate, max_iterations):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    def fit(self, x, y):
        n_samples = x.shape[0]
        n_features = x.shape[1]

        bias = np.ones(shape=(n_samples, 1))
        x_bias = np.append(bias, x, axis=1)
        n_features += 1

        self.weights = np.zeros(n_features)

        for i in range(self.max_iterations):
            y_prediction = self.predict_proba(x)

            accuracy = (self.predict(x) == y).sum() / n_samples
            if i % 20 == 0:
                print("Accuracy - ", accuracy)

            error = y_prediction - y

            gradient = np.dot(x_bias.T, error)
            gradient /= n_samples
            gradient *= self.learning_rate

            self.weights -= gradient

    def sigmoid(self, predictions):
        return 1/(1 + np.exp(-predictions))

    def predict(self, x):
        bias = np.ones(shape=(x.shape[0], 1))
        x_bias = np.append(bias, x, axis=1)
        predictions = np.dot(x_bias, self.weights)
        predictions = self.sigmoid(predictions)
        predictions[predictions >= 0.5] = 1
        predictions[predictions < 0.5] = 0
        return predictions

    def predict_proba(self, x):
        bias = np.ones(shape=(x.shape[0], 1))
        x_bias = np.append(bias, x, axis=1)
        predictions = np.dot(x_bias, self.weights)
        return self.sigmoid(predictions)
