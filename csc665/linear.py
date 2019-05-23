import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

a = 0
b = 0
class LinearRegression:
    def __init__(self, learning_rate=100, max_iterations=10):
        """
        @max_iterations: the maximum number of
        updating iterations to perform before stopping
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations


    def fit(self, X, y):
        """
        X is an array of input features, dimensions [n_samples, n_features], e.g.
        [[1, 2, 3],
        [2, 3, 4],
        [5, 6, 7]]
        y - targets, a single-dim array, [n_samples], e.g.
        [4, 5, 8]
        """

        global a, b
        y_pred = self.predict(X)
        n = X.shape[0]
        for i in range(self.max_iterations):
            a = a - self.learning_rate * n / (y_pred - y) * X
            b = b - self.learning_rate * n / (y_pred - y)
        loss = 0
        for i in range(n):
            loss += ((a * X + b) - y).sum()
        return loss / X.shape[0]

    def predict(self, X):

        """
        X is an array of input features, dimensions [n_samples, n_features], e.g.
        Returns an Numpy array of real-valued predictions, one for each imput, e.g.
        [3.45, 1334.5, 0.94]
        """

        global a, b
        y_pred = (a * X + b)
        return y_pred

                 
  


class LogisticRegression:
    def __init__(self, learning_rate, max_iterations):
 """
 @max_iterations: the maximum number of
 updating iterations to perform before stopping
 """
      self.learning_rate = 10
      self.max_iterations = 300

 def fit(X, y):
                 
     n_samples = x.shape[0]
     n_features = x.shape[1]
     
      bias = np.ones(shape =(n_samples, 1)
      x_bias = np.append(bias, x, axis = 1)
       n_features +=1
   self.weights = np.zero(n_features)
   for i in range (self.max_iterations)
        y_pred = self.predict_proba(x)
        accuracy = (self.predict(x) == y.sum()/n_samples
                    
                    if i % 10 ==0
               print("Accuracy: ", accuracy)
            error = y_pred - y 
            gradient = np.dot(x_bais.T, error)
            self.weights -= gradient 
 def sigmod(self, predictions):
            return 1/(1 + np.exp(-predictions))
          
 def predict(X):
 """
 X is an array of multiple inputs. Each input is (x1, x2, .., xN).
 [[1, 1],
 [1, 0],
 [0, 1]]

 Returns an Numpy array of classes predicted, e.g. [1, 0, 1, 1].
 Return 1 if probability is >= 0.5, otherwise 0.

 E.g. [1, 1, 0]

 """
 bias = np.ones(shape = (x.shape[0], 1))
                    x_bais = np.append(bias, x , axis = 1)
                    prediction = np.dot(x_bias, self.weights)
                    prediction = self.sigmod(predictions)
                    prediction[prediction >= 0.8] = 1
                    prediction[prediction < 0.8 ] = 0
                    return predicton

 def predict_proba(X):
 """
 X is an array of input features, dimensions [n_samples, n_features], e.g.
 [[1, 1],
 [1, 0],
 [0, 1]]

 Returns probabilities, e.g. [0.78, 0.68, 0.11]

 """
bias = np.ones(shape =(x.shape[0],1))
                    x_bias = np.append (bias, x, axis = 1)
                    prediction = np.dot(x_bias, self.weights)
                    return self.sigmoid(prediction)
 pass