import numpy as np


class PerceptronLayer:

    def __init__(self, in_count, out_count, weights=None):

        """
        :param in_count: the size of first dimension of weights
        :param out_count: the size of second dimension of weights
        :param weights: a numpy array (with bias elements in the first element of each row)
        """

        self.in_count = in_count
        self.out_count = out_count
        self.weights = weights

    def forward(self, x):
        """
        :param x: a numpy array of inputs (e.g. [x1, x2, etc.])
        :return: an array of 1s and 0s based off of inputs and weights
        """

        y = np.zeros(self.out_count)

        # Situations with one output value
        if self.out_count == 1:

            # Handle the bias element before handling inputs
            y[0] += 1 * self.weights[0]

            if x.size > 1:

                for i in range(self.in_count):
                    y[0] += (x[i] * self.weights[i + 1])

            else:

                y[0] += (x * self.weights[1])

            if y[0] <= 0:
                y[0] = 0
            else:
                y[0] = 1

        # Situations with multiple output values
        else:

            for i in range(self.out_count):

                # Handle the bias elements before handling inputs
                y[i] += (1 * self.weights[i][0])

                if x.size > 1:

                    for j in range(self.in_count):
                        y[i] += (x[j] * self.weights[i][j+1])

                else:

                    y += (x * self.weights[i][1])

                if y[i] <= 0:
                    y[i] = 0
                else:
                    y[i] = 1

        return y


class Sequential:

    def __init__(self, layers):
        """
        :param layers: the perceptron layers in the sequence
        """

        self.layers = layers

    def forward(self, x):
        step = 0
        for perceptron in self.layers:
            if step == 0:
                y = perceptron.forward(x)
                step += 1
            else:
                y = perceptron.forward(y)
                step += 1
        return y


class BooleanFactory:

    def create_AND(self):
        and_nn = PerceptronLayer(2, 1, np.array([-1, 1, 1]))
        return and_nn

    def create_OR(self):
        or_nn = PerceptronLayer(2, 1, np.array([0, 1, 1]))
        return or_nn

    def create_NOT(self):
        not_nn = PerceptronLayer(1, 1, np.array([1, -1]))
        return not_nn

    def create_XNOR(self):
        # x1 AND x2, x1 NAND x2
        xnor_1 = PerceptronLayer(2, 2, np.array([[-1, 1, 1], [1, -1, -1]]))
        # x1 OR x2
        xnor_2 = PerceptronLayer(2, 1, np.array([0, 1, 1]))
        xnor_nn = Sequential([xnor_1, xnor_2])
        return xnor_nn

    def create_XOR(self):
        # x1 AND x2, x1 NAND x2
        xor_1 = PerceptronLayer(2, 2, np.array([[-1, 1, 1], [1, -1, -1]]))
        # x1 OR x2
        xor_2 = PerceptronLayer(2, 1, np.array([0, 1, 1]))
        # NOT
        xor_3 = PerceptronLayer(1, 1, np.array([1, -1]))
        xor_nn = Sequential([xor_1, xor_2, xor_3])
        return xor_nn
