import numpy as np

class XnorGraph:

    def __init__(self, x1, x2, y, learning_rate, initial_weights):

        # self.x1 = x1
        # self.x2 = x2
        temp_list = []
        temp_list.append(x1)
        temp_list.append(x2)
        temp_list = np.asarray(temp_list)
        self.X = temp_list
        print("self.X: ", self.X)
        self.y = y
        print("self.y: ", self.y)
        self.learning_rate = learning_rate
        print("self.learning_rate: ", self.learning_rate)
        self.layers = initial_weights
        print("self.layers: ", self.layers)
        self.n_samples = temp_list.shape
        print("self.n_samples: ", self.n_samples)

        self.loss = 0

        self.weights_layer = np.array([])
        self.bias_layer = np.array([])


        self.weights = []
        self.bias = []

        w1 = []
        b1 = []

        w2 = []
        b2 = []
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i])):
                if len(self.layers[i]) > 1:
                    w1.append(self.layers[i][j][1:])
                    b1.append(self.layers[i][j][0])
                else:
                    w2.append(self.layers[i][j][1:])
                    b2.append(self.layers[i][j][0])
        self.weights.append(w1)
        self.weights.append(w2)
        self.bias.append(b1)
        self.bias.append(b2)
        print("self.weights: ", self.weights)
        print("self.bias: ", self.bias)
        # self.weights = np.asarray(self.weights)
        # self.bias = np.asarray(self.bias)
        """
        x1, x2 - inputs , i.e. [1, 1]
        y - expected output, e.g. 1.
        learning_rate - to be used for updating weights.

        initial_weights: a LIST of Numpy arrays containing all weights;
        bias is the first element, [bias, w1, w2], e.g.

       initial_weights = [
            np.array([[-0.06712879,  1.32756237, -0.37880782],
                     [ 0.578714  ,  1.13425279,  1.29724594]]),
            np.array([[0.77283363, -1.14595272,  1.60115422]])]
        
        Create your graph and use the weights provided as starting weights
        """


    #y = pre_activation
    def forward(self):

        """
        Should return a *loss* value for a given y_true (passed in the constructor)
        Use the binary log-loss
        -y * log(h) - (1-y) * log(1-h)

        Run your forward calculate to return the loss
        """
        for i in range(len(self.layers)):
            self.weights_layer = self.weights[i]
            print("self.weights_layer: ", self.weights_layer)
            self.bias_layer = self.bias[i]
            print("self.bias_layer: ", self.bias_layer)
            self.loss = self.cost()
            print("self.loss: ", self.loss)
        return self.loss

    def backward(self):
        """
        Run the backward pass once, calculating all the gradients and *updating* weights
        at the end of the pass.

        Your network is trained by running:
            forward()
            backward()

        forward() to calculate the output values and the loss; backward() - to calculate
        and propagate gradients, and update all weights.

        """

        # current_weight = self.weights_layer
        # current_bias = self.bias_layer

        for i in reversed(range(len(self.layers))):
            self.weights_layer = self.weights[i]
            print("self.weights_layer: ", self.weights_layer)
            self.bias_layer = self.bias[i]
            print("self.bias_layer: ", self.bias_layer)
            # partial_weights = self.predict() * self.sigmoid_deriv(self.y) * (2 * (self.forward() - self.y))
            partial_weights = self.predict() * self.tanh_deriv(self.y) * (2 * (self.forward() - self.y))
            print("partial_weights: ", partial_weights)

            for k in range(len(self.bias_layer)):
                # print("self.bias_layer: ", self.bias_layer)
                # partial_bias = float(self.bias_layer[k]) * self.sigmoid_deriv(self.y) * (2 * (self.forward() - self.y))
                partial_bias = float(self.bias_layer[k]) * self.tanh_deriv(self.y) * (2 * (self.forward() - self.y))


            self.weights[i] = self.weights_layer - self.learning_rate * partial_weights
            print("self.weights[i]: ", self.weights[i])
            self.bias[i] = self.bias_layer - self.learning_rate * partial_bias
            print("self.bias[i]: ", self.bias[i])



    def predict(self):
        """
        Run your forward calculation to return the current *prediction*.
        You should use the same graph, just output the output of an earlier node.
        """
        # print("self.X: ", self.X.shape)
        # print("self.weights_layer: ", self.weights_layer)
        # print("self.bias_layer: ", self.bias_layer)
        for i in range(len(self.weights_layer)):
            for j in range(len(self.weights_layer[i])):
                # print("self.sigmoid(np.dot(self.X, self.weights_layer[i][j]) + self.bias_layer): ", self.sigmoid(np.dot(self.X, self.weights_layer[i][j]) + self.bias_layer))
                # return self.sigmoid(np.dot(self.X, self.weights_layer[i][j]) + self.bias_layer)
                print("self.tanh(np.dot(self.X, self.weights_layer[i][j]) + self.bias_layer): ", self.tanh(np.dot(self.X, self.weights_layer[i][j]) + self.bias_layer))
                return self.tanh(np.dot(self.X, self.weights_layer[i][j]) + self.bias_layer)



    def sigmoid(self, pre_activation):
        print("test: ", 1 / (1 + np.exp(-pre_activation)))
        return 1 / (1 + np.exp(-pre_activation))

    # backward
    def sigmoid_deriv(self, pre_activation):
        return self.sigmoid(pre_activation) * (1 - self.sigmoid(pre_activation))

    def tanh(self, pre_activation):
        print("test: ", (np.exp(pre_activation) - np.exp(-pre_activation))/(np.exp(pre_activation) + np.exp(-pre_activation)))
        return (np.exp(pre_activation) - np.exp(-pre_activation))/(np.exp(pre_activation) + np.exp(-pre_activation))

    # backward
    def tanh_deriv(self, pre_activation):
        return 1 - np.power(self.tanh(pre_activation), 2)

    #cross-entropy
    def cost(self):
        return (-self.y * np.log(self.predict()) - (1 - self.y) * np.log(1 - self.predict())).mean()

    #backward
    def cost_deriv(self, h):
        return (-1 * (self.y * (1/h) + (1-self.y) * (1/(1-h))))


class Variable:
    def __init__(self, value):
        self.value = value

    def forward(self):
        return self.value

    def backward(self, gradient):
        pass

class Parameter:
    def __init__(self, initial_value, learning_rate):
        self.value = initial_value
        self.learning_rate = learning_rate

    def forward(self):
        return self.value

    def backward(self, gradient):
        self.value -= self.learning_rate * gradient

class Multiply:
    def __init__(self, m1, m2):
        self.m1 = m1
        self.m2 = m2

    def forward(self):
        return self.m1.forward() * self.m2.forward()

    def backward(self, gradient):
        self.m1.backward(gradient * self.m2.forward())
        self.m2.backward(gradient * self.m1.forward())

class Add:
    def __init__(self, a1, a2):
        self.a1 = a1
        self.a2 = a2

    # Return results of applying this gate

    def forward(self):
        return self.a1.forward() + self.a2.forward()

    def backward(self, gradient):
        local_gradient = 1
        self.a1.backward(gradient * 1)
        self.a2.backward(gradient * 1)



# self.layer_One = []
        # self.layer_Two = []
        #
        # self.bias_layer_One = []
        # self.w1_layer_One = []
        # self.w2_layer_One = []
        #
        # self.bias_layer_Two = []
        # self.w1_layer_Two = []
        # self.w2_layer_Two = []
        #
        # initial_weights = np.asarray(initial_weights)
        #
        # for i in range(len(initial_weights)):
        #     self.layer_One = initial_weights[0]
        #     self.layer_Two = initial_weights[1]
        #     #     if len(layers[i]) > 1:
        #     #     layer_Two = initial_weights[1:]
        #     #
        #     # print("Layer 1: ", self.layer_One)
        #     # print("Layer 2: ", self.layer_Two)
        #     #
        #     # print("Layer 1_len: ", len(self.layer_One[0]))
        #     # print("Layer 2_len: ", len(layer_Two))
        #     self.bias_layer_One = np.asarray(self.layer_One[:, 0])
        #     self.w1_layer_One = np.asarray(self.layer_One[:, 1])
        #     self.w2_layer_One = np.asarray(self.layer_One[:, 2])
        #
        #     self.bias_layer_Two = np.asarray(self.layer_Two[:, 0])
        #     self.w1_layer_Two = np.asarray(self.layer_Two[:, 1])
        #     self.w2_layer_Two = np.asarray(self.layer_Two[:, 2])
        #
        # bias = []
        # bias2 = []
        # # w1 = []
        # # w2 = []
        # weights = []
        # weights2 = []
        #
        # for i in range(len(initial_weights)):
        #     if len(initial_weights[i]) > 1:
        #         for k in range(len(initial_weights[i])):
        #             bias.append(initial_weights[i][k][0])
        #             weights.append(initial_weights[i][k][1:])
        #     else:
        #         bias2.append(initial_weights[i][0][0])
        #         weights2.append(initial_weights[i][0][1:])
        # #     w1.append(initial_weights[i][:, 1])
        # #     w2.append(initial_weights[i][:, 2])
        #
        # # bias = np.asarray(bias)
        # # weights = np.asarray(weights)
        # # bias2 = np.asarray(bias2)
        # # weights2 = np.asarray(weights2)
        # # print(bias)
        # # w1 = np.asarray(w1)
        # # w2 = np.asarray(w2)