import numpy as np

class PerceptronLayer:
	"""
	Use a step-wise function
	"""

	"""
	Activation function: y > threshold, A = 1, otherwise 0
	output: activation(weighted sum of inputs)
	"""
	# self.in_count = in_count
	# self.out_count = out_count
	# self.weights = weights


	def __init__(self, in_count, out_count, weights = None):
			self.in_count = in_count
			self.out_count = out_count
			self.weights = weights[1:]
			self.bias = weights[:1]


	def forward(self, x):
		"""
		1) x is numpy array of inputs, (x1, x2, etc).
		2) Don't forget to add bias (the first weight in the weight matrix).
		"""
		pre_activation = (np.dot(x, self.weights)) + self.bias
		pre_activation = self.step_function(pre_activation)
		return pre_activation

	def step_function(self, pre_activation):
		return 1 if pre_activation > 0 else 0


class Sequential:

	def __init__(self, layers):
		self.layers = layers

	def forward(self, x):
		pre_activation_array = np.array([])
		pre_activation = 0
		for i in range(len(self.layers)):
			weight = self.layers[i].weights
			bias = self.layers[i].bias
			if i == len(self.layers)-1:
				pre_activation = (np.dot(pre_activation_array, weight) + bias)
			else:
				pre_activation = (np.dot(x, weight)) + bias
				if pre_activation > 0:
					pre_activation = 1
				else:
					pre_activation = 0
				pre_activation_array = np.append(pre_activation_array, pre_activation)
		pre_activation = self.step_function(pre_activation)
		return pre_activation

	def step_function(self, pre_activation):
		return 1 if pre_activation > 0 else 0


class BooleanFactory:

	def create_AND(self):
		"""
			Creates AND perceptron, fully initialized
			Must return either Sequential or PerceptronLayer class.
		"""
		# weights = np.array([-1, 1, 1])
		weights = np.array([-30, 20, 20])
		return PerceptronLayer(1, 1, weights)

	def create_OR(self):
		# weights = np.array([0, 1, 1])
		weights = np.array([-10, 20, 20])
		return PerceptronLayer(1, 1, weights)

	def create_NOT(self):
		# weights = np.array([0.5, -1])
		weights = np.array([1, -1])
		return PerceptronLayer(1, 1, weights)


	def create_XNOR(self):
		#non-linear

		#and
		w1 = np.array([-30, 20, 20])

		#not x1 and not x2
		w2 = np.array([10, -20, -20])

		#or
		w3 = np.array([-10, 20, 20])

		return Sequential([PerceptronLayer(1, 1, w1),
						   PerceptronLayer(1, 1, w2),
						   PerceptronLayer(1, 1, w3)])

	def create_XOR(self):
		#non-linear

		#or
		w1 = np.array([-10, 20, 20])

		#nand
		w2 = np.array([1.5, -1, -1])

		#and
		w3 = np.array([-30, 20, 20])

		return Sequential([PerceptronLayer(1, 1, w1),
						   PerceptronLayer(1, 1, w2),
						   PerceptronLayer(1, 1, w3)])


# #checking how many dimensions does the array have
	# if weights.ndim == 1:
	# 	self.weights = weights[1:]
	# 	self.bias = weights[:1]
	# else:
	# 	self.weights = [i[1:] for i in weights]
	# 	self.bias = [i[:1] for i in weights]

	# print("self.weights: ", self.weights)
	# self.bias = [i[0] for i in weights]

	# weight is Numpy array
	# -- dimensions: out_count * in_count
	# -- for example, for the first layer in XNOR example, weights will be:
	#     np.array([[-30, 20, 20], [10, 10, -20]])
	# -- bias is the first element in each row

#XNOR
	# def forward(self, x):
	# 	a = np.array([])
	# 	for i in range(len(self.layers)):
	# 		# print("len(self.layers): ", self.layers[1].weight)
	# 		weight = self.layers[i].weights
	# 		bias = self.layers[i].bias
	# 		# print("bias: ", bias)
	# 		# print("weight: ", weight)
	# 		# print("x: ", x)
	# 		a_equ = (np.dot(x, weight)) + bias
	# 		# print("a_equ1: ", a_equ)
	# 		if a_equ > 0:
	# 			a_equ = 1
	# 		else:
	# 			a_equ = 0
	# 		# print("a_equ2: ", a_equ)
	# 		a = np.append(a, a_equ)
	# 		# print("a: ", a)
	# 		# print("\n")
	#
	# 	or_weight = np.array([20, 20])
	# 	or_bias = -10
	# 	pre_activation = (np.dot(a, or_weight) + or_bias)
	#
	#
	# 	# print("pre_activation: ", pre_activation)
	# 	return 1 if pre_activation > 0 else 0
	#

	# #XOR
	# def forward(self, x):
	# 	a = np.array([])
	# 	for i in range(len(self.layers)):
	# 		# print("len(self.layers): ", self.layers[1].weight)
	# 		weight = self.layers[i].weights
	# 		bias = self.layers[i].bias
	# 		# print("bias: ", bias)
	# 		# print("weight: ", weight)
	# 		# print("x: ", x)
	# 		a_equ = (np.dot(x, weight)) + bias
	# 		# print("a_equ1: ", a_equ)
	# 		if a_equ > 0:
	# 			a_equ = 1
	# 		else:
	# 			a_equ = 0
	# 		# print("a_equ2: ", a_equ)
	# 		a = np.append(a, a_equ)
	# 		# print("a: ", a)
	# 		# print("\n")
	#
	# 	and_weight = np.array([20, 20])
	# 	and_bias = -30
	# 	pre_activation = (np.dot(a, and_weight) + and_bias)
	#
	#
	# 	# print("pre_activation: ", pre_activation)
	# 	return 1 if pre_activation > 0 else 0