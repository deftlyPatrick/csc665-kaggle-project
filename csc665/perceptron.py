import numpy as np
import csc665.perceptron as csc665_perceptron
import unittest as ut

class Object(object): pass
class TC(ut.TestCase): pass
tc = TC()

class PerceptronLayer:
"""
Use a step-wise function
"""
def __init__(self, in_count, out_count, weights=None):
"""
 weights is a Numpy array
- dimensions: out_count x in_count
- for example, for the first layer in XNOR example, weights will be:
np.array([[-30, 20, 20],
[10, 10, -20]])
- the bias is the first element in each row.
"""
self.in_count = in_count
self.out_count = out_count
self.weights = weights
pass
def forward(self, x):
"""
1) x is a numpy array of inputs, (x1, x2, etc).
2) Don't forget to add bias (the first weight in the weight matrix).
"""
i = np.dot(x, self.out_count)+self.weights
     if i >= 0: 
        Activation = 1
        else:
            Activation = 0
            return Activation

class Sequential:
      def __init__(self, layers):
      self.layers  

      def forward(self, x):
       i = np.dot(x, self.layers)
    if i >= 0:
        Activation = 1
        else:
            Activation = 0
            return Activation

class BooleanFactory:
    def create_AND(self):
"""
Creates AND perceptron, fully initialized.
Must return either Sequential, or PerceptronLayer class.
       
"""  
            j = -1.5
            k = np.array([1,0])
            x= np.array ([0,0])
            return perceptronLayer(x,k, j)
        
    def create_OR(self):
            j = -0.5
            k = np.array([1,0])
            x= np.array ([0,0])
            return perceptronLayer(x,k, j)
        
        def create_NOT(self)
            j = 0
            k = np.array([-1])
            x= np.array ([0])
            return perceptronLayer(x,k, j)
    def create_XNOR(self):
           XNOR_nn = perceptronLayer(1,1, np.array([[20, 20, 10]])
                                     return  XNOR_nn
    def create_XOR(self):
                XNOR_nn = perceptronLayer(1,0, np.array([[20, -10]])
                                     return  XOR_nn                     
