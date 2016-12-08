from __future__ import division
import math
import numpy as np
class funcs:
    def __init__(self):
        pass
    
    def sigmoid(self,z):
        """The sigmoid function."""
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(z):
        """Derivative of the sigmoid function."""
        return sigmoid(z)*(1.0-sigmoid(z))

    def signum(self,z):
        """
        The sign function.
        -1 if x < 0, 0 if x==0, 1 if x > 0.
        nan is returned for nan inputs.
        """
        return np.sign(z)

    def tanh(self,z):
        """
        Compute hyperbolic tangent element-wise
        """
        return np.tanh(z)

    def softmax(self,z):
        """Compute softmax values across z."""
        return np.exp(z) / np.sum(np.exp(z), axis=0)
    
    def ReLU(self,x):
        """ Return maximum of zero or x"""
        res = x*(x>0)
        return res
    
    def softplus(self,x):
        """ Returns ln(1+e^x) """
        return np.log(1.0+np.exp(x))
    
    def LReLU(self,x, eps = 0.01):
        """ Returns max(x,0) + eps*max(0,-x) """
        res = x*(x>0) + eps*x*(x < 0)
        return res
    
    def arctg(self,x):
        res = 2*np.arctan(math.pi * x/2)/math.pi
        return res
    
    def cos(self,x):
        return np.cos(math.pi * x/np.linalg.norm(x))
    
    def sin(self,x):
        return np.sin(math.pi*x/np.linalg.norm(x))
    
    def x_squash(self,x):
        res = x/(1.0+np.abs(x))
        return res