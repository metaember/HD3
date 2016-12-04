import numpy as np

class Functions:

    def __init__(self, type):
        self.type = type

    def apply(self, dataset):

        if self.type == "identity":
            return dataset
        elif self.type == "sigmoid":
            return (np.apply_along_axis(sigmoid, 0, dataset[0]), dataset[1])
        elif self.type == "sigmoid_prime":
            return (np.apply_along_axis(sigmoid_prime, 0, dataset[0]), dataset[1])
        elif self.type == "signum":
            return (np.apply_along_axis(signum, 0, dataset[0]), dataset[1])
        elif self.type == "tanh":
            return (np.apply_along_axis(tanh, 0, dataset[0]), dataset[1])
        elif self.type == "softmax":
            return (np.apply_along_axis(softmax, 0, dataset[0]), dataset[1])



def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def signum(z):
    """
    The sign function.
    -1 if x < 0, 0 if x==0, 1 if x > 0.
    nan is returned for nan inputs.
    """
    return np.sign(z)

def tanh(z):
    """
    Compute hyperbolic tangent element-wise
    """
    return np.tanh(z)

def softmax(z):
    """Compute softmax values across z."""
    return np.exp(z) / np.sum(np.exp(z), axis=0)