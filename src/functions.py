import numpy as np

class Function:

    def __init__(self, type):
        self.type = type
        self.func = self.get_func(type)

    def get_func(self, func_name):
        """
        Returns desired function by name
        """

        if self.type == "identity":
            return identity
        elif self.type == "sigmoid":
            return sigmoid
        elif self.type == "sigmoid_prime":
            return sigmoid_prime
        elif self.type == "signum":
            return signum
        elif self.type == "tanh":
            return tanh
        elif self.type == "softmax":
            return softmax
        else:
            raise NameError("Function {} does not exist".format(func_name))


    def apply(self, dataset):
        """
        Apply a function over a dataset.
        """

        function_appplied_dataset = (dataset[0].copy(), dataset[1].copy())

        if self.type == "identity":
            return function_appplied_dataset
        else:
            return (np.apply_along_axis(self.func, 0, function_appplied_dataset[0]), function_appplied_dataset[1])


def identity(z):
    """ Dunno why we want this but whatever """
    return z

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
