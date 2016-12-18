''' Just generating matrix and then doing matrix multiplication.
    It surely isn't the most efficient way (time and space complexity
    speaking) to compute the results but it's a good way to see the
    effect on the accuracy without too much code consideration.
'''
import numpy as np
import scipy as sp
import scipy.linalg as linalg

class Transformer:

  def __init__(self, type):
    self.type = type
    rng = np.random.RandomState(5112)
    self.rng = rng

  def transform(self, dataset, target_dimension):

    transformed_dataset = (dataset[0].copy(), dataset[1].copy())

    if self.type == "drop":
        matrix = np.diag(np.ones(transformed_dataset[0].shape[1]))[:, 0:target_dimension]
        return (np.matmul(transformed_dataset[0], matrix), dataset[1])
    elif self.type == "g":
        matrix = np.random.randn(transformed_dataset[0].shape[1], target_dimension)
        return (np.matmul(transformed_dataset[0], matrix), dataset[1])
    elif self.type == "g_circ":
        matrix = np.transpose(linalg.circulant(np.random.randn(transformed_dataset[0].shape[1]))[0:target_dimension, :])
        return (np.matmul(transformed_dataset[0], matrix), dataset[1])
    elif self.type == "hd3":
        return (self.HD3(transformed_dataset[0]), transformed_dataset[1])

    return

  def HD3(self, x):
    x = x.T
    x = self.preprocess(x)
    n = x.shape[0]
    H = linalg.hadamard(n)
    D1 = 2 * self.rng.binomial(1, 0.5, n) - 1
    D2 = 2 * self.rng.binomial(1, 0.5, n) - 1
    D3 = 2 * self.rng.binomial(1, 0.5, n) - 1
    Dx = (x.T * D1).T
    HDx = np.dot(H, Dx) / np.sqrt(n)
    DHDx = (HDx.T * D2).T
    HDHDx = np.dot(H, DHDx) / np.sqrt(n)
    DHDHDx = (HDHDx.T * D3).T
    HDHDHDx = np.dot(H, DHDHDx) / np.sqrt(n)
    return HDHDHDx.T

  def preprocess(self, x):
    """
    As Walsh-Hadamard and Hadamard matrices only work for n = 2**k,
    we need to preprocess x to have the correct size
    """
    n = x.shape[0]
    try:
        N = x.shape[1]
    except:
        N = 1

    k = 1
    while (2 ** k < n):
        k = k + 1
    if (n != 2 ** k):
        m = 2 ** k - n
        n = 2 ** k
        if N == 1:
            x_new = np.concatenate([x, np.zeros(m)])
        else:
            x_new = np.concatenate([x, np.zeros(m * N).reshape(m, N)])
    else:
        x_new = x
    return x_new
