''' Just generating matrix and then doing matrix multiplication.
    It surely isn't the most efficient way (time and space complexity
    speaking) to compute the results but it's a good way to see the
    effect on the accuracy without too much code consideration.
'''
import numpy as np
import scipy as sp

class Transformer:
  def __init__(self, type):
    self.type = type

  def transform(self, dataset, target_dimension):

    if self.type == "drop":
        matrix = np.diag(np.ones(dataset[0].shape[1]))[:, 0:target_dimension]
        return (np.matmul(dataset[0], matrix), dataset[1])
    elif self.type == "g":
        matrix = np.random.randn(dataset[0].shape[1], target_dimension)
        return (np.matmul(dataset[0], matrix), dataset[1])
    elif self.type == "g_circ":
        matrix = np.transpose(sp.linalg.circulant(np.random.randn(dataset[0].shape[1]))[0:target_dimension, :])
        return (np.matmul(dataset[0], matrix), dataset[1])
    # elif self.type == "hd3":
    #     matrix =
    #     return (np.matmul(dataset[0], matrix), dataset[1])

