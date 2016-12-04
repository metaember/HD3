#
# Small file to test code. Just run it using python 3 : "python testing.py"
#
from scipy.linalg import hadamard
from scipy.linalg import circulant
from scipy.signal import fftconvolve

import time

import numpy as np
from scipy import linalg


from loaders import *
import transformations as trans
# import classifiers as cl
import tensorflow as tf


G_circ = linalg.circulant(np.random.randn(10))
# G = np.random.randn(2, 10)

# a = load_data("mnist")
# compressed = trans.transform(a[0][0], "i", 500)
# compressed_test = trans.transform(a[1][0], "i", 500)
#
# start = time.clock()
# res = cl.softmax_nn(a)
# end = time.clock()
# print("It took {} seconds".format(round(end - start, 2)))
# print("Accuracy is {} with {} dimensions".format(res, a[0][0].shape[1]))
#
# start_comp = time.clock()
# res_comp = cl.softmax_nn_comp(a, compressed, compressed_test)
# end_comp = time.clock()
# print("It took {} seconds".format(round(end_comp - start_comp, 2)))
# print("Accuracy is {} with {} dimensions".format(res_comp, compressed.shape[1]))

print(G_circ[:, 1:3])
# print(len(a))
# print(a[0][1])
# print(a[0][0].shape)
# print(compressed.shape)
# print(list(map(clearlab, a[0][1])))