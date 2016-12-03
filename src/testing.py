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

start = time.clock()

a = load_data("cifar10", batch = 1)
#D1 = 2*np.random.binomial(1,0.5,10)-1
#Gx = fftconvolve(np.random.normal(0,1,5),x, mode = 'same')

G_circ = linalg.circulant(np.random.randn(10))

end = time.clock()
print(end - start)

print(G_circ[0:10, 1:3])

print(a)