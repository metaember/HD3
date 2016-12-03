from scipy.linalg import hadamard
from scipy.linalg import circulant
from scipy.signal import fftconvolve
# WalsHadamard thanks to http://www.quantatrisk.com/files/wht/WalshHadamard.py
#from WalshHadamard import *

import numpy as np
import math


def transform(dataset, transformation, target_dimension):
    transformations = ("hdhdhd", "gcircd", "pd")
    transformation = transformation.lower()
    if transformation not in transformations:
        raise ValueError("Available transformations are : {}".format(", ".join(transformations)))

    if transformation == "hdhdhd":
        matrix = generate_hdhdhd()
    elif transformation == "gcircd":
        matrix = generate_gcircd()
    elif transformation == "pd":
        matrix = generate_pd()
    else:
        assert False

    return np.dot(matrix, dataset)


def generate_hdhdhd():
    return

def generate_gcircd():
    return

def generate_pd():
    return

def generate_d():
    return

def generate_h():
    return

def generate_g():
    return

def generate_gcirc():
    return

class Transform(object):
    def __init__(self):
        pass


    def preprocess(self,x):
        """
        As Walsh-Hadamard and Hadamard matrices only work for n = 2**k,
        we need to preprocess x to have the correct size
        """
        n = len(x)
        k = 1
        while (2**k < n):
            k = k+1
        if (n != 2**k):
            m = 2**k - n
            n = 2**k
            x = np.concatenate([x,np.zeros(m)])
        return x

    def HD3(self,x):
        x = self.preprocess(x)
        n = len(x)
        H = hadamard(n)
        D1 = 2*np.random.binomial(1,0.5,n)-1
        D2 = 2*np.random.binomial(1,0.5,n)-1
        D3 = 2*np.random.binomial(1,0.5,n)-1
        Dx = D3*x
        HDx = np.dot(H,Dx)
        DHDx = D2*HDx
        HDHDx = np.dot(H,DHDx)
        DHDHDx = D1*HDHDx
        HDHDHDx = np.dot(H,DHDHDx)
        return HDHDHDx

    def JLT(self,x,N=2,eps = 0.25,m=-1):
        if (m < 1):
            m = int(math.ceil(8*math.log(N)/(eps**2)))

        n = len(x)
        G = 1/math.sqrt(m) * np.random.normal(0,1,(m,n))
        return np.dot(G,x)

    def G_circ(self, x):
        n = len(x)
        Gx = fftconvolve(np.random.normal(0,1,n),x, mode = 'same')
        return Gx

    def GD(self,x):
        n = len(x)
        G = np.random.normal(0,1,n)
        D = 2*(np.random.binomial(1,0.5,n)-1)
        Dx = D*x
        GDx = fftconvolve(G,Dx,mode='same')
        return GDx

    def GDH(self,x):
        x = self.preprocess(x)
        n = len(x)
        G = np.random.normal(0,1,n)
        D = 2*(np.random.binomial(1,0.5,n)-1)
        Hx = FWHT(x)
        DHx = D*Hx
        GDHx = fftconvolve(G,DHx, mode = 'same')
        return GDHx


    # TO DO
    def FJLT(self,x,N,eps=0.25,m=-1):
        if (m < 1):
            m = int(math.ceil(8*math.log(N)/(eps**2)))
        n = len(x)

T = Transform()
x = (1,0,-3,2)

print(T.HD3(x))
print(T.JLT(x,m=4))
print(T.G_circ(x))
print(T.GD(x))
print(T.GDH(x))
