from scipy.linalg import hadamard
from scipy.linalg import circulant
from sklearn.decomposition import PCA as PCAscl
from scipy.signal import fftconvolve
from scipy.linalg import circulant

from functions import funcs

import numpy as np
import math

class Transform(object):
    def __init__(self):
        rng = np.random.RandomState(5112)
        self.rng = rng

    def preprocess(self,x):
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
        while (2**k < n):
            k = k+1
        if (n != 2**k):
            m = 2**k - n
            n = 2**k
            if N == 1:
                x_new = np.concatenate([x,np.zeros(m)])
            else:
                x_new = np.concatenate([x,np.zeros(m*N).reshape(m,N)])
        else:
            x_new = x
        return x_new

    '''
    PCA decomposition
    '''
    def PCA(self,x, K = 256):
        pca = PCAscl(n_components=K)
        pca.fit(x)
        PCAx = pca.transform(x)
        return PCAx
    
    '''
    HDHDHD transformation
    '''
    def HD3(self,x):
        x = x.T
        x = self.preprocess(x)
        n = x.shape[0]
        H = hadamard(n)
        D1 = 2*self.rng.binomial(1,0.5,n)-1
        D2 = 2*self.rng.binomial(1,0.5,n)-1
        D3 = 2*self.rng.binomial(1,0.5,n)-1
        Dx = (x.T * D1).T
        HDx = np.dot(H,Dx)/np.sqrt(n)
        DHDx = (HDx.T * D2).T
        HDHDx = np.dot(H, DHDx)/np.sqrt(n)
        DHDHDx = (HDHDx.T * D3).T
        HDHDHDx = np.dot(H, DHDHDx)/np.sqrt(n)
        return HDHDHDx.T   
        
    '''
    Standard JLT transform with corresponding bound for m
    '''
    def JLT(self,x,K=None):
        eps = 0.4999 # Largest possible epsilon for which we have guarantees
        x = x.T
        try:
            N = x.shape[1]
        except:
            N = 2
        if K == None:    
            K = int(math.ceil(8*np.log(N)/(eps**2)))
        else:
            eps = np.sqrt(8*np.log(N)/K)
            print('... epsilon for JLT = '+str(eps))

        n = x.shape[0]
        G = self.rng.normal(0,1,(K,n))
        Gx = np.dot(G,x)/np.sqrt(n)
        return Gx.T

    '''
    Vector product with circulant matrices can be quickly computed using Fast Fourier Transofrm in O(n log(n))
    '''
    def G_circ(self, x, K = 256):
        x = x.T
        try:
            N = x.shape[1]
        except:
            N = 1
            
        n = x.shape[0]
        g = self.rng.normal(0,1,n)
        G = circulant(g)
        G = np.delete(G, range(K,n), 0)

        # Doing it in loop might not be the most efficient way, but I haven't found a way to do fftconvolve on matrices
        #if N == 1:
        #    Gx = fftconvolve(self.rng.normal(0,1,n),x,mode='same')
        #else:
        #    for i in xrange(N):
        #        Gx[:,i] = fftconvolve(self.rng.normal(0,1,n),x[:,i], mode = 'same')
        Gx = np.dot(G,x)
        return Gx.T/np.sqrt(n)

    '''
    Fast JLT using Vybiral's proposal
    '''
    def GD(self,x,K=256):
        x = x.T
        n = x.shape[0]
        try:
            N = x.shape[1]
        except:
            N = 0
        g = self.rng.normal(0,1,n)
        G = circulant(g)
        G = np.delete(G, range(K,n), 0)
        
        D = 2*(self.rng.binomial(1,0.5,n)-1)
        Dx = (x.T * D).T
        GDx = np.dot(G,x)
        #GDx = Dx
        #if N == 0:
        #    GDx = fftconvolve(G, Dx,mode='same')
        #else:
        #    for i in xrange(N):
        #       GDx[:,i] = fftconvolve(G,Dx[:,i],mode='same')
        
        return GDx.T/math.sqrt(2*n)
    
    '''
    Multiply input by a diagonal matrix with +-1 on diagonal
    '''
    def D(self,x):
        x = x.T
        n = x.shape[0]
        D = 2*self.rng.binomial(1,0.5,n) - 1
        Dx = (x.T * D).T
        return Dx.T
    '''
    Multiply input by a Gaussian matrix, basically standard JLT without 'epsilon guarnatees'
    '''
    def G(self,x,m=256):
        x = x.T
        n = x.shape[0]
        G = self.rng.normal(0,1,(m,n))
        Gx = np.dot(G,x)/np.sqrt(n)
        return Gx.T
    
    '''
    Structured compression
    '''
    def GDH(self,x,K = 256):
        x = x.T
        x = self.preprocess(x)
        n = x.shape[0]
        try:
            N = x.shape[1]
        except:
            N = 0
            
        g = self.rng.normal(0,1,n)
        G = circulant(g)
        G = np.delete(G, range(K,n),0)
        
        D = 2*(self.rng.binomial(1,0.5,n)-1)
        #Hx = x
        #for i in xrange(N): # Here it might just suffice to multiply by Walsh-Hadamard Matrix
        #    Hx[:,i] = FWHT(x[:,i])
        H = hadamard(n)
        Hx = np.dot(H,x)/np.sqrt(n)
        DHx = (Hx.T * D).T
        #GDHx = DHx
        #for i in xrange(N):
        #    GDHx[:,i] = fftconvolve(G,DHx[:,i], mode = 'same')
        GDHx = np.dot(G, DHx)
        return GDHx.T/np.sqrt(n)
    
    '''
    PHD matrix = Fast JLT transform
    '''
    def FJLT(self,x,K = 256):
        eps = 0.5
        x = x.T
        x = self.preprocess(x)
        n = x.shape[0]
        
        try:
            N = x.shape[1]
        except:
            N = 1
         
        D = 2*(self.rng.binomial(1,0.5,n)-1)
        Dx = (x.T * D).T
        H = hadamard(n)
        HDx = np.dot(H,Dx)/np.sqrt(n)
        
        q = eps*(np.log(N)**2)/n
        P = self.rng.normal(0,1/q, size=(K,n))
        Q = self.rng.binomial(1,q,size=(K,n))
        P_c = P*Q
        PHDx = np.dot(P_c,HDx)/np.sqrt(n)
        
        return PHDx.T
    
    def PDHR(self,x,K=256):
        x = x.T
        x = self.preprocess(x)
        n = x.shape[0]
        
        R = 2*(self.rng.binomial(1,0.5,n))-1
        D = 2*(self.rng.binomial(1,0.5,n))-1
        H = hadamard(n)
        g = self.rng.normal(0,1,n)
        P_circ = circulant(g)
        P_circ = np.delete(P_circ,range(K,n),0)
        Rx = (x.T * R).T
        HRx = np.dot(H,Rx)/np.sqrt(n)
        DHRx = (HRx.T *D).T
        PDHRx = np.dot(P_circ,DHRx)/np.sqrt(n)
        return PDHRx.T

    '''
    Function to calculate angular distance for two vectors
    '''
    def AngDist(self, x,y):
        xy = np.dot(x,y)
        nx = np.linalg.norm(x)
        ny = np.linalg.norm(y)
        similarity = xy/(nx*ny)
        if similarity > 1.0:
            similarity = 1.0
        if similarity < -1.0:
            similarity = -1.0
        distance = math.acos(similarity)/math.pi
        return distance
        
    def AngDistDataset(self, X,Y):
        pass
        
        
def transformDataset(orig_set, t = None, f = None, K = 256):
    T = Transform()
    if (t == 'PCA'):
        trans_set = T.PCA(orig_set,K)
        print('... performing PCA, dimension reduced to '+str(K))
    elif (t == 'HD3'):
        trans_set = T.HD3(orig_set)
        print('... performing HDHDHD')
    elif (t == 'JLT'):
        trans_set = T.JLT(orig_set,K)
        print('... performing JLT, dimension reduced to '+str(K))
    elif (t == 'G_circ'):
        trans_set = T.G_circ(orig_set,K)
        print('... performing G circulant, dimension reduced to '+str(K))
    elif (t == 'GD'):
        trans_set = T.GD(orig_set,K)
        print('... performing GD, dimension reduced to '+str(K))
    elif (t == 'D'):
        trans_set = T.D(orig_set)
        print('... performing D')
    elif (t == 'GDH'):
        trans_set = T.GDH(orig_set,K)
        print('... performing GDH, dimension reduced to '+str(K))
    elif (t == 'G'):
        trans_set = T.G(orig_set,K)
        print('... performing G, dimension reduced to '+str(K))
    elif (t == 'PDHR'):
        trans_set = T.PDHR(orig_set,K)
        print('... performing PDHR, dimension reduced to '+str(K))
    elif (t == 'FJLT'):
        trans_set = T.FJLT(orig_set,K)
        print('... performing FJLT, dimension reduced to '+str(K))
    else:
        trans_set = orig_set
        print('... no transformation')
       
    fc = funcs()
    if (f == 'sigmoid'):
        trans_set_f = fc.sigmoid(trans_set)
        print('... applying sigmoid')
    elif (f == 'signum'):
        trans_set_f = fc.signum(trans_set)
        print('... applying signum')
    elif (f == 'tanh'):
        trans_set_f = fc.tanh(trans_set)
        print('... applying tanh')
    elif (f == 'softmax'):
        trans_set_f = fc.softmax(trans_set)
        print('... applying softmax')
    elif (f == 'ReLU'):
        trans_set_f = fc.ReLU(trans_set)
        print('... applying ReLU')
    elif (f == 'softplus'):
        trans_set_f = fc.softplus(trans_set)
        print('... applying softplus')
    elif (f == 'LReLU'):
        trans_set_f = fc.LReLU(trans_set)
        print('... applying Leaky ReLU')
    elif (f == 'arctg'):
        trans_set_f = fc.arctg(trans_set)
        print('... applying arctg')
    elif (f == 'cos'):
        trans_set_f = fc.cos(trans_set)
        print('... applying cos')
    elif (f == 'sin'):
        trans_set_f = fc.sin(trans_set)
        print('... applying sin')
    elif (f == 'x_squash'):
        trans_set_f = fc.x_squash(trans_set)
        print('... applying x/(1+|x|)')
    else:
        trans_set_f = trans_set
        print('... no function applied')
    return trans_set_f
        
    
       
