import numpy as np
import matplotlib.pyplot as plt
from numpy import random as rnd
from numpy import linalg as la
from numpy.linalg import svd
from math import factorial


def nchoosek(n, k):
    f = factorial
    return f(n) // f(k) // f(n-k)


def plot_svd(X):
    u, s, v = svd(X)
    plt.figure()
    plt.title('Singular values in log scale')
    plt.semilogy(s, 'bo')
    plt.grid(True)
    plt.show()


def my_rescale(X, alpha):
    # reshapes the matrix X coordinate-wise so that the max entry is now alpha
    n, s = X.shape
    P = np.zeros([n, s])
    for i in range(0, n):
        # for each row we will rescale
        b = np.amax(X[i, :])
        a = np.amin(X[i, :])
        if(b == a):
            for k in range(0, s):
                P[i, k] = alpha*(X[i, k] - 0.5*(b+a))
        else:
            for k in range(0, s):
                P[i, k] = alpha*(X[i, k] - 0.5*(b+a))/(0.5*(b-a))
    return P


def truncate_svd(K, r):
    u, s, v = svd(K)
    return u[:, 0:r]
