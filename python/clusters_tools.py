import autograd.numpy as np
from numpy import linalg as la, random as rnd
# from math import floor
# import operator as op
import pymanopt
# from pymanopt.manifolds import Product, Grassmann, Gaussian_Subspace, Samples
from pymanopt import Problem
# from pymanopt.solvers import SteepestDescent, TrustRegions
import liftings as lift
import nlmc_tools


def evaluate_clustering(X, M, sigma):
    n, s = X.shape
    K_X = lift.gaussian_kernel(X, sigma)
    K_M = lift.gaussian_kernel(M, sigma)
    Adj_X = np.rint(K_X)  # nearest integer
    Adj_M = np.rint(K_M)
    bb = np.count_nonzero(Adj_X == Adj_M)
    aplusd = (bb-s)/2
    RI = aplusd/nlmc_tools.nchoosek(s, 2)
    return RI
