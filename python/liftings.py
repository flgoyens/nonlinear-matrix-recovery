#!/usr/bin/python3
import autograd.numpy as np
from math import acos
from cmath import exp


def gaussian_kernel(X, sigma):
    n1sq = np.sum(X*X, axis=0)
    n, s = X.shape
    D = np.array([n1sq, ]*s).transpose() + np.array([n1sq, ]*s) - 2 * X.transpose() @ X
    return np.exp(-D/(2*sigma*sigma))


def monomial_kernel(X, d=2, c=1):
    return (X.T@X + c)**d


def monomial_kernel_ij(xi, xj, d, c):
    return (xi.T@xj + c)**d
