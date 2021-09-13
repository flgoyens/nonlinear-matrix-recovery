import numpy as np
from numpy import linalg as la, random as rnd
from math import floor
import operator as op
from functools import reduce


def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom


def generate_gaussian_measurements(M, n, s, rho):
    m = floor(rho*n*s)
    A = rnd.randn(m, n*s)
    b = A@M.reshape([n*s, 1])
    Q, _ = la.qr(A.T)
    return (A, b, Q)


def generate_missing_entries(M, rho):
    n, s = M.shape
    m = floor(rho*n*s)
    mask = np.zeros((n*s, 1), dtype=bool)
    order = rnd.permutation(n*s)
    for i in range(m):
        mask[order[i]] = True
    mask = np.reshape(mask, (-1, s))
    samples = M[mask]
    return (mask, samples)


def generate_uos_2(n, n_sub, dim_sub, s):
    (pts_per_sub, remainder) = divmod(s, n_sub)
    X = np.zeros((n, s))
    U, *_ = la.qr(rnd.randn(n, dim_sub))
    p1 = pts_per_sub + remainder
    X[:, (0*p1):(0*p1 + p1)] = U@rnd.randn(dim_sub, p1)
    p = pts_per_sub
    for i in range(0, n_sub-1, 1):
        U, *_ = la.qr(rnd.randn(n, dim_sub))
        X[:, (i*p + p1):(i*p + p + p1)] = U@rnd.randn(dim_sub, p)
    return X


def generate_uos(n, n_sub, dim_sub, pts_per_sub):
    p = pts_per_sub
    s = n_sub*p
    X = np.zeros((n, s))
    for i in range(0, n_sub, 1):
        U, *_ = la.qr(rnd.randn(n, dim_sub))
        X[:, (i*p):(i*p + p)] = U@rnd.randn(dim_sub, p)
    return (X, s)


def generate_clusters2(n, nc, sigma, npoints):
    centers = rnd.randn(n, nc)
    s = nc*npoints
    X = np.zeros([n, s])
    for k in range(0, nc):
        temp = sigma*rnd.randn(n, npoints) + np.array([centers[:, k], ]*npoints).transpose()
        X[:, k*npoints:(k+1)*npoints] = temp
    return X
