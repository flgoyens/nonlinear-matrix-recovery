#!/usr/bin/python3
import pymanopt
import autograd.numpy as np
import matplotlib.pyplot as plt
# import numpy as np
from numpy import random as rnd
from numpy import linalg as la
from numpy.linalg import svd

from pymanopt.manifolds import Product, Grassmann, Gaussian_Subspace, Samples
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, TrustRegions
import data_generation as dg
import liftings as lift
import solvers
import nlmc_tools
from math import floor


def main():
    verbose = 0
    n = 10
    rho = 0.8
    n_sub = 1
    dim_sub = 1
    s = 50
    xsi = 0.01  # standard deviation of measurement noise

    M = dg.generate_uos_2(n, n_sub, dim_sub, s)
    d_monomial = 2
    c = 1
    monomial_ker = lift.monomial_kernel(M, d_monomial, c)
    r_monomial = la.matrix_rank(monomial_ker, 1e-10)
    # print("rank monomial kernel: "+str(r_monomial))

    mask, samples = dg.generate_missing_entries(M, rho)
    samples_noisy = samples + xsi*rnd.randn(len(samples))

    print('||M[mask]-samples|| = '+str(la.norm(M[mask]-samples_noisy)))
    return

    maxiteration = 500
    tolerance = 1e-5
    # U, sing, v = svd(monomial_ker)
    # Ur = U[:, 0:r_monomial]
    # x0 = (M, Ur)
    x0 = None
    Xend_monomial, Uend = solvers.solve_monomials_noise(mask, samples_noisy, samples, xsi, d_monomial, r_monomial, s, c, maxiteration, tolerance, verbose, x0)
    print("distance to solution monomials: "+str(np.linalg.norm(Xend_monomial - M)))
    print("feasibility : "+str(np.linalg.norm(Xend_monomial[mask] - samples)))


if __name__ == '__main__':
    main()
