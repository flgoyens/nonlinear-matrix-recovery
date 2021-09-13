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


def main():
    verbose = 2
    n = 10
    rho = 0.9
    n_sub = 1
    dim_sub = 1
    # pts_per_sub = 30
    s = 200
    # M, s = dg.generate_uos(n, n_sub, dim_sub, pts_per_sub)
    M = dg.generate_uos_2(n, n_sub, dim_sub, s)

    d_monomial = 3
    c = 1
    monomial_ker = lift.monomial_kernel(M, d_monomial, c)
    r_monomial = la.matrix_rank(monomial_ker, 1e-10)
    print("rank monomial kernel: "+str(r_monomial))
    N_mono = nlmc_tools.nchoosek(n + d_monomial, n)
    print('N monomials: ' + str(N_mono))

    mask, samples = dg.generate_missing_entries(M, rho)
    maxiteration = 500
    tolerance = 1e-5
    # U, sing, v = svd(monomial_ker)
    # Ur = U[:, 0:r_monomial]
    # x0 = (M, Ur)
    x0 = None
    Xend_monomial, Uend = solvers.solve_monomials(mask, samples, d_monomial, r_monomial, s, c, maxiteration, tolerance, verbose, x0)
    print("distance to solution monomials: "+str(np.linalg.norm(Xend_monomial - M)))

    # kernel_end = lift.monomial_kernel(Xend_monomial, d, c)
    # u, s, v = svd(kernel_end)
    # plt.figure(3)
    # plt.title('monomials end')
    # plt.semilogy(s, 'bo')
    # plt.show()


if __name__ == '__main__':
    main()
