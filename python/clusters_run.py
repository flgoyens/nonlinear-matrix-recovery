# import matplotlib
# matplotlib.use('Agg')
#!/usr/bin/python3
# import pymanopt
import numpy as np
import matplotlib.pyplot as plt
from numpy import random as rnd
from numpy import linalg as la
from numpy.linalg import svd
import nlmc_tools

import data_generation as dg
import liftings as lift
import solvers
import clusters_tools as clust


def main():
    n = 10
    nc = 5
    npoints = 30
    radius = 0.1
    M = dg.generate_clusters2(n, nc, radius, npoints)
    sigma = 5

    # K_M = lift.gaussian_kernel(M, sigma)
    # nlmc_tools.plot_svd(K_M)
    # plt.figure(1)
    # plt.scatter(M[0, :], M[1, :], c='r', marker='.', label="M")
    # plt.axis('scaled')
    # plt.show()

    rho = 0.1
    mask, samples = dg.generate_missing_entries(M, rho)
    verbose = 2
    s = nc*npoints
    x0 = None
    # direct solve:
    Xdirect, Uend = solvers.solve_clusters(mask, samples, nc, sigma, s, verbose, x0)
    RI = clust.evaluate_clustering(Xdirect, M, sigma)
    print('direct solve:')
    print('RI = '+str(RI))
    print('dist = '+str(la.norm(Xdirect-M)))

    plt.figure(2)
    plt.scatter(M[0, :], M[1, :], c='r', marker='.', label="M")
    plt.scatter(Xdirect[0, :], Xdirect[1, :], c='b', marker='.', label="Xend")
    plt.legend()
    plt.savefig('clusters_run.eps', format='eps')
    plt.show()


if __name__ == '__main__':
    main()
