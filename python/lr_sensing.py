# import numpy as np
import pymanopt
# from numpy import random
import autograd.numpy as np
from numpy.linalg import svd

from pymanopt.manifolds import Product, Grassmann, Gaussian_Subspace, Samples
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent
from pymanopt.solvers import TrustRegions
from matplotlib import pyplot as plt
import data_generation as dg


def main():
    n = 20
    s = 30
    r = 1
    rho = 0.5
    M = np.random.randn(n, r)@np.random.randn(r, s)

    A, b, Q = dg.generate_gaussian_measurements(M, n, s, rho)
    print("initial error : "+str(np.linalg.norm(A@M.reshape(n*s, 1)-b)))

    M1 = Gaussian_Subspace(A, b, n, s)
    M2 = Grassmann(n, r)
    manifold = Product([M1, M2])

    @pymanopt.function.Autograd
    def cost(x, u): return np.linalg.norm(x - u@u.T@x)**2
    problem = Problem(manifold=manifold, cost=cost)

    # solver = SteepestDescent(maxiter=500, mingradnorm=1e-06, logverbosity=2)
    solver = TrustRegions(maxiter=100, mingradnorm=1e-06, logverbosity=2)

    Xopt, optlog = solver.solve(problem)
    xend = Xopt[0]
    print("feasibility error :"+str(np.linalg.norm(A@xend.reshape(n*s, 1)-b)))
    print("distance to solution: "+str(np.linalg.norm(M-xend)))
    print("rank xend: " + str(np.linalg.matrix_rank(xend, tol=1e-8)))

    u, s, v = svd(xend)
    plt.figure(1)
    plt.semilogy(s)
    plt.show()

    dic_iter = optlog['iterations']
    x = dic_iter['iteration']
    y = dic_iter['gradnorm']

    plt.figure(2)
    plt.semilogy(x, y, 'b', label='line 1', linewidth=2)
    plt.xlabel('iterations')
    plt.ylabel('Gradient norm')
    plt.title('Test manifolds problem')
    plt.savefig("test.png")
    plt.show()


if __name__ == '__main__':
    main()
