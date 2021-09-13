#!/usr/bin/python3
import pymanopt
import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
from numpy import random as rnd
from numpy import linalg as la
# from numpy.linalg import svd
from pymanopt.manifolds import Product, Euclidean, Grassmann, Gaussian_Subspace, Samples
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, TrustRegions
import liftings as lift
import nlmc_tools


def solve_monomials(mask, samples, d, r, s, c=1, maxiteration=500, tolerance=1e-5, verbose=1, x0=None):
    manifold = Product((Samples(mask, samples), Grassmann(s, r)))
    @pymanopt.function.Autograd
    def cost(x, u):
        K = lift.monomial_kernel(x, d, c)
        return np.trace(K - u@u.T@K)
    problem = Problem(manifold=manifold, cost=cost)
    problem.verbosity = verbose
    # solver = SteepestDescent(maxiter=2000, mingradnorm=1e-04, logverbosity=2)
    solver = TrustRegions(maxiter=maxiteration, mingradnorm=tolerance, logverbosity=1)
    Xopt, optlog = solver.solve(problem, x0)
    return Xopt


def solve_monomials_noise(mask, samples_noisy, samples, sigma, d, r, s, c=1, maxiteration=500, tolerance=1e-5, verbose=0, x0=None):
    n, s = mask.shape
    manifold = Product((Euclidean(n, s), Grassmann(s, r)))
    @pymanopt.function.Autograd
    def cost(x, u):
        K = lift.monomial_kernel(x, d, c)
        diff = x[mask]-samples_noisy
        return np.trace(K - u@u.T@K) + mu*diff.T@diff
    problem = Problem(manifold=manifold, cost=cost)
    problem.verbosity = verbose
    # solver = SteepestDescent(maxiter=2000, mingradnorm=1e-04, logverbosity=2)
    solver = TrustRegions(maxiter=maxiteration, mingradnorm=tolerance, logverbosity=1)
    tol1 = 1e-7
    error = tol1 + 1
    i = 0
    mu = 1e-6
    gamma = 10
    n_iter = 10
    mu_array = np.zeros([n_iter+1, 1])
    error_rank_array = np.zeros([n_iter+1, 1])
    error_noise_array = np.zeros([n_iter+1, 1])
    error_feasibility_array = np.zeros([n_iter+1, 1])
    while(error >= tol1 and i <= n_iter):
        Xopt, optlog = solver.solve(problem, x0)
        x0 = Xopt
        x, u = Xopt
        K = lift.monomial_kernel(x, d, c)
        error_rank = np.trace(K - u@u.T@K)
        error_noise = la.norm(x[mask]-samples_noisy)
        infeasibility = la.norm(x[mask]-samples)
        error_rank_array[i] = error_rank
        error_noise_array[i] = error_noise
        error_feasibility_array[i] = infeasibility
        mu_array[i] = mu
        print('Iteration: '+str(i)+', mu = '+str(mu)+', error_rank = '+str(error_rank)+', error_noise = '+str(error_noise)+', infeasibility = '+str(infeasibility))
        error = error_rank + error_noise
        mu = gamma*mu
        i = i + 1
    plt.figure()
    titre = 'Noisy problem: sigma = '+str(sigma)
    plt.title(titre)
    plt.semilogy(error_rank_array[0:i], 'bx-', label='rank error')
    plt.semilogy(error_noise_array[0:i], 'rx-', label='noise error')
    plt.semilogy(error_feasibility_array[0:i], 'gx-', label='infeasibility')
    plt.grid(True)
    plt.legend([r'$|| \Phi(X) - P_{U}\Phi(X)||^2$', r'$||A(X)-\tilde{b}||$', r'$||A(X)-b||$'])
    x = np.arange(0, len(mu_array))
    plt.xticks(x, mu_array)
    plt.savefig('./plots/test_noise_small_lambda.eps', format='eps')
    plt.show()
    return Xopt


def solve_clusters(mask, samples, rank, sigma, s, verbose, x0=None):
    manifold = Product((Samples(mask, samples), Grassmann(s, rank)))
    @pymanopt.function.Autograd
    def cost(x, u):
        K = lift.gaussian_kernel(x, sigma)
        return np.trace(K - u@u.T@K)
    problem = Problem(manifold=manifold, cost=cost)
    problem.verbosity = verbose
    # solver = SteepestDescent(maxiter=2000, mingradnorm=1e-04, logverbosity=2)
    solver = TrustRegions(maxiter=500, mingradnorm=1e-04, logverbosity=1)
    Xopt, optlog = solver.solve(problem, x0)
    # rr = la.matrix_rank(lift.monomial_kernel(Xend, d, c), 1e-9)
    # print("final rank monomial: " + str(rr))
    return Xopt
