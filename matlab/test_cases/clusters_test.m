% clusters_test.m
clear all;
close all;

n = 10;
rho = 0.8;

n_clusters = 7;
pts_per_clusters = 20;
data.n = n;
data.s = s;
data.sigma = 1;
radius = 0.01;

data.measurements = 'completion';
data.model = 'gaussian_kernel';
data.tolerance = 1e-6;
data.verbosity = 0;
data.solver = 'RTR';
data.mask = mask;
data.samples = samples;
data.maxiter = 500;

[X,s] = generate_clusters(n,pts_per_cluster,n_clusters,radius);
X = my_rescale(X);
[mask, samples] = generate_mask(X,rho);
k_mono = kernel_uos(X,data.d,data.c);
r = rank(k_mono,1e-7);
data.x_true = struct('X',X,'U',svd_r(k_mono,r));

[f_end,x_end,error_end,grad_end,k,info] = solve_recovery_grass(data);









