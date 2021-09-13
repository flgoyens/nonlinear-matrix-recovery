% increasing the number of subspaces 
clear all;
close all;
rng(0)
n = 10; 
rho = 0.8;

% generate union of subspace
dim_subspace = 2;
n_subspace = 2; 
pts_per_sub = 1000;
[X,s] = generate_uos(n,dim_subspace,n_subspace,pts_per_sub);


[mask, samples] = generate_mask(X,rho);

data.n = n;
data.s = s;
data.d = 2;
data.c = 1;
N = nchoosek(n+data.d,data.d);

data.measurements = 'completion';

data.tolerance = 1e-6;
data.verbosity = 2;
data.solver = 'RTR';
data.mask = mask;
data.samples = samples;
data.maxiter = 25;
data.random_svd = 1;
data.theta = 0.9;
data.verbosity = 2;

monomial_kernel = kernel_uos(X,data.d);
r_mono = rank(monomial_kernel,1e-7);
data.r = ceil(r_cheb + 0.0*N);
data.x_true = struct('X',X,'U',svd_r(cheb_features,data.r));
figure();
semilogy(1:s,svd(monomial_kernel),'*');
title('monomial singulat values');


solve_monomial_kernel = 1;

% monomials
if(solve_monomial_kernel)
data.model = 'monomial_kernel';
data.r = ceil(r_mono + 0.0*N);
data.x_true = struct('X',X,'U',svd_r(monomial_kernel,data.r));
%   [f_end,x_end,rmse_monomial,grad_end,k,info] = solve_recovery_grass(data);
 [f_end,x_end,rmse_monomial,grad_end,k,info] = solve_recovery_altmin(data);
end

% fprintf('Rank mono: %d, Rank cheb: %d, N: %d\n',r_mono,r_cheb,N);
fprintf('N: %d, s: %d, d: %d, r: %d\n',N,s,data.d,data.r);

if(solve_monomial_kernel)
fprintf('Monomials error: %d, cond Hess: %d\n',rmse_monomial,1);
end



