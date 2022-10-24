% increasing the number of subspaces 
clear all;
close all;
rng(0)
n = 15; 
rho = 0.9;

% generate union of subspace
dim_subspace = 2;
n_subspace = 2; 
pts_per_sub = 50;
[X,s] = generate_uos(n,dim_subspace,n_subspace,pts_per_sub);


[mask, samples] = generate_mask(X,rho);

data.n = n;
data.s = s;
data.d = 2;
data.c = 1;
N = nchoosek(n+data.d,data.d);

data.measurements = 'completion';

data.decrease_type = 'absolute';
data.tolerance = 1e-6;
data.verbosity = 2;
data.solver = 'RTR';
data.mask = mask;
data.samples = samples;
data.maxiter = 500;
data.random_svd = 0;
data.theta = 0.9;
data.verbosity = 2;

monomial_kernel = kernel_uos(X,data.d);
r_mono = rank(monomial_kernel,1e-7);

solve_monomial_kernel = 1;

% monomials
if(solve_monomial_kernel)
data.model = 'monomial_kernel';
data.r = ceil(r_mono);
data.x_true = struct('X',X,'U',svd_r(monomial_kernel,data.r));
%   [f_end,x_end,rmse_monomial,grad_end,k,info] = solve_recovery_grass(data);
 [f_end,x_end,rmse_monomial,grad_end,k,info] = solve_recovery_altmin(data);
end

fprintf('N: %d, s: %d, d: %d, r: %d\n',N,s,data.d,data.r);

% if(solve_monomial_kernel)
% fprintf('Monomials error: %d, cond Hess: %d\n',rmse_monomial,1);
% end



