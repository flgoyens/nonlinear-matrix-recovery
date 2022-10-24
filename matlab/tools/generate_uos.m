function [X,s] = generate_uos(n,dim_subspace,n_subspace,pts_per_sub,distribution)
% Function to generate data points belonging to a union of random subspaces
% [X,s] = generate_uos(n,dim_subspace,n_subspace,pts_per_sub)
% Input: 
% n = ambien dimension
% dim_subspace = all the subspaces of that dimension
% n_subspace = the number of subspaces
% pts_per_sub = the number of data points on each subspace 
% Output: 
% X the data points in the columns of X (nxs)
% s: the total number of points s = n_subspace * pts_per_sub.
if nargin == 4
    distribution = 'normal';
end
p= pts_per_sub;
s = n_subspace*p;
X = zeros(n,s);

if strcmp(distribution, 'normal')
    for i=1:n_subspace
%             U = orth(randn(n,dim_subspace));
        [U,~] = qr(rand(n,dim_subspace),"econ");
        X(:,(i-1)*p + (1:p) ) =  U*randn(dim_subspace,p);
    end
elseif strcmp(distribution, 'uniform')
    for i=1:n_subspace
            U = orth(randn(n,dim_subspace));
%         U = qr(randn(n,dim_subspace),0);
        X(:,(i-1)*p + (1:p) ) =  2*U*(rand(dim_subspace,p)-0.5*ones(dim_subspace,p));
    end
end