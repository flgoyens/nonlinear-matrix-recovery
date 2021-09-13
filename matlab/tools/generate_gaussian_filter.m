function [A,b,X0] = generate_gaussian_filter(X,rho,sigma)
% Function to generate a Gaussian filter and measurements on a matrix
% [A,b,X0] = generate_gaussian_filter(X,rho)
% Input:
% X: matrix to sample of size nxs
% rho : sampling rate between 0 and 1 (number of entries visible/ns)
% sigma: standard deviation of the Gaussian
% Output:
% A is in  mxns
% b = A*X(:) is in R^m
% X0 = pinvA*b is a point that satisfies the measurements.

if(nargin == 2)
    sigma = 1; % is that really a good value ? given that I normalize afterwards too ... See Tanner2013 maybe
end
[n,s] = size(X);
m = ceil(rho*n*s);
A = sigma*randn(m,n*s);
A = A/norm(A); % why do we normalize again ?
b = A*X(:);

% this is in case we ever want to return some starting point x0
X0 = A\b;
% pinvA = pinv(A);
% X0 = reshape(pinvA*b,[n,s]);
end