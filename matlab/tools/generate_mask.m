function [mask, samples] = generate_mask(X,rho)
% Function to generate an entry sensing mask on a matrix
% [mask, samples] = generate_mask(X,rho)
% Input:
% X: matrix to sample of size nxs
% rho : sampling rate between 0 and 1 (number of entries visible/ns)
% Output: 
% mask : boolean matrix of size nxs such that X(mask) = samples
[n,s] = size(X);
m = floor(rho*n); % number of observations per column

mask = false(n,s);
for j = 1:s
    rind = randperm(n);
    mask(rind(1:m),j) = true;
end
samples = X(mask);
end