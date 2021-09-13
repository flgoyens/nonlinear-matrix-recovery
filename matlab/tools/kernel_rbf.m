function K = kernel_rbf(X,sigma)
%RBF Compute Gaussian RBF kernel matrix with standard deviation sigma
%Output is matrix K = k_sigma(X,X2) with entries given by
%    [K]_{i,j} = exp(-||x_i - x2_j||^2/(2*sigma^2))
n1sq = sum(X.^2,1);
n1 = size(X,2);

D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
%surf(D)

K = exp(-D/(2*sigma^2));
%K(find(K<0.7))=1e-8*K(find(K<0.7));

end