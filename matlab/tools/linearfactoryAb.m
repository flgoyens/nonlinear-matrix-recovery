function M = linearfactoryAb(n,s,A,b)
% X is n x s
% A has dimension m x ns
% and b has dimension m x 1
% m is the number of measurements i.e. rho*n*s with rho undersampling
% X is n x s and X(:) is a vector of size ns
% A*X(:) = b 
% X is stored as a matrix so it constantly needs to be vectorized and
% reshaped so that might not be optimal but that is mathematically the best
% way to do this I think. 
% one row of A is a matrix A_i in nxs and the m constraints are <A_i,X> = b_i

% Returns a manifold struct to optimize over real matrices.
%
% function M = euclideanfactory(n)
% function M = euclideanfactory(n, s)
%
% Returns M, a structure describing the Euclidean space of real matrices,
% equipped with the standard Frobenius distance and associated trace inner
% product, as a manifold for Manopt.
%
% n and s in general can be vectors to handle multidimensional arrays.
% If either of n or s is a vector, they are concatenated as [n, s].
%
% Using this simple linear manifold, Manopt can be used to solve standard
% unconstrained optimization problems, for example in replacement of
% Matlab's fminunc.
%
% See also: euclideancomplexfactory

% This file is derived from Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: Bamdev Mishra, May 4, 2015. Florentin Goyens
% Change log: 
%
%   July 5, 2013 (NB):
%       Added egred2rgrad, ehess2rhess, mat, vec, tangent.
%   May 4, 2015 (BM):
%       Added functionality to handle multidimensional arrays.

[Qhat,~] = qr(A',0);


    % The size can be defined using both m and n, or simply with m.
    % If m is a scalar, then n is implicitly 1.
    % This mimics the use of built-in Matlab functions such as zeros(...).
    if ~exist('s', 'var') || isempty(s)
        if numel(n) == 1
            s = 1;
            fprintf('should be here\n');
        else
            s = [];
            fprintf('should not be here\n');
        end
    end
    
    dimensions_vec = [n(:)', s(:)']; % We have a row vector.
    
    M.size = @() dimensions_vec;
    
    M.name = @() sprintf('Euclidean space R^(%s) such that AX(:)=b', num2str(dimensions_vec));
    
    M.dim = @() size(A,2)- size(A,1); % it used to be prod(dimensions_vec)-size(A,1)... 
    
    M.inner = @(x, d1, d2) d1(:).'*d2(:) ;
    
    M.norm = @(x, d) norm(d(:), 'fro');
    
    M.dist = @(x, y) norm(x(:) - y(:), 'fro');
    
    M.typicaldist = @() sqrt(prod(dimensions_vec));
    
    M.proj = @proj;
    function y = proj(x,d)
    % projection of a vector on the tangent space
    % P_KerA = d - pinv(A)*A*d; and KerA = tgt space
%         y = reshape(d(:)-(pinvA*A)*d(:),[n,s]);
        y = reshape(d(:)-Qhat*(Qhat'*d(:)),[n,s]);
    end

    M.egrad2rgrad = @egrad2rgrad;
    function y = egrad2rgrad(x,g)
               y = reshape(g(:)-Qhat*(Qhat'*g(:)),[n,s]);
    end

    % eh = Hess \bar{f}[u]
    M.ehess2rhess = @(x, eg, eh, d)  reshape(eh(:)-Qhat*(Qhat'*eh(:)),[n,s]); 
    
    M.tangent = M.proj;
    
    M.exp = @exp;% retraction is the Id
    function y = exp(x, d, t)
        
        if nargin == 3
            y = x + t*d;
        else
            y = x + d;
        end
    end
    
    M.retr = M.exp;
	
	M.log = @(x, y) y-x; 

    M.hash = @(x) ['z' hashmd5(x(:))];
    
    M.rand = @rand;
    function y = rand()
        % returns random point on the manifold
        v = randn(dimensions_vec);
        y = reshape( A\b + v(:)- Qhat*(Qhat'*v(:)),[n,s]) ;
    end

    M.randvec = @randvec;
    function u = randvec(x)
        %returns a unit norm tangent vector at x
        u = randn(dimensions_vec);
        u = u(:) - Qhat*(Qhat'*u(:));
        u = u / norm(u(:), 'fro');
        u = reshape(u,[n,s]);
    end
    
    M.lincomb = @matrixlincomb;
    
    M.zerovec = @(x) zeros(dimensions_vec);
    
    M.transp = @(x1, x2, d) d;
    
    M.pairmean = @(x1, x2) .5*(x1+x2);
    
    M.vec = @(x, u_mat) u_mat(:);
    M.mat = @(x, u_vec) reshape(u_vec, dimensions_vec);
    M.vecmatareisometries = @() true;

end
