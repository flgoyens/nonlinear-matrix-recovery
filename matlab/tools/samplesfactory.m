function M = samplesfactory(n,s ,mask,samples)
% Affine manifold of matrix X in (n x s) such that X(mask) = samples.
% Some of the entries of X are fixed. 
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
% Contributors: Bamdev Mishra, May 4, 2015.
% Change log: 
%   Florentin GOYENS - 2018
%   July 5, 2013 (NB):
%       Added egred2rgrad, ehess2rhess, mat, vec, tangent.
%   May 4, 2015 (BM):
%       Added functionality to handle multidimensional arrays.


    % The size can be defined using both m and n, or simply with m.
    % If m is a scalar, then n is implicitly 1.
    % This mimics the use of built-in Matlab functions such as zeros(...).
    if ~exist('s', 'var') || isempty(s)
        if numel(n) == 1
            s = 1;
        else
            s = [];
        end
    end
    
    dimensions_vec = [n(:)', s(:)']; % We have a row vector.
    
    M.size = @() dimensions_vec;
    
    M.name = @() sprintf('Euclidean space R^(%s) such that X(mask)=samples', num2str(dimensions_vec));
    
    M.dim = @() prod(dimensions_vec)- length(samples); 
    
    M.inner = @(x, d1, d2) d1(:).'*d2(:) ;
    
    M.norm = @(x, d) norm(d(:), 'fro');
    
    M.dist = @(x, y) norm(x(:) - y(:), 'fro');
    
    M.typicaldist = @() sqrt(prod(dimensions_vec));
    
    
    M.proj = @proj;
    function d = proj(x,d)% projection of a vector on the tangent space
        d(mask) = zeros(size(samples));
    end

    M.egrad2rgrad = @egrad2rgrad;
    function g = egrad2rgrad(x,g)
        g(mask) = zeros(size(samples));
    end

    M.ehess2rhess = @ehess2rhess;
    function eh = ehess2rhess(x, eg, eh, d)
                % ehess = Hess \bar{f}[d] 
        eh(mask) =  zeros(size(samples)); 
    end
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
    function y = rand()% returns random point on the manifold
        y = randn(dimensions_vec);
        y(mask) = samples ;
    end

    M.randvec = @randvec;
    function u = randvec(x) %returns a unit norm tangent vector at x
        u = randn(dimensions_vec);
        u(mask) = zeros(size(samples));
        u = u / norm(u(:), 'fro');
    end
    
    M.lincomb = @matrixlincomb;
    
    M.zerovec = @(x) zeros(dimensions_vec);
    
    M.transp = @(x1, x2, d) d;
    
    M.pairmean = @(x1, x2) .5*(x1+x2);
    
    M.vec = @(x, u_mat) u_mat(:);
    M.mat = @(x, u_vec) reshape(u_vec, dimensions_vec);
    M.vecmatareisometries = @() true;

end
