function [X,info] = subproblem_x(X,PUorth,data)
% Minimizes the subproblem in X in Alternating minimization using Manopt
% min_{X} f(X,U) = trace( k(X) - P_U k(X) ) 
%             st. X is in the manifold Ax=b
%                 X in R^{n x s}
% 
[n,s] = size(X);

% Define affine manifold
if(strcmp(data.measurements ,'dense'))
        problem.M = linearfactoryAb(n,s,data.A,data.b,data.pinvA); 
elseif(strcmp(data.measurements,'completion'))
        problem.M = samplesfactory(n,s,data.mask,data.samples);
else
    fprintf('Error: measurement type not recognized\n');
end

% Monomial kernel
    function [fx,store] = cost_monomial_kernel(x,store)
%         store = prepare(x, store); % maybe I could use that for computig
%         k only once per point but not for PUorth
        fx = trace( PUorth*kernel_uos(x,d,c));
        store = incrementcounter(store, 'costcalls');
    end
    function [gx,store] = egrad_monomial_kernel(x,store)
%         store = prepare(x, store);
        gx = 2*d*x*(kernel_uos(x,d-1,c).*PUorth);
        store = incrementcounter(store, 'gradcalls');
    end
    function [hv,store] = ehess_monomial_kernel(x, xdot,store)
%         store = prepare(x, store);
        hv = 2*(d-1)*d*x*(kernel_uos(x,d-2,c).* PUorth .* ( x'*xdot + xdot'*x) )...
            +2*d*xdot*(kernel_uos(x,d-1,c).*PUorth);
        
        store = incrementcounter(store, 'hesscalls');
    end
% Gaussian kernel
    function [fx,store] = cost_gaussian_kernel(x,store)
%         store = prepare(x, store);
        fx = trace( PUorth*kernel_rbf(x,sigma));
        store = incrementcounter(store, 'costcalls');
    end
    function [gx,store] = egrad_gaussian_kernel(x,store)
%         store = prepare(x, store);
        W = PUorth;
        KdW = kernel_rbf(x,sigma).*W;
        gx = -(2/sigma^2)*x*(diag(sum(KdW,1))-KdW);
        store = incrementcounter(store, 'gradcalls');
    end


%  Define model for lifting 
if(strcmp(data.model,'gaussian_kernel'))    
    sigma = data.sigma;
    problem.cost  =  @cost_gaussian_kernel;
    problem.egrad = @egrad_gaussian_kernel;   
elseif(strcmp(data.model,'monomial_kernel'))
    d = data.d;
    if(isfield(data, 'c'))
        c = data.c;
    else
        c = 1;
    end
    problem.cost  =  @cost_monomial_kernel;
    problem.egrad = @egrad_monomial_kernel;
    problem.ehess = @ehess_monomial_kernel;
else
    fprintf('Error: the model for the lifting is not recognized. Check data.model\n');
end
% % remove this after testing
%     figure;
%     checkgradient(problem);
%     figure;
%     checkhessian(problem);

% Some parameters
stats = statscounters({'costcalls', 'gradcalls', 'hesscalls'});
options.statsfun = statsfunhelper(stats);
if(isfield(data, 'subproblem_tolerance'))
    options.tolgradnorm = data.subproblem_tolerance;
else
    options.tolgradnorm = 1e-5;
end
if(isfield(data, 'subproblem_verbosity'))
    options.verbosity = data.subproblem_verbosity;
else
    options.verbosity = 0;
end
if(isfield(data,'solver'))
    solver = data.solver;
else
    solver = 'RTR';
end
if(isfield(data, 'subproblem_maxiter'))
    options.maxiter = data.subproblem_maxiter;
else
    options.maxiter = 500;
end

% Choose solver and solve
if(strcmp(solver, 'RTR'))
    [X, ~, info] = trustregions(problem,X,options);
elseif(strcmp(solver, 'RSD'))
    [X, ~, info] = steepestdescent(problem,X,options);% Goldstein condition
elseif(strcmp(solver, 'RLBFGS'))
    [X, ~, info] = rlbfgs(problem,X,options);
elseif(strcmp(solver, 'TRlinear'))
    % not implemented, I need to give another hessian
    [X, ~, info] = trustregions(problem,X,options);
elseif(strcmp(solver, 'TRFD'))
        % not implemented, I need to give no hessian
    [X, ~, info] = trustregions(problem,X,options);
end

end