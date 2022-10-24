function [f_end,x_end,rmse,grad_end,k,info, lambda_min,kappa] = solve_recovery_grass(data)
% function solve_recovery_grass()
% This is the general solver for nonlinear matrix completion on the Grassmann
% manifold. The user needs to specify the kernel or features to use as
% lifting depending on the geometry of the data. Handles matrix completion
% (entry sensing) and dense sensing (A(X) = b).
%
% For kernel map k:
% min_{X,U} f(X,U) = trace( k(X) - P_U k(X) )
%             st. U in Grass(s,r)
%                 X is in the manifold Ax=b or P_omega(X) = b
%                 X in R^{n x s}
% For features phi:
% min_{X,U} f(X,U) = norm( phi(X) - P_U phi(X) )^2_Frob
%             st. U in Grass(s,r)
%                 X is in the manifold Ax=b or P_omega(X) = b
%                 X in R^{n x s}
%
% note that P_U = U*U' with U an orthogonal basis of the subspace.
%
% Required input:
% data.n
% data.s
% data.r
% data.d (for polynomial features)
% data.sigma (for Gaussian kernel)
% data.model: 'monomial_kernel','gaussian_kernel'
% data.measurements: 'dense','completion'
% data.A and data.b or data.mask and data.samples
% Todo: it would be better to have one notation for A and mask
%
% Optional input:
% data.c (for monomial kernel)
% data.verbosity
% data.tolerance (epsilon on gradient norm)
% data.x0
% data.x_true
% data.solver
% data.maxiter


n = data.n;
s = data.s;
r = data.r;

if(~strcmp(data.model,'gaussian_kernel'))
    d = data.d;
    N=nchoosek(n+d,n);
end

if(strcmp(data.measurements ,'dense'))
        problem.M = productmanifold(struct('X', linearfactoryAb(n,s,data.A,data.b),...
            'U', grassmannfactory(s, r)));
elseif(strcmp(data.measurements,'completion'))
        problem.M = productmanifold(struct('X', samplesfactory(n,s,data.mask,data.samples),...
            'U', grassmannfactory(s, r)));
else
    fprintf('Error: measurement type not recognized\n');
end



    function store = prepare(x, store)
        if(strcmp(data.model ,'monomial_features'))
            if ~isfield(store, 'PUorth')
                store.PUorth = eye(N,N)- x.U*x.U';
            end
        else
            if ~isfield(store, 'PUorth')
                store.PUorth = eye(s,s)- x.U*x.U';
            end
        end
    end

% Monomial kernel
    function [fx,store] = cost_monomial_kernel(x,store)
        store = prepare(x, store);
        fx = trace( store.PUorth*kernel_uos(x.X,d,c));
        store = incrementcounter(store, 'costcalls');
    end
    function [gx,store] = egrad_monomial_kernel(x,store)
        store = prepare(x, store);
        gx = struct('X',2*d*x.X*(kernel_uos(x.X,d-1,c).* store.PUorth),...
            'U',-2*kernel_uos(x.X,d,c)*x.U );
        store = incrementcounter(store, 'gradcalls');
    end
    function [hv,store] = ehess_monomial_kernel(x, xdot,store)
        store = prepare(x, store);
        hv = struct('X',2*(d-1)*d*x.X*(kernel_uos(x.X,d-2,c).* store.PUorth .* ( x.X'*xdot.X + xdot.X'*x.X) )...
            +2*d*xdot.X*(kernel_uos(x.X,d-1,c).*store.PUorth)...
            -2*d*x.X*(kernel_uos(x.X,d-1,c).*(x.U*xdot.U' + xdot.U*x.U' )), ...
            'U',-2*kernel_uos(x.X,d,c)*xdot.U  -2*d*(kernel_uos(x.X,d-1,c).*( x.X'*xdot.X + xdot.X'*x.X  ))*x.U);
        store = incrementcounter(store, 'hesscalls');
    end

% Gaussian kernel
    function [fx,store] = cost_gaussian_kernel(x,store)
        store = prepare(x, store);
        fx = trace(store.PUorth* kernel_rbf(x.X,sigma));
        store = incrementcounter(store, 'costcalls');
    end
    function [gx,store] = egrad_gaussian_kernel(x,store)
        store = prepare(x, store);
        W = store.PUorth;
        KdW = kernel_rbf(x.X,sigma).*W;
        gx = struct('X',-(2/sigma^2)*x.X*(diag(sum(KdW,1))-KdW),...
            'U',-2*kernel_rbf(x.X,sigma)*x.U );
        store = incrementcounter(store, 'gradcalls');
    end


if(strcmp(data.model,'gaussian_kernel'))
    
    sigma = data.sigma;
    problem.cost  =  @cost_gaussian_kernel;
    problem.egrad = @egrad_gaussian_kernel;
    
elseif(strcmp(data.model,'monomial_kernel'))
    
    if(isfield(data, 'c'))
        c = data.c;
    else
        c = 1;
    end

    problem.cost  =  @cost_monomial_kernel;
    problem.egrad = @egrad_monomial_kernel;
    problem.ehess = @ehess_monomial_kernel;
    
else
    fprintf('Error: the embedding is not recognized. Check data.model\n');
end


% ---- Options
stats = statscounters({'costcalls', 'gradcalls', 'hesscalls'});
if(isfield(data,'x_true'))
    stats.rmse = @(problem, x) norm(x.X-data.x_true.X,'fro')/sqrt(n*s);
end
options.statsfun = statsfunhelper(stats);
if(isfield(data, 'tolerance'))
    options.tolgradnorm = data.tolerance;
else
    options.tolgradnorm = 1e-4;
end
if(isfield(data, 'verbosity'))
    options.verbosity = data.verbosity;
else
    options.verbosity = 1;
end
if(isfield(data, 'x0'))
    x0 = data.x0;
else
    x0 = [];
end

% ---- Solve.
if(isfield(data,'solver'))
    solver = data.solver;
else
    solver = 'RTR';
end
if(strcmp(solver, 'RSD'))
    if(isfield(data, 'maxiter'))
        options.maxiter = data.maxiter;
    else
        options.maxiter = 1000;
    end
    [x, xcost, info] = steepestdescent(problem,x0,options);
elseif(strcmp(solver, 'RLBFGS'))
    if(isfield(data, 'maxiter'))
        options.maxiter = data.maxiter;
    else
        options.maxiter = 1000;
    end
    [x, xcost, info] = rlbfgs(problem,x0,options);
elseif(strcmp(solver, 'TR')||strcmp(solver, 'RTR'))
    if(isfield(data, 'maxiter'))
        options.maxiter = data.maxiter;
    else
        options.maxiter = 500;
    end
    [x, xcost, info] = trustregions(problem,x0,options);
else
    fprintf('Error: no solver selected in solve_recovery_grass.m \n');
end

% Display some statistics.
if(options.verbosity==2)
    if(isfield(data,'x_true'))
        figure();
        semilogy([info.iter], [info.rmse], '.-');
        xlabel('Iteration #');
        ylabel('RMSE');
        title('Root mean square error over iterations');
    end
    
    
    figure();
    semilogy([info.iter], [info.gradnorm], '.-');
    xlabel('Iteration #');
    ylabel('Gradient norm');
    title('Convergence of the gradient norm over iterations');
end

lambda_min = 1;
kappa = 1;
if(nargout >=7)
    [~, lambda_min] = hessianextreme(problem, x, 'min', []);
    if(lambda_min < -1e-6)
        fprintf('Warning: Hessian is not PSD at termination in solve_grass_uos.m\n');
    end
end
if(nargout == 8)
    [~, lambda_max] = hessianextreme(problem, x, 'max', []);
    kappa = lambda_max/lambda_min; % conditionning number of Hessian at solution
end

f_end = xcost;
k = info(end).iter;
grad_end = info(end).gradnorm;
x_end = x;

if(isfield(data,'x_true'))
    rmse = norm(x.X-data.x_true.X,'fro')/sqrt(n*s);
else
    rmse = NaN;
end

end