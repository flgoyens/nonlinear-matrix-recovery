function [f_end,X,rmse,grad_end,k,stats] = solve_recovery_altmin(data)
%
% data.decrease_type : 'absolute' norm(grad_k+1)<= data.epsilon_x
%                       also define data.epsilon_x
% data.decrease_type : 'relative' : norm(grad f(X_k+1,U_k+1)) <= data.theta*norm(grad f(X_k,U_k+1))
%                       also define data.theta
%
% data.subproblem_maxiter:
% data.tolerance :
% data.x0:
% data.solver:
% data.maxiter:
% data.verbosity:
%
% 

% sigma, d,c and PUorth are global variables.
n = data.n;
s = data.s;
r = data.r;


% for random svd
data.q = 2;% number of power iterations
data.p = 5;% extra width of random matrix

% Define affine manifold
if(strcmp(data.measurements ,'dense'))
    problem.M = linearfactoryAb(n,s,data.A,data.b);
elseif(strcmp(data.measurements,'completion'))
    problem.M = samplesfactory(n,s,data.mask,data.samples);
else
    fprintf('Error: measurement type not recognized\n');
end

%  Define model for lifting
if(strcmp(data.model,'gaussian_kernel'))
    sigma = data.sigma;
    problem.cost  =  @cost_gaussian_kernel;
    problem.egrad = @egrad_gaussian_kernel;
elseif(strcmp(data.model,'monomial_kernel'))
    if(~isfield(data, 'c'))
        data.c = 1;
    end
    c = data.c;
    d = data.d;
    problem.cost  =  @cost_monomial_kernel;
    problem.egrad = @egrad_monomial_kernel;
    problem.ehess = @ehess_monomial_kernel;
else
    fprintf('Error: the model for the lifting is not recognized. Check data.model\n');
end

% Define lift
if(strcmp(data.model,'monomial_kernel'))
    lift = @(x) kernel_uos(x,d,c);
elseif(strcmp(data.model,'gaussian_kernel'))
    lift = @(x) kernel_rbf(x,sigma);
else
    fprintf('Error: the model for the lifting is not recognized. Check data.model\n');
end

% Some options for main AM scheme
if(~isfield(data,'solver'))
    data.solver = 'RTR';
end
if(~isfield(data, 'tolerance'))
    data.tolerance = 1e-6;
end

if(~isfield(data, 'maxiter'))
    data.maxiter = 1000;
end
if(isfield(data, 'x0'))
    X = data.x0.X;
else
    X = problem.M.rand();
end

% Some options for the subproblems
if(isfield(data, 'subproblem_maxiter'))
    options.maxiter = data.subproblem_maxiter;
else
    options.maxiter = 1000;
end
if(~isfield(data, 'decrease_type'))
    data.decrease_type = 'relative';
end
if(strcmp(data.decrease_type,'absolute'))
    if(isfield(data,'epsilon_x'))
        options.tolgradnorm = data.epsilon_x;
    else
        options.tolgradnorm = 1e-6;
    end
else% we assume decrease_type is 'relative'
    if(~isfield(data,'theta'))
        data.theta = 0.9;
    end
end
options.verbosity = 0;

if(~isfield(data,'random_svd'))
    data.random_svd = 0;
end

if(isfield(data,'x_true'))
    %     stats.rmse = @(problem, x) norm(x.X-data.x_true.X,'fro')/sqrt(n*s);
    err = @(X) norm(X-data.x_true.X,'fro')/sqrt(n*s); % rmse
end


errvec = zeros(1,data.maxiter);
norm_grad = zeros(1,data.maxiter);
f_values = zeros(1,data.maxiter);
f_eval = zeros(1,data.maxiter);
grad_eval = zeros(1,data.maxiter);
hess_eval = zeros(1,data.maxiter);
times = zeros(1,data.maxiter);

G = lift(X);

if(data.random_svd)
    [Ur,~, ~] = rsvd(G, r,data.q,data.p);
    Ur= Ur(:,1:r);
else
    Ur = truncate_svd(G,r);
end
PUorth = eye(size(G,1))-Ur*Ur';
store = StoreDB();
tic()
norm_grad(1) = problem.M.norm(X,problem.M.egrad2rgrad(X,problem.egrad(X,store)));
f_values(1) = problem.cost(X,store);
times(1) = toc();
if(isfield(data,'x_true'))
    errvec(1) = err(X);
end
if(data.verbosity >=2)
    fprintf('X_0:    grad = %d.  ||X_0-X_*|| = %d\n',norm_grad(1),errvec(1));
end
% Alternating minimisation
grad_norm_x = data.tolerance + 1;
k = 1;

while( k<data.maxiter && grad_norm_x > data.tolerance)

    if(strcmp(data.decrease_type,'relative'))
        options.tolgradnorm = data.theta*grad_norm_x;
    end

    stats = statscounters({'costcalls', 'gradcalls', 'hesscalls'});
    options.statsfun = statsfunhelper(stats);
    % subproblem in x
    if(strcmp(data.solver, 'RTR'))
        [X, ~, info] = trustregions(problem,X,options);
    elseif(strcmp(data.solver, 'RSD'))
        [X, ~, info] = steepestdescent(problem,X,options);% Goldstein condition
    elseif(strcmp(data.solver, 'RLBFGS'))
        [X, ~, info] = rlbfgs(problem,X,options);
    end

    f_eval(k+1) = f_eval(k) + info(end).costcalls;
    grad_eval(k+1) =  grad_eval(k) +info(end).gradcalls;
    hess_eval(k+1) = hess_eval(k) + info(end).hesscalls;

    % (truncated) svd
    if(k==5)
        data.q = 1;
    end
    if(k==50)
        data.q = 0;
    end
    G = lift(X);
    if(data.random_svd)
        [Urnew,~, ~] = rsvd(G, r,data.q,data.p);
    else
        Urnew = truncate_svd(G,r);
    end
    PUorth = eye(size(G,1))-Urnew*Urnew';
    grad_norm_x = problem.M.norm(X,problem.M.egrad2rgrad(X,problem.egrad(X,store)));

    % storing values
    errvec(k+1) = err(X);
    norm_grad(k+1) = grad_norm_x;
    f_values(k+1) = problem.cost(X,store);
    times(k+1) = toc();
    if(data.verbosity >=2)
        fprintf('Iter %d: Inner steps: %d. eps_x = %d. grad f_x(X_k+1,U_k)= %d. grad f(X_k+1,U_k+1) = %d. ||X-M||/sqrt(ns) = %d\n',...
            k, info(end).iter, options.tolgradnorm,  info(end).gradnorm, grad_norm_x,errvec(k+1));
    end
    k = k+1;
end % end of altmin loop

time_solve = toc();
if(k>=data.maxiter)
    fprintf("Warning: Maximum number of iterations reached in alternative minimisation\n");
end
fprintf("Altmin ends in %d seconds and %d iterations with grad norm: %d and function value %d\n", time_solve,k, norm_grad(k),f_values(k));

if(isfield(data,'x_true'))
    fprintf('RMSE for Altmin: %d\n',errvec(k-1));
end
%     figure();
%     semilogy(0:(k-1),norm_grad(1:k) , '.-');
%     xlabel('Iteration #');
%     ylabel('Gradient norm');
%     title('Convergence of alternative minimisation');

stats.nsvd = k-1;
stats.costcalls = f_eval(1:k);
stats.gradcalls = grad_eval(1:k);
stats.hesscalls = hess_eval(1:k);
stats.costvalues = f_values(1:k);
stats.gradnorms = norm_grad(1:k);
stats.times = times(1:k);
stats.errors = errvec(1:k);
stats.iter = 0:(k-1);


grad_end = grad_norm_x;
rmse = errvec(k-1);
f_end = problem.cost(X,store);
% Monomial kernel
    function [fx,store] = cost_monomial_kernel(x,store)
        %         store = prepare(x, store);
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


end
