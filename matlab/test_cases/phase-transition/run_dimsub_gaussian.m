clear all; close all;
format long;
% Tests over dimension of subspaces - Gaussian measurements

decrease_type ='relative';
epsilon = 1e-6;
epsilon_x = 0.1;
n = 10;
pts_per_sub = 20;
n_subspace = 2;
ntrials = 20;

RHO = 0.9:-0.1:0.4;
DIMSUBSPACES = 1:5;
recovery = zeros(length(RHO),length(DIMSUBSPACES));
distance = zeros(length(RHO),length(DIMSUBSPACES));

AMgrad = zeros(length(RHO), length(DIMSUBSPACES), ntrials, 7);
AMtr = zeros(length(RHO), length(DIMSUBSPACES), ntrials, 7);
RTR = zeros(length(RHO), length(DIMSUBSPACES), ntrials, 7);

for i = 1: length(RHO)
    rho = RHO(i)
    for j = 1:length(DIMSUBSPACES)
        dim_subspace = DIMSUBSPACES(j) 
        parfor k = 1:ntrials
            
            [Xtrue,s] = generate_X_uos(n,dim_subspace,n_subspace,pts_per_sub);
            m = ceil(rho*n*s);
            A = randn(m,n*s);
            b = A*Xtrue(:);
            r = rank(kernel_uos(Xtrue,2));
            X = reshape(A\b,[n,s]);
            
            solver= 'myArmijo';
            [~,X1,error_end1,grad_end1,k1,time1,stats1] = solve_altmin_uos(A,b,n,s,r,X,Xtrue,solver,decrease_type,epsilon,epsilon_x);
            AMgrad(i,j,k,:) = [error_end1  k1  time1 stats1.costcalls(end) stats1.gradcalls(end) stats1.hesscalls(end) grad_end1];
            
            solver= 'TRhess';
            [f_end6,X6,error_end6,grad_end6,k6,time6,stats6] = solve_altmin_uos(A,b,n,s,r,X,Xtrue,solver,decrease_type,epsilon,epsilon_x);
            AMtr(i,j,k,:) = [error_end6  k6  time6 stats6.costcalls(end) stats6.gradcalls(end) stats6.hesscalls(end) grad_end6];
            
            solver = 'RTR';
            [f_end8,X8,error_end8,grad_end8,k8,stats8] = solve_grass_uos(A,b,n,s,r,X,Xtrue,epsilon,solver);
            RTR(i,j,k,:) = [error_end8  stats8(end).iter  stats8(end).time stats8(end).costcalls stats8(end).gradcalls stats8(end).hesscalls grad_end8];
                    
        end
    end
    
end

    save('results_gaussian_dimsub_20pps_20trials.mat','RTR','AMgrad','AMtr');
