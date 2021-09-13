% Adaptive decrease in altmin
clear all;
close all;
n = 10;
dim_subspace = 2;
n_subspace = 2;
pts_per_sub = 30; 
Theta = [1e-3 0.3 0.7 0.9];
[X,s] = generate_uos(n,dim_subspace,n_subspace,pts_per_sub);

rho = 0.9;
[mask, samples] = generate_mask(X,rho);
k = kernel_uos(X,2,1);
r = rank(k,1e-7);
U = svd_r(k,r);
data.x_true = struct('X',X,'U',U);
X0 = randn(n,s);
X0(mask) = samples;
data.x0 = struct('X',X0,'U', orth(randn(s,r)));

data.n = n;
data.s = s;
data.r = r;
data.verbosity =1;
data.decrease_type =  'relative'; %  norm(grad f(X_k+1,U_k+1)) <= data.theta*norm(grad f(X_k,U_k+1)) 
data.tolerance = 1e-6;
data.maxiter = 1000;
data.measurements = 'completion';
data.mask = mask;
data.samples = samples;
data.model = 'monomial_kernel';
data.d = 2;

i = 1;



[f_end,x_end,rmse,grad_end,k,info, lambda_min,kappa] = solve_recovery_grass(data);
rmse

for theta = Theta
    data.theta  = theta;
    [f_end,X,rmse,grad_end,k,stats] = solve_recovery_altmin(data);
    
   
    time{i} = stats.times;
    dist{i} = stats.errors;
    legendInfo{i} = ['\theta = ' num2str(theta)];
    i = i+1;
end


data.decrease_type =  'absolute';%  norm(grad_k+1)<= data.epsilon_x 
data.epsilon_x = 1e-6;
[f_end,X,rmse,grad_end,k,stats] = solve_recovery_altmin(data);

    time{i} = stats.times;
    dist{i} = stats.errors;
    legendInfo{i} = 'Greedy ';


% plot time to solution
hold on;
figure(1)
for i = 1:(length(Theta)+1)
%     length(time{i})
%     length(dist{i})
    semilogy([0 time{i}],dist{i});
end
legend(legendInfo);
ylabel('RMSE');
xlabel('Time (seconds)');

    
    
    



