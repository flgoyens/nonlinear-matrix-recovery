% This script generates a plot to test the algorithm against wrong rank
% estimations on a recovery of clusters.
% Fix rho, do 2 loops; generate 10 problems and for each problems solve for
% rank from r-3,r-2,..,r+3.
clear all;
close all;

n = 10;
rho = 0.8;

% generate union of subspace
n_clusters = 7;
pts_per_clusters = 20;
data.n = n;
data.s = s;
data.sigma = 1;
radius = 0.01;

data.measurements = 'completion';
data.model = 'gaussian_kernel';
data.tolerance = 1e-6;
data.verbosity = 0;
data.solver = 'RTR';
data.mask = mask;
data.samples = samples;
data.maxiter = 500;

recovery = zeros(ntests,nrank, 3);
ntests = 10; 
nrank = 5;
for k = 1:ntests
    [X,s] = generate_clusters(n,pts_per_cluster,n_clusters,radius);
    X = my_rescale(X);
    [mask, samples] = generate_mask(X,rho);
    k_mono = kernel_uos(X,data.d,data.c);
    r = rank(k_mono,1e-7);
    data.x_true = struct('X',X,'U',svd_r(k_mono,r));
    
    for j = -nrank:nrank

        data.r = r + j;
        
        [f_end,x_end,error_end,grad_end,k,info] = solve_recovery_grass(data);
        recovery(k,j,:) =  [error_end<=1e-3 error_end grad_end] ;
    end
   
end

mean_recovery = sum(recovery(:,:,1),1)/ntests; 


% I = mat2gray(recovery,[0 100]);
% colormap(gray)
% imagesc(I)
figure(1)
plot(-nrank:nrank,mean_recovery,'*');
xlabel('Rank estimation');ylabel('Recovery rate');
set(gca,'xtick', -nrank:nrank);
set(gca,'xticklabel',-nrank:nrank);
set(gca,'ytick', [0 1]);
set(gca,'yticklabel',0:0.2:1);
saveas(gcf,'bad_rank_estimation_clusters','epsc')










