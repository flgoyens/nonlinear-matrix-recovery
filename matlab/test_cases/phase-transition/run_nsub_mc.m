clear all; close all;
format long;
rng(0)
decrease_type ='relative';
epsilon = 1e-6;
theta = 0.5;
n = 10;
pts_per_sub = 10;
dim_subspace = 2;
ntrials = 1;

RHO = 0.9:-0.1:0.4;
NSUBSPACES = 1:10;
recovery = zeros(length(RHO),length(NSUBSPACES));
distance = zeros(length(RHO),length(NSUBSPACES));

AM1 = zeros(length(RHO), length(NSUBSPACES), ntrials, 8);
AM2 = zeros(length(RHO), length(NSUBSPACES), ntrials, 8);
RTR2 = zeros(length(RHO), length(NSUBSPACES), ntrials, 8);
VMC = zeros(length(RHO), length(NSUBSPACES), ntrials, 2);

for i = 1: length(RHO)
    rho = RHO(i)
    for j = 1:length(NSUBSPACES)
        n_subspace = NSUBSPACES(j) 
        for k = 1:ntrials
            
            [Xtrue,s] = generate_uos(n,dim_subspace,n_subspace,pts_per_sub);
            [mask, samples] = generate_mask(Xtrue,rho);
            r = rank(kernel_uos(Xtrue,2));
            X = zeros(n,s);
            X(mask) = samples;
            
            solver= 'myArmijo';
            [X1,stats1] = solve_am_mc(mask,samples,r,X,Xtrue,epsilon,solver,decrease_type,epsilon_x,theta);
            AM1(i,j,k,:) = [stats1.errors(end)  stats1.iter(end)  stats1.times(end) stats1.costcalls(end) stats1.gradcalls(end) stats1.hesscalls(end) stats1.grad(end) stats1.errors(end)<=1e-3];
            
            solver= 'TRhess';
            [X6,stats6] = solve_am_mc(mask,samples,r,X,Xtrue,epsilon,solver,decrease_type,epsilon_x,theta);
            AM2(i,j,k,:) =  [stats1.errors(end)  stats1.iter(end)  stats1.times(end) stats1.costcalls(end) stats1.gradcalls(end) stats1.hesscalls(end) stats1.grad(end) stats1.errors(end)<=1e-3];
            
          
            solver = 'TR';
            [f_end8,X8,error_end8,grad_end8,k8,stats8] = solve_grass_mc(mask,samples,n,s,r,X,Xtrue,epsilon,solver);
            RTR2(i,j,k,:) = [error_end8  stats8(end).iter  stats8(end).time stats8(end).costcalls stats8(end).gradcalls stats8(end).hesscalls stats8(end).grad stats1.errors(end)<=1e-3];
            
            
            [X4,cost4,update4,error4] = vmc(X,mask,samples,options,Xtrue);
            VMC(i,j,k,:) = [error4 errors4<=1e-3];
            
        end

    end
    
end

AM1_phase = zeros(length(RHO),length(NSUBSPACES));
AM2_phase = zeros(length(RHO),length(NSUBSPACES));
% RTR1_phase = zeros(length(RHO),length(NSUBSPACES));
RTR2_phase = zeros(length(RHO),length(NSUBSPACES));
VMC_phase = zeros(length(RHO),length(NSUBSPACES));
save('results_nsub_mc.mat','RTR','AMgrad','AMtr','VMC');
for i = 1: length(RHO)
    for j = 1:length(NSUBSPACES)
        for k = 1:ntrials
            AM1_phase(i,j) = AM1(i,j,k,8)/ntrials;
            AM2_phase(i,j) = AM2(i,j,k,8)/ntrials;
            RTR2_phase(i,j) = RTR2(i,j,k,8)/ntrials;  
        end 
    end 
end

I = mat2gray(RTR2_phase,[0 100]); 
colormap(gray)
figure(1)
imagesc(I)
xlabel('Number of points per subspaces');ylabel('Sampling rate');
set(gca,'xtick', 1:length(NSUBSPACES));
set(gca,'xticklabel',NSUBSPACES);
set(gca,'ytick', 1:length(RHO));
set(gca,'yticklabel',RHO);
saveas(gcf,'uos_phase_transition_test0','epsc')
    
    
