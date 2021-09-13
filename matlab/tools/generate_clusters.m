function [X,s] = generate_clusters(n,np,nc,radius,Amplitude)
%Function to generate random clusters in R^n.
% [X,s] = generate_clusters(n,np,nc,radius,Amplitude)
% Input:
%     n: ambien dimension
%     np: number of point per cluster
%     nc: number of clusters
% radius is the standard deviation of each cluster
if(nargin == 4)
    Amplitude = 1;
end
centers = Amplitude*randn(n,nc);
X = [];
for k=1:nc
    X = [X, radius*randn(n,np) + centers(:,k)*ones(1,np)];
end
s = np*nc;
end