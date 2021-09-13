function [X,s] = generate_2_clusters(n,np,radius,Amplitude)
nc = 2;
centers = Amplitude*[ones(n,1) -ones(n,1)];
X = []; 
for k=1:nc
    X = [X, radius*randn(n,np) + centers(:,k)*ones(1,np)];
    % radius is the standard deviation
end
s = np*nc;

end