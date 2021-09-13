function X = my_rescale(X,beta)
% beta can be a number <=1 in order to rescale slightly inside the
% unit box
if(beta >1)
    print('Error: my_rescale, beta must be <1\n')
end
if(nargin ==1)
    beta = 1;
end
b=max(X,[],2);
a=min(X,[],2);
alpha=[b-a,b+a]/2;
alpha(alpha(:,1)==0,1)=1;
X=beta*(X-alpha(:,2))./(alpha(:,1));
end