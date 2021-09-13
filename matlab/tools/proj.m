function X = proj(X,a)
if(nargin==1)
    a = 1;
end
    X(X<-a) = -a;
    X(X>a) = a;
end