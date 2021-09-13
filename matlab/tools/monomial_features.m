function [phi,N] = monomial_features(X,d)

[n,s] = size(X);
N = nchoosek(n+d,n);
phi = zeros(N,s);


% compute the monomial exponents
powers = zeros(N,n);
powers(1,:) = [zeros(1,n-1) 0];
for i = 2:N
%     powers(i,:) = mono_between_next_grlex( n, 0, d, powers(i-1,:) );
    powers(i,:) = mono_between_next_grevlex( n, 0, d, powers(i-1,:) );
end

% powers
for j = 1:s
    phi(:,j) = prod(flip(X(:,j)').^powers,2);
end

end