function M = truncate(M1,p)
M = zeros(size(M1));
M(M1>=p) = 1;
end