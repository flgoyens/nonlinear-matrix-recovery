function [Ur,sv,Vr] = truncate_svd(K,r)
% returns the r singular vectors corresponding to the r leading sing values
% [Ur,sv,Vr] = svd_r(K,r)
% Ur and Vr have r orthonormal columns and sv is a vector of size r. 
[Ur,Strue,Vr] = svds(K,r);
sv = diag(Strue);
end