function [ W ] = compute_whitener( X, lam )
% Compute a ZCA whitening transform for the observations in the rows of X.

if ~exist('lam','var')
    lam = 1e-6;
end

H = cov(X);
[U S V] = svd(H);

D = diag(1 ./ sqrt(diag(S)+lam));
W = U * D * U';

return

end