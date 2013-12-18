function [ X ] = ZMUN( X )
% Set each row in X to zero mean and unit norm.

X = bsxfun(@minus, X, mean(X,2));
X = bsxfun(@rdivide, X, sqrt(sum(X.^2,2) + 1e-8));

return

end

