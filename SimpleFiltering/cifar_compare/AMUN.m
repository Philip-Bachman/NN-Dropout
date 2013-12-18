function [ X ] = AMUN( X )
% Set each row in X to any mean and unit norm.
X = bsxfun(@rdivide, X, sqrt(sum(X.^2,2) + 1e-8));
return
end

