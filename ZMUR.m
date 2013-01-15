function [ X ] = ZMUR( X )
% Set each column in X to zero mean and unit range.

X = bsxfun(@minus, X, mean(X));
X = bsxfun(@rdivide, X, (max(abs(X)) + 1e-8));

return

end

