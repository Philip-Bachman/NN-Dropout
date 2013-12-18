function [ X ] = ZMUV( X )
% Set each column in X to zero mean and unit standard deviation.

X = bsxfun(@minus, X, mean(X));
X = bsxfun(@rdivide, X, (std(X) + 1e-8));

return

end

