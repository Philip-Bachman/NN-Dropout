function [ Xh A B ] = factor_huber_alt( X, r, s, Ai, Bi )
% Compute a factorization of the matrix X as X = A*B, in which A and B are both
% rank r matrices. Use Huberized squared loss, setting the l2-l1 transition so
% that numel(X)*s entries incur l1, rather than l2 loss. Use alternating
% descent on A and B.
%
% Inputs:
%   X: matrix to factorize (obs_count x obs_dim)
%   r: rank of the factorization
%   s: estimate of the fraction of outlier entries
%   Ai: initializer for A
%   Bi: initializer for B
% Outputs:
%   Xh: learned approximation Xh = A*B
%   A: factor matrix of dimension (obs_count x r)
%   B: factor matrix of dimension (obs_dim x r)
%

obs_count = size(X,1);
obs_dim = size(X,2);
s_count = ceil(s * obs_count * obs_dim);
if ~exist('Ai','var')
    A = randn(obs_count,r);
else
    A = Ai;
end
if ~exist('Bi','var')
    B = randn(r,obs_dim);
else
    B = Bi;
end
if (size(A,2) ~= r || size(B,1) ~= r)
    error('factor_huber(): initializing matrices should agree with r.\n');
end

options = struct();
options.Display = 'off';
options.Method = 'lbfgs';
options.Corr = 5;
options.LS = 3;
options.LS_init = 3;
options.MaxIter = 5;
options.MaxFunEvals = 50;
options.TolX = 1e-8;

for i=1:20,
    % Check error of reconstruction and set l2->l2 transition threshold
    Xh = A * B;
    Rsq = (Xh - X).^2;
    Rsq = sort(Rsq(:),'descend');
    d = sqrt(Rsq(s_count));
    % Optimize A while holding B fixed
    funObj = @( w ) objfun_A(w, X, B, d);
    A = minFunc(funObj, A(:), options);
    A = reshape(A, obs_count, r);
    % Check error of reconstruction and set l2->l2 transition threshold
    Xh = A * B;
    Rsq = (Xh - X).^2;
    Rsq = sort(Rsq(:),'descend');
    d = sqrt(Rsq(s_count));
    % Optimize B while holding A fixed
    funObj = @( w ) objfun_B(w, X, A, d);
    [B fval flag output] = minFunc(funObj, B(:), options);
    B = reshape(B, r, obs_dim);
    fprintf('Iter %d, fval=%.8f\n',i,fval);
    if (output.iterations == 1)
        break;
    end
end

Xh = A*B;

return

end

function [ L dLdW ] = objfun_A( w, X, B, d )
% Compute the loss and gradients for approximating the matrix X as a product of
% two rank r matrices, whose elements are stored (linearly) in w, using robust
% Huberized squared loss.
%
% Parameters:
%   w: linearized encoding of the matrix A in X = A*B
%   X: the observation matrix being "factorized"
%   B: the matrix B in X = A*B
%   d: the threshold at which to transition from l2 to l1 loss
% Outputs:
%   L: the loss, measured per element of X
%   dLdW: partial gradients of loss with respect to elements of w
%
p_count = numel(X);
r = size(B,1);
A = reshape(w, size(X,1), r);
X_hat = A*B;
R = X_hat - X;
Rsq = R.^2;
% Create a mask of entries of X_res that incur l2 loss
loss_mask = Rsq <= d^2;
L = Rsq;
L(~loss_mask) = (2 * d * abs(R(~loss_mask))) - d^2;
L = sum(sum(L)) / p_count;
if (nargout > 1)
    dLdR = zeros(size(X));
    dLdR(loss_mask) = 2 * R(loss_mask);
    dLdR(~loss_mask) = 2 * d * sign(R(~loss_mask));
    dLdA = (2 / p_count) * dLdR * B';
    dLdW = dLdA(:);
end
return
end

function [ L dLdW ] = objfun_B( w, X, A, d )
% Compute the loss and gradients for approximating the matrix X as a product of
% two rank r matrices, whose elements are stored (linearly) in w, using robust
% Huberized squared loss.
%
% Parameters:
%   w: linearized encoding of the matrix B in X = A*B
%   X: the observation matrix being "factorized"
%   A: the matrix A in X = A*B
%   d: the threshold at which to transition from l2 to l1 loss (optional)
% Outputs:
%   L: the loss, measured per element of X
%   dLdW: partial gradients of loss with respect to elements of w
%
p_count = numel(X);
r = size(A,2);
B = reshape(w, r, size(X,2));
X_hat = A*B;
R = X_hat - X;
Rsq = R.^2;
% Create a mask of entries of X_res that incur l2 loss
loss_mask = Rsq <= d^2;
L = Rsq;
L(~loss_mask) = (2 * d * abs(R(~loss_mask))) - d^2;
L = sum(sum(L)) / p_count;
if (nargout > 1)
    dLdR = zeros(size(X));
    dLdR(loss_mask) = 2 * R(loss_mask);
    dLdR(~loss_mask) = 2 * d * sign(R(~loss_mask));
    dLdB = (2 / p_count) * A' * dLdR;
    dLdW = dLdB(:);
end
return
end