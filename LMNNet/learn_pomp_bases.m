function [ A ] = learn_pomp_bases( X, basis_count, omp_num, round_count, train_count, Ai )
% Learn linear bases for the given patches, using "non-negative" orthogonal
% matching pursuit. That is, OMP with strictly non-negative coefficients.
% 
% Parameters:
%   X: observations to use in basis learning (obs_count x obs_dim)
%   basis_count: number of bases to learn
%   omp_num: number of bases with which to represent each observation
%   round_count: number of update rounds to perform
%   train_count: number of examples to include in each update round
%   Ai: optional initial set of bases (obs_dim x basis_count)
% Outputs:
%   A: learned set of bases (obs_dim x basis_count)
%
obs_dim = size(X,2);
if exist('Ai','var')
    if (size(Ai,1) ~= obs_dim || size(Ai,2) ~= basis_count)
        error('learn_omp_bases: mismatched initial basis size.\n');
    end
    A = Ai;
else
    A = randn(obs_dim,basis_count);
    A = bsxfun(@rdivide, A, sqrt(sum(A.^2) + 1e-8));
end

% Do round_count alternations between OMP encoding and basis updating
for r=1:round_count,
    fprintf('ROUND %d\n',r);
    if (train_count < size(X,1))
        tr_idx = randsample(size(X,1),train_count);
        Xtr = X(tr_idx,:);
        B = pomp_encode(Xtr, A, omp_num);
        [ A obj ] = omp_basis_update(Xtr, A, B);
    else
        B = pomp_encode(X, A, omp_num);
        [ A obj ] = omp_basis_update(X, A, B);
    end
end

return
end

function [ B ] = pomp_encode( X, A, omp_num )
% Use non-negative orthogonal matching pursuit to encode the observations in X
% using the bases in A.
%
% Parameters:
%   X: observations to encode (obs_count x obs_dim)
%   A: bases with which to encode observations (obs_dim x basis_count)
%   omp_num: the number of bases to include in each reconstruction
% Outputs:
%   B: encoding of observations in terms of bases (obs_count x basis_count)
%
obs_var = sum(sum((bsxfun(@minus,X,mean(X,2))).^2));
obs_count = size(X,1);
basis_count = size(A,2);
A_sqs = sum(A.^2);
A_norms = sqrt(A_sqs);
B = zeros(obs_count, basis_count);
B_idx = zeros(obs_count, omp_num);
fprintf('  OMP encoding {\n');
for i=1:omp_num,
    fprintf('    B%d:', i);
    X_norms = sqrt(sum(X.^2,2));
    scores = (X * A);
    scores = bsxfun(@rdivide, scores, X_norms);
    scores = bsxfun(@rdivide, scores, A_norms);
    [max_scores max_idx] = max(scores,[],2);
    for j=1:obs_count,
        if (mod(j,round(obs_count / 50)) == 0)
            fprintf('.');
        end
        idx = max_idx(j);
        B_idx(j,i) = idx;
        w = (X(j,:) * A(:,idx)) / A_sqs(idx);
        X(j,:) = X(j,:) - (A(:,idx)' .* w);
        B(j,idx) = B(j,idx) + w;
    end
    fprintf('\n');
end
obj = sum(X(:).^2) / obs_var;
fprintf('    obj: %.6f\n', obj);
fprintf('  }\n');

return
end

function [ A_new obj ] = omp_basis_update( X, A, B )
% Update the bases in A, given the encoding of the observations in X according
% to the weightts in B. Use gradient descent step size "step".
%
% Parameters:
%   X: observations that were encoded (obs_count x obs_dim)
%   A: bases used in the encoding (obs_dim x basis_count)
%   B: encoding weights (obs_count x basis_count)
%
% Outputs:
%   A: updated bases
%   obj: average squared error of the reconstructions
%

p = 0.8;

fprintf('  OMP updating {\n');

obs_var = sum(sum((bsxfun(@minus,X,mean(X,2))).^2));

Xh = B * A';
obj = sum((X(:) - Xh(:)).^2) / obs_var;
fprintf('    pre_obj: %.6f\n', obj);

A_new = (B +(0.01 * randn(size(B)))) \ X;
A_new = A_new';
A_new = bsxfun(@rdivide,A_new,sqrt(sum(A_new.^2) + 1e-8));
A_new = (p * A) + ((1 - p) * A_new);
A_new = bsxfun(@rdivide,A_new,sqrt(sum(A_new.^2) + 1e-8));
Xh = B * A_new';
best_obj = sum((X(:) - Xh(:)).^2) / obs_var;

fprintf('    post_obj: %.6f\n', best_obj);
fprintf('  }\n');

return
end


