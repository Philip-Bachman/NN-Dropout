function [ A b_enc b_dec ] = learn_ksparse_bases(...
    X, basis_count, k_sparse, batch_size, round_count, step )
% Learn linear bases for the given patches, using orthogonal matching pursuit.
%
% Parameters:
%   X: observations to use in basis learning (obs_count x obs_dim)
%   basis_count: number of bases to learn
%   k_sparse: number of bases with which to represent each observation
%   batch_size: size of batches to use for each dictionary update round
%   round_count: number of update rounds to perform
%   step: step size for basis gradient descent
% Outputs:
%   A: learned set of (tied) encoder/decoder weights (obs_dim x basis_count)
%   b_enc: learned set of encoder biases
%   b_dec: learned set of decoder biases
%

drop_rate = 0.5;
obs_dim = size(X,2);

% Generate a set of random initial bases and encoder/decoder biases
A = initial_bases(X, basis_count, k_sparse, drop_rate);
b_enc = zeros(1, basis_count);
b_dec = zeros(1, obs_dim);

% Setup a structure of options for controlling updates
up_opts = struct();
up_opts.step = step;
up_opts.momentum = 0.9;
up_opts.lam_l1 = 0.0;
up_opts.mom_A = zeros(size(A));
up_opts.mom_b_enc = zeros(size(b_enc));
up_opts.mom_b_dec = zeros(size(b_dec));
up_opts.do_out = 0;

fig = figure();
% Do round_count alternations between encoding and basis updating
k_now = min(basis_count, 5*k_sparse);
for r=1:round_count,
    if ((r == 1) || (mod(r, 250) == 0))
        fprintf('ROUND %4d -- ',r);
        up_opts.do_out = 1;
        up_opts.step = up_opts.step * 0.985;
        if ((k_now > k_sparse) && (r > 1000))
            k_now = k_now - 10;
            k_now = max(k_now, k_sparse);
        end
        %check_loss_grad(X, A, b_enc, b_dec, k_sparse, batch_size);
        draw_usps_filters(A',64,0,1,fig);
        drawnow();
    else
        up_opts.do_out = 0;
    end
    Xb = X(randsample(size(X,1),batch_size),:);
    B = ksparse_encode(Xb, A, b_enc, k_now, drop_rate);
    [A b_enc b_dec up_opts] = ...
        parameter_update(Xb, A, b_enc, b_dec, B, up_opts);
    b_enc = zeros(1, basis_count);
    b_dec = zeros(1, obs_dim);
end

return
end

function [ B B_mask ] = ksparse_encode(X, A, b_enc, k_sparse, drop_rate, B_mask)
% Use k-sparse activation function to encode the observations in X using the
% bases in A, with the activation biases b_enc. Apply dropout with drop
% probability "drop_rate".
%
% Parameters:
%   X: observations to encode (obs_count x obs_dim)
%   A: bases with which to encode observations (obs_dim x basis_count)
%   b_enc: biases for each encoder basis (1 x basis_count)
%   k_sparse: the number of bases to include in each reconstruction
%   drop_rate: rate at which to randomly drop bases during encoding
%   B_mask: activation mask, will be computed for k-sparseness if not given
% Outputs:
%   B: encoding of observations in terms of bases (obs_count x basis_count)
%   B_mask: k-sparse mask used for encoding (obs_count x basis_count)
%
if ~exist('drop_rate','var')
    drop_rate = 0;
end
B = bsxfun(@plus, (X * A), b_enc);
B_abs = abs(B);
B_abs = B_abs .* (rand(size(B_abs)) > drop_rate);
B_abs_sorted = sort(B_abs,2,'descend');
if ~exist('B_mask','var')
    B_mask = bsxfun(@gt, B_abs, (B_abs_sorted(:,k_sparse) - 1e-8));
end
B = B .* B_mask;
return
end

function [ A b_enc b_dec opts ] = parameter_update(X, A, b_enc, b_dec, B, opts)
% Update the bases in A, given the encoding of the observations in X
% according to the weights in B. Use gradient descent step size "step".
%
% Parameters:
%   X: observations that were encoded (obs_count x obs_dim)
%   A: bases used in the encoding (obs_dim x basis_count)
%   b_enc: biases used in encoder step (1 x basis_count)
%   b_dec: biases used in decoder step (1 x obs_dim)
%   B: coefficients of k-sparse encoding (obs_count x basis_count)
%   opts: options structure containing the following...
%     step: gradient descent step size (i.e. learning rate)
%     momentum: momentum parameter for descent step
%     lam_l1: regularization weight for l1 penalty on normalized activations
%     mom_A: stored momentum of A
%     mom_b_enc: stored momentum of b_enc
%     mom_b_dec: stored momentum of b_dec
%     do_out: whether to display progress info for this update
% Outputs:
%   A: updated dictionary of filters
%   b_enc: updated biases for encoder
%   b_dec: updated biases for decoder
%   opts: options used for update, with momentums updated as appropriate
%
if ~exist('opts','var')
    opts = struct();
end
if ~isfield(opts,'step')
    opts.step = 0.01;
end
if ~isfield(opts,'momentum')
    opts.momentum = 0.8;
end
if ~isfield(opts,'lam_l1')
    opts.lam_l1 = 0;
end
if ~isfield(opts,'mom_A')
    opts.mom_A = zeros(size(A));
end
if ~isfield(opts,'mom_b_enc')
    opts.mom_b_enc = zeros(size(b_enc));
end
if ~isfield(opts,'mom_b_dec')
    opts.mom_b_dec = zeros(size(b_dec));
end
if ~isfield(opts,'do_out')
    opts.do_out = 0;
end

obs_count = size(X,1);

% Compute stuff for L1 regularization of unit-normalized activations
[Bn BP_Bn] = norm_transform(B);
L_l1 = (opts.lam_l1 * sum(abs(Bn(:)))) / obs_count;
dL1 = opts.lam_l1 * (X' * BP_Bn(sign(Bn)));

% Compute the combined encoder/decoder approximation and its residual
Xh = bsxfun(@plus, (B * A'), b_dec);
Xr = Xh - X;

% Compute gradients for each set of parameters
B_mask = (abs(B) > 1e-8);
Ad_grad = Xr' * B;
Ae_grad = X' * ((Xr * A) .* B_mask);
A_grad = (Ad_grad + Ae_grad + dL1) ./ obs_count;
b_enc_grad = sum((Xr * A) .* B_mask) ./ obs_count;
b_dec_grad = sum(Xr,1) ./ obs_count;

% Compute updated momentum for each set of parameters
mo = opts.momentum;
opts.mom_A = (mo * opts.mom_A) + ((1 - mo) * A_grad);
opts.mom_b_enc = (mo * opts.mom_b_enc) + ((1 - mo) * b_enc_grad);
opts.mom_b_dec = (mo * opts.mom_b_dec) + ((1 - mo) * b_dec_grad);

% Apply updates to each set of parameters
A = A - (opts.mom_A .* opts.step);
b_enc = b_enc - (opts.mom_b_enc .* opts.step);
b_dec = b_dec - (opts.mom_b_dec .* opts.step);

% Display pre and post-update losses, if so desired
if (opts.do_out == 1)
    % Compute pre-update loss
    pre_obj = sum(Xr(:).^2) / obs_count;
    % Compute post-update loss
    B_mask = (abs(B) > 1e-8);
    B = bsxfun(@plus, (X * A), b_enc);
    B = B .* B_mask;
    Xh = bsxfun(@plus, (B * A'), b_dec);
    post_obj = sum((X(:) - Xh(:)).^2) / obs_count;
    fprintf('pre_rec: %.4f, post_rec: %.4f, l1: %.4f\n', ...
        pre_obj, post_obj, L_l1);
end
return
end

function [ A_init ] = initial_bases(X, basis_count, k_sparse, drop_rate)
% Generate reasonably-scaled random bases, using observations in X. Bases are
% generated by (1) sampling their weights from a standard normal distribution
% and then (2) rescaling the bases such that the reconstructions they induce
% subject to a given k-sparsity level have the same variance as the inputs.
%
obs_count = size(X,1);
obs_dim = size(X,2);
% Start with unit-normed directions, selected uniformly at random.
A = randn(obs_dim, basis_count);
%A = bsxfun(@rdivide, A, sqrt(sum(A.^2,1) + 1e-8));
b_enc = zeros(1, basis_count);
X_scale = mean(std(X,false,2));
b_size = 500;
b_count = floor(obs_count / b_size);
b_end = 0;
Xh_scale = 0;
for b=1:b_count,
    b_start = b_end + 1;
    b_end = b_start + (b_size - 1);
    Xb = X(b_start:b_end,:);
    B = ksparse_encode(Xb, A, b_enc, k_sparse, drop_rate);
    Xh = B * A';
    Xh_scale = Xh_scale + mean(std(Xh,false,2));
end
Xh_scale = Xh_scale / b_count;
A_init = A * (X_scale / Xh_scale);
return
end

function [ F BP ] = norm_transform(X)
% L2 normalize X by rows, and return both the row-normalized matrix and a
% function handle for backpropagating through the normalization.
%
% Parameters:
%   X: matrix to which to apply row-wise unit-normalization
% Outputs:
%   F: row-wise unit-normalized version of X
%   BP: function handle for transforming grads on F into grads on X.
%
N = sqrt(sum(X.^2,2) + 1e-8);
F = bsxfun(@rdivide,X,N);
% Backpropagate through unit-normalization
BP = @( D ) ...
    (bsxfun(@rdivide,D,N) - bsxfun(@times,F,(sum(D.*X,2)./(N.^2))));
return
end

function [ res ] = check_loss_grad( X, A, b_enc, b_dec, k_sparse, batch_size)
% Check loss and gradient computations using minFunc. This applies the fast,
% approximate numerical gradient checking from minFunc to the loss and gradient
% that are implemented in the parameter_update() function. 
%
%
obs_count = size(X,1);
for i=1:10,
    Xb = X(randsample(obs_count,batch_size),:);
    [B B_mask] = ksparse_encode(Xb, A, b_enc, k_sparse, 0.0);
    w = [A(:); b_enc(:); b_dec(:)];
    mf_func = @( w ) compute_loss_grad(w, Xb, B_mask, A, b_enc, b_dec);
    fastDerivativeCheck(mf_func,w,1,2);
end
res = 0;
return
end

function [ L dLdw ] = compute_loss_grad(w, X, B_mask, A, b_enc, b_dec)
% Loss and gradient computations for use in minFunc numerical grad check. These
% gradients are the same as in parameter_update() function, but wrapped for use
% by minFunc's fast numerical gradient checking.
%
obs_count = size(X,1);
% Unpack the parameter vector w.
A_num = numel(A);
be_num = numel(b_enc);
A = reshape(w(1:A_num), size(A,1), size(A,2));
b_enc = reshape(w((A_num+1):(A_num+be_num)), size(b_enc,1), size(b_enc,2));
b_dec = reshape(w((A_num+be_num+1):end), size(b_dec,1), size(b_dec,2));

% Compute encoding coefficients, reconstruction residual, and loss
B = bsxfun(@plus, (X * A), b_enc) .* B_mask;
Xr = bsxfun(@plus, (B * A'), b_dec) - X;
L = (0.5 * sum(Xr(:).^2)) / obs_count;

% Compute gradients for each set of parameters
Ad_grad = Xr' * B;
Ae_grad = X' * ((Xr * A) .* B_mask);
A_grad = (Ad_grad + Ae_grad) ./ obs_count;
b_enc_grad = sum((Xr * A) .* B_mask) ./ obs_count;
b_dec_grad = sum(Xr,1) ./ obs_count;

% Combine gradients into vector form, to appease minFunc
dLdw = [A_grad(:); b_enc_grad(:); b_dec_grad(:)];

return
end


