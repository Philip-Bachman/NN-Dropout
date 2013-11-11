clear;

% Load data to run through network
% load('mnist_pca.mat');
% tr_idx = randsample(1:size(X,1),40000,false);
% te_idx = setdiff(1:size(X,1),tr_idx);
% te_idx = randsample(te_idx,10000,false);
% Xtr = X(tr_idx,1:200);
% Xte = X(te_idx,1:200);
% Ytr = Y(tr_idx,:);
% Yte = Y(te_idx,:);
% clear X Y;

load('satimage.mat');

Ytr_c = class_cats(Ytr);
Yte_c = class_cats(Yte);


% Create the ShepNet instance
layer_dims = [size(Xtr,2) 64 64 size(Xtr,2)];
NET = LMNNet(Xtr, Ytr, layer_dims, ActFunc(5), ActFunc(1));
NET.init_weights(0.1);

% Set the output layer type (i.e. 0 => encode OR 1 => classify)
NET.out_type = 0;

% Set whole-network regularization parameters
NET.lam_out = 0.1;
NET.weight_noise = 0.01;
NET.drop_rate = 0.10;
NET.drop_input = 0.00;

% Set per-layer regularization parameters
c_layer = 4;
o_layer = numel(layer_dims);
NET.const_layer = c_layer;
NET.layer_lams(c_layer).lam_lmnn = 1.0;
NET.layer_lams(c_layer).lam_grad = 0.0;
NET.layer_lams(c_layer).lam_hess = 0.0;
%NET.layer_lams(o_layer).lam_grad = 1e-2;
%NET.layer_lams(o_layer).lam_hess = 1.0;
for i=1:numel(layer_dims),
    NET.layer_lams(i).lam_l2 = 1e-4;
end

% Setup param struct for training the net
params = struct();
params.rounds = 75000;
params.lmnn_start = 10000;
params.lmnn_count = size(Xtr,1)+1;
params.start_rate = 0.001;
params.decay_rate = 0.2^(1 / params.rounds);
params.momentum = 0.9;
params.batch_size = 150;
% Regularization parameters
params.lam_l2 = 1e-4;
% Validation stuff
%params.do_validate = 1;
%params.Xv = Xte;
%params.Yv = Yte;