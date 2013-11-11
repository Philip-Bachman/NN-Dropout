function [ NET result params ] = train_smoothnet( Xtr, Ytr, Xte, Yte )
% Accessory function for training a smoothnet instance. This sets up the
% parameters required for training, based on the given train/test data.

Ytr_i = class_inds(Ytr);
Yte_i = class_inds(Yte);

% Create the SmoothNet instance
layer_dims = [size(Xtr,2) 250 250 250 size(Ytr_i,2)];
NET = SmoothNet(Xtr, Ytr_i, layer_dims, ActFunc(6), ActFunc(1));
NET.out_loss = @(yh, y) SmoothNet.loss_mcl2h(yh, y);
NET.init_weights(0.1);

% Set whole-network regularization parameters
NET.weight_noise = 0.00;
NET.drop_rate = 0.10;
NET.drop_input = 0.00;

% Set per-layer regularization parameters
for i=1:numel(layer_dims),
    NET.layer_lams(i).lam_l1 = 0;
    NET.layer_lams(i).lam_l2 = 0;
    NET.layer_lams(i).wt_bnd = 6;
end
orders = 1:4;
sigma = 1.0;
ord_lams = sigma.^(2*orders) ./ factorial(orders);
NET.layer_lams(numel(layer_dims)).ord_lams = 0.0 * ord_lams;

% Setup param struct for training the net
params = struct();
params.rounds = 25000;
params.start_rate = 0.1;
params.decay_rate = 0.1^(1 / params.rounds);
params.momentum = 0.98;
params.batch_size = 50;
% Regularization parameters
params.lam_l2 = 1e-6;
% Validation stuff
params.do_validate = 1;
params.Xv = Xte;
params.Yv = Yte_i;

% Train the network on training data using the given parameters
result = NET.train(Xtr,Ytr_i,params);


end

