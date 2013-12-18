function [ NET result params ] = train_lmnnet( Xtr, Ytr, Xte, Yte )
% Accessory function for training a smoothnet instance. This sets up the
% parameters required for training, based on the given train/test data.

Ytr_i = class_inds(Ytr);
Yte_i = class_inds(Yte);

% Create the SmoothNet instance
layer_dims = [size(Xtr,2) 256 256 256];
NET = LMNNet(layer_dims, ActFunc(6), ActFunc(6));
NET.init_weights(0.2);

% Set whole-network regularization parameters
NET.weight_noise = 0.00;
NET.drop_input = 0.2;
NET.drop_hidden = 0.5;
NET.drop_output = 0.2;

% Set per-layer regularization parameters
for i=1:numel(layer_dims),
    NET.layer_lams(i).lam_l1 = 0;
    NET.layer_lams(i).lam_l2 = 0;
    NET.layer_lams(i).wt_bnd = 5;
end
NET.const_layer = 4;
NET.layer_lams(4).lam_lmnn = 1.0;

% Setup param struct for training the net
params = struct();
params.rounds = 50000;
params.lmnn_start = 1;
params.lmnn_count = 1000;
params.start_rate = 0.1;
params.decay_rate = 0.1^(1 / params.rounds);
params.momentum = 0.9;
params.batch_size = 100;
% Regularization parameters
params.lam_l2 = 0;
% Validation stuff
params.do_validate = 1;
params.Xv = Xte;
params.Yv = Yte_i;

% Train the network on training data using the given parameters
result = 0;
%result = NET.train(Xtr,Ytr_i,params);

end

