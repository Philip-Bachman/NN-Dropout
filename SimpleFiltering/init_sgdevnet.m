function [ NET params ] = init_sgdevnet( Xtr, Ytr, Xte, Yte )
% Accessory function for training a smoothnet instance. This sets up the
% parameters required for training, based on the given train/test data.

Ytr_i = SGDevNet.to_inds(Ytr);
Yte_i = SGDevNet.to_inds(Yte);

% Create the SGDevNet instance
layer_dims = [size(Xtr,2) 800 800 size(Ytr_i,2)];
NET = SGDevNet(layer_dims, ActFunc(5), ActFunc(1));
NET.init_weights(0.1);

% Set dropout-related parameters
NET.drop_input = 0.2;
NET.drop_hidden = 0.5;
NET.do_dev = 0;
NET.do_sde = 1;

% Set per-layer regularization parameters
for i=1:numel(layer_dims),
    NET.layer_lams(i).l2_bnd = 5;
    NET.layer_lams(i).lam_dev = 5.0;
    NET.layer_lams(i).dev_type = 1;
end

% Set network-wide weight regularization parameters
NET.lam_l2 = 1e-5;
NET.lam_l1 = 0;

% Setup param struct for training the net
params = struct();
params.rounds = 50000;
params.start_rate = 0.25;
params.decay_rate = 0.1^(1 / params.rounds);
params.momentum = 0.9;
params.batch_size = 25;
params.dev_reps = 4;
% Validation stuff
params.do_validate = 1;
params.Xv = Xte;
params.Yv = Yte_i;

% Train the network on training data using the given parameters
%NET.train(Xtr,Ytr_i,params);

end

