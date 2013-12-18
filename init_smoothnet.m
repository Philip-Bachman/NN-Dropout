function [ NET params ] = init_smoothnet( Xtr, Ytr, Xte, Yte )
% Accessory function for training a smoothnet instance. This sets up the
% parameters required for training, based on the given train/test data.

Ytr_i = SmoothNet.to_inds(Ytr);
Yte_i = SmoothNet.to_inds(Yte);

% Create the SmoothNet instance
layer_dims = [size(Xtr,2) 250 250 size(Ytr_i,2)];
NET = SmoothNet(layer_dims, ActFunc(6), ActFunc(1));
NET.init_weights(0.2);
NET.out_loss = @SmoothNet.loss_mcl2h;

% Set dropout-related parameters
NET.drop_input = 0.0;
NET.drop_hidden = 0.0;
NET.weight_noise = 0;

% Set per-layer regularization parameters
for i=1:numel(layer_dims),
    NET.layer_lams(i).l2_bnd = 5;
    NET.layer_lams(i).ord_lams = [0.1 0.1];
end

% % Set network-wide weight regularization parameters
% NET.lam_l2 = 1e-5;
% NET.lam_l1 = 0;

% Setup param struct for training the net
params = struct();
params.rounds = 50000;
params.start_rate = 0.1;
params.decay_rate = 0.1^(1 / params.rounds);
params.momentum = 0.9;
params.batch_size = 50;
% Validation stuff
params.do_validate = 1;
params.Xv = Xte;
params.Yv = Yte_i;

% Train the network on training data using the given parameters
%NET.train(Xtr,Ytr_i,params);

end

