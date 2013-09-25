clear;

% Load data to run through network
load('satimage.mat');
Ytr_c = class_cats(Ytr); Yte_c = class_cats(Yte);

% Create the ShepNet instance
layer_dims = [size(Xtr,2) 25 25 size(Xtr,2)];
NET = ShepNet(Xtr, Ytr, layer_dims, ActFunc(5), ActFunc(1));
NET.init_weights(0.1);
NET.weight_noise = 0.01;
NET.drop_rate = 0.1;
NET.drop_input = 0.0;

% Setup param struct for training the net
params = struct();
params.rounds = 5000;
params.start_rate = 0.01;
params.decay_rate = 0.1^(1 / params.rounds);
params.momentum = 0.8;
params.batch_size = 200;
% Regularization parameters
params.lam_l2 = 1e-5;
% Validation stuff
params.do_validate = 1;
params.Xv = Xte;
params.Yv = Yte;