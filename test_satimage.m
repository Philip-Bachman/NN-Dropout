%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test dropout-nn classification of "satimage" data %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('satimage.mat');
Xtr = ZMUV(X_train);
Xte = ZMUV(X_test);
obs_dim = size(Xtr,2);
class_labels = sort(unique(Y_train),'ascend');
class_count = numel(class_labels);
% Create 'binarized' class matrices
Ytr = -ones(size(Y_train,1),class_count);
Yte = -ones(size(Y_test,1),class_count);
for c=1:class_count,
    c_label = class_labels(c);
    Ytr(Y_train == c_label,c) = 1;
    Yte(Y_test == c_label,c) = 1;
end

% Use sigmoid activation in hidden layers, linear activation in output layer,
% and binomial-deviance loss at output layer.
act_func = ActFunc(2);
out_func = ActFunc(1);
loss_func = LossFunc(3);
% Create the network object
net = BlockNet(act_func, out_func, loss_func);
% Setup blocky parameters
layer_bsizes = [1 2 2 1];
layer_bcounts = [obs_dim 128 128 class_count];
% Initialize network blocks
net.init_blocks(layer_bsizes, layer_bcounts, 0.05);

% Set up parameter struct for updates
params = struct();
params.epochs = 10000;
params.start_rate = 1.0;
params.decay_rate = 0.2^(1 / params.epochs);
params.momentum = 0.8;
params.weight_bound = 50;
params.batch_size = 500;
params.batch_rounds = 1;
params.dr_obs = 0.0;
params.do_validate = 1;
params.X_v = Xte;
params.Y_v = Yte;






%%%%%%%%%%%%%%
% EYE BUFFER %
%%%%%%%%%%%%%%