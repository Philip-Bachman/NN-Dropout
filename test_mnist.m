%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load and "preprocess" MNIST digit data %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('mnist_data.mat');
idx = (Y_mnist == 3) | (Y_mnist == 6);
Xi = X_mnist(idx,:);
Yi = Y_mnist(idx,:);
classes = unique(Yi);
X = ZMUV(double(Xi));
Y = zeros(size(Yi,1),numel(classes));
for c=1:numel(classes),
    c_label = classes(c);
    c_idx = (Yi == c_label);
    Y(c_idx,c) = 1;
    Y(~c_idx,c) = -1;
end

obs_dim = size(X,2);
out_dim = size(Y,2);
obs_count = size(X,1);
train_count = round(obs_count * 0.8);

% Split data into training and testing portions
tr_idx = randsample(obs_count,train_count,false);
te_idx = setdiff(1:obs_count,tr_idx);
X_tr = X(tr_idx,:);
X_te = X(te_idx,:);
Y_tr = Y(tr_idx,:);
Y_te = Y(te_idx,:);

clear Xi X Y;

% Use sigmoid activation in hidden layers, linear activation in output layer,
% and binomial-deviance loss at output layer.
act_func = ActFunc(5);
out_func = ActFunc(1);
loss_func = LossFunc(3);
% Create the network object
net = BlockNet(act_func, out_func, loss_func);
% Setup blocky parameters and initialize network
layer_bsizes = [1 5 5 1];
layer_bcounts = [obs_dim 100 100 out_dim];
net.init_blocks(layer_bsizes, layer_bcounts, 0.05);
net.do_drop = 1;

% Set up parameter struct for updates
params = struct();
params.epochs = 10000;
params.start_rate = 0.5;
params.decay_rate = 0.1^(1 / params.epochs);
params.momentum = 0.5;
params.weight_bound = 50;
params.batch_size = 250;
params.batch_rounds = 1;
params.dr_obs = 0.1;
params.do_validate = 1;
params.X_v = X_te;
params.Y_v = Y_te;





%%%%%%%%%%%%%%
% EYE BUFFER %
%%%%%%%%%%%%%%
