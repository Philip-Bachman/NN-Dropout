%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load and "preprocess" MNIST digit data %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('mnist_data.mat');
idx = randsample(size(X_mnist,1),20000);
X = single(X_mnist(idx,:));
Y = single(Y_mnist(idx,:));
X = single(ZMUR(X));
Y_vals = unique(Y);
Ym = zeros(size(X,1),numel(Y_vals));
for i=1:size(X,1),
    for j=1:numel(Y_vals),
        if (Y(i) == Y_vals(j))
            Ym(i,j) = 1;
        else
            Ym(i,j) = -1;
        end
    end
end
Y = Ym;

obs_dim = size(X,2);
out_dim = size(X,2);
obs_count = size(X,1);
train_count = 10000;

% Split data into training and testing portions
tr_idx = randsample(obs_count,train_count,false);
te_idx = setdiff(1:obs_count,tr_idx);
X_tr = X(tr_idx,:);
X_te = X(te_idx,:);
Y_tr = X(tr_idx,:);
Y_te = X(te_idx,:);
clear('X','Y');

% Set hidden/output layer activation functions and set loss function.
act_func = ActFunc(1);
out_func = ActFunc(1);
loss_func = LossFunc(1);
% Create a BlockNet instance
net = BlockNet(act_func, out_func, loss_func);
% Setup blocky parameters
layer_bsizes = [1 1 1];
layer_bcounts = [obs_dim 32 out_dim];

% Set up parameter struct for updates
params = struct();
params.epochs = 5000;
params.start_rate = 0.01;
params.decay_rate = 0.1^(1 / params.epochs);
params.momentum = 0.5;
params.weight_bound = 10;
params.batch_size = 100;
params.batch_rounds = 1;
params.dr_obs = 0.2;
params.do_validate = 1;
params.X_v = X_te;
params.Y_v = Y_te;






%%%%%%%%%%%%%%
% EYE BUFFER %
%%%%%%%%%%%%%%