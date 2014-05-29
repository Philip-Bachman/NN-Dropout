clear;

% Load MNIST digits data
load('mnist_data.mat');
X = double(X_mnist) ./ max(max(double(X_mnist)));
Y = LDNet.class_inds(double(Y_mnist));
[Xtr Ytr Xte Yte] = trte_split(X,Y,0.8);

% Setup SGD-related training options
opts = struct();
opts.do_draw = 1; % draw first layer filters, set to 0 for not MNIST/USPS data
opts.lam_l2 = 0; % standard L2 weight decay, often worse than L2 constraint
opts.rounds = 20000;
opts.batch_size = 100;
opts.dev_reps = 2; % experimental, only matters if LDN.do_dev ~= 1
opts.do_validate = 1; % whether to check loss on validation sample
opts.Xv = Xte; % validation inputs
opts.Yv = Yte; % validation outputs

%%%%%%%%%%%%
% Init a network, given a list of layer sizes, a hidden layer activation and
% a loss function to optimize (set loss to loss_lsq for regression.
LDN = LDNet([size(Xtr,2) 200 200 size(Ytr,2)], ...
    @LDLayer.relu_trans, @LDNet.loss_lsq);
LDN.lam_l2a = [0.0 0.0 0.0]; % L2 regularization on per-layer activations
LDN.wt_bnd = 3.0; % L2 constraint on weights into each hidden node
LDN.drop_input = 0.2; % drop rate at input layer
LDN.drop_hidden = 0.5; % drop rate at hidden layers
LDN.drop_undrop = 0.0; % rate to pass samples drop-free
%%%%%%%%%%%%
% These are experimental regularizer params, probably best left at 0
LDN.dev_lams = [0.0 0.0 0.0];
LDN.dev_types = [1 1 2];
LDN.do_dev = 0;
%%%%%%%%%%%%

% Initialize network weights and biases
LDN.init_weights(0.05,0.1);

% Set core learning params and train a bit
opts.momentum = 0.5;
opts.decay_rate = 0.5^(1 / opts.rounds);
opts.start_rate = 0.1;
LDN.train(Xtr,Ytr,opts);

% Set core learning params and train a bit
opts.momentum = 0.75;
opts.decay_rate = 0.5^(1 / opts.rounds);
opts.start_rate = 0.05;
LDN.train(Xtr,Ytr,opts);

% Set core learning params and train a bit
opts.momentum = 0.9;
opts.decay_rate = 0.5^(1 / opts.rounds);
opts.start_rate = 0.02;
LDN.train(Xtr,Ytr,opts);

% Set core learning params and train a bit
opts.momentum = 0.95;
opts.decay_rate = 0.1^(1 / opts.rounds);
opts.start_rate = 0.01;
LDN.train(Xtr,Ytr,opts);


