clear;

% Make a simple dataset for testing least-squares regression with LDNet
n = 2000;
%rng(4) % seed random number generator
r = RandStream('mrg32k3a','Seed',4);
RandStream.setDefaultStream(r);

func = @( x ) (x(:,1) - (0.5*x(:,1).^2) + 0.5*sin(3*x(:,2)));
noise = @( y, scale ) (y + scale*(2*rand(size(y)) - 1));

X = 2 * (rand(n,2) - 0.5);
Y = noise(func(X), 0.1);
[Xtr Ytr Xte Yte] = trte_split(X,Y,0.75);

% Setup SGD-related training options
opts = struct();
opts.do_draw = 1; % plot per-round loss on training batches
opts.lam_l2 = 0; % standard L2 weight decay, often worse than L2 constraint
opts.rounds = 15000;
opts.batch_size = 100;
opts.dev_reps = 1; % experimental, only matters if LDN.do_dev ~= 1
opts.do_validate = 1; % whether to check loss on validation sample
opts.Xv = Xte; % validation inputs
opts.Yv = Yte; % validation outputs

%%%%%%%%%%%%
% Init a network, given a list of layer sizes, a hidden layer activation and
% a loss function to optimize (set loss to loss_lsq for regression).
layer_sizes = [size(Xtr,2) 50 50 50 size(Ytr,2)]; 
LDN = LDNet(layer_sizes, @LDLayer.norm_relu_trans, @LDNet.loss_lsq);
LDN.lam_l2a = zeros(size(layer_sizes)); % L2 regularization on activations
LDN.lam_l2 = 1e-4; % Standard L2 regularization on network weights
LDN.wt_bnd = 100; % L2 constraint on weights into each hidden node
LDN.drop_input = 0; % drop rate at input layer
LDN.drop_hidden = 0; % drop rate at hidden layers
LDN.drop_undrop = 0; % rate to pass samples drop-free
LDN.bias_noise = 0; % noise to perturb activations during training
LDN.weight_noise = 0.01; % noise to perturb weights during training
%%%%%%%%%%%%
% These are experimental regularizer params, probably best left at 0
LDN.dev_lams = zeros(size(layer_sizes));
LDN.dev_types = ones(size(layer_sizes));
LDN.dev_types(end) = 2;
LDN.do_dev = 0;
%%%%%%%%%%%%

% Initialize network weights and biases
LDN.init_weights(0.05, 0.05);

% Set core learning params and train a bit
opts.momentum = 0.9;
opts.decay_rate = 0.2^(1 / opts.rounds);
opts.start_rate = 0.05;
LDN.train(Xtr,Ytr,opts);

% Plot the function learned by LDN
plot_netfunc(X, Y, LDN, 50, 0.25);


