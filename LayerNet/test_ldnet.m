clear;

% Load USPS digits data
load('mnist_data.mat');
X = double(X_mnist) ./ max(max(double(X_mnist)));
Y = LDNet.class_inds(double(Y_mnist));
[Xtr Ytr Xte Yte] = trte_split(X,Y,0.8);

% Setup SGD-related training options
opts = struct();
opts.do_draw = 1;
opts.lam_l2 = 0;
opts.rounds = 10000;
opts.batch_size = 100;
opts.dev_reps = 2;
opts.do_validate = 1;
opts.Xv = Xte;
opts.Yv = Yte;

LDN = LDNet([size(Xtr,2) 800 800 size(Ytr,2)], @LDLayer.relu_trans, @LDNet.loss_mcl2h);
LDN.lam_l2a = [0.0 0.0 0.0];
LDN.dev_lams = [0.0 0.0 0.0];
LDN.dev_types = [1 1 2];
LDN.drop_input = 0.2;
LDN.drop_hidden = 0.5;
LDN.drop_undrop = 0.0;
LDN.do_dev = 0;
LDN.dev_pre = 0;
LDN.wt_bnd = 2.0;
LDN.init_weights(0.1,0.01);

% Set core learning params and train a bit
opts.momentum = 0.5;
opts.decay_rate = 0.5^(1 / opts.rounds);
opts.start_rate = 0.5;
LDN.train(Xtr,Ytr,opts);

% Set core learning params and train a bit
opts.momentum = 0.75;
opts.decay_rate = 0.5^(1 / opts.rounds);
opts.start_rate = 0.25;
LDN.train(Xtr,Ytr,opts);

% Set core learning params and train a bit
opts.momentum = 0.9;
opts.decay_rate = 0.5^(1 / opts.rounds);
opts.start_rate = 0.125;
LDN.train(Xtr,Ytr,opts);

% Set core learning params and train a bit
opts.momentum = 0.95;
opts.decay_rate = 0.1^(1 / opts.rounds);
opts.start_rate = 0.05;
LDN.train(Xtr,Ytr,opts);


