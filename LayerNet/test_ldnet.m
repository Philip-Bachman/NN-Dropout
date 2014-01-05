clear;

% Load USPS digits data
load('usps.mat');
X = bsxfun(@minus, X, mean(X, 2));
X = bsxfun(@rdivide, X, max(abs(X),[],2));
Y = LDNet.class_inds(Y);
[Xtr Ytr Xte Yte] = trte_split(X,Y,0.8);

% Create a base network to work with
LDN = LDNet([size(Xtr,2) 500 500 size(Ytr,2)], @LDLayer.rehu_trans, @LDNet.loss_mcl2h);

% Setup SGD-related training options
opts = struct();
opts.lam_l2 = 1e-5;
opts.rounds = 15000;
opts.decay_rate = 0.1^(1 / opts.rounds);
opts.start_rate = 0.1;
opts.batch_size = 50;
opts.dev_reps = 4;
opts.momentum = 0.9;
opts.do_validate = 1;
opts.Xv = Xte;
opts.Yv = Yte;

% Setup network parameters
LDN.dev_lams = [0.0 2.0 20];
LDN.dev_types = [1 1 2];
LDN.drop_input = 0.2;
LDN.drop_hidden = 0.5;
LDN.drop_undrop = 0.0;
LDN.do_dev = 1;
LDN.dev_pre = 0;
LDN.wt_bnd = 3.0;

% Run training with DEV regularization
LDN.do_dev = 1;
LDN.init_weights(0.1,0.01);
LDN.train(Xtr,Ytr,opts);
[L_dev A_dev] = LDN.check_loss(Xte,Yte);

% Run training with SDE regularization
LDN.do_dev = 0;
LDN.init_weights(0.1,0.01);
LDN.train(Xtr,Ytr,opts);
[L_sde A_sde] = LDN.check_loss(Xte,Yte);

% Run training without special regularization
LDN.do_dev = 0;
LDN.drop_input = 0.0;
LDN.drop_hidden = 0.0;
LDN.init_weights(0.1,0.01);
LDN.train(Xtr,Ytr,opts);
[L_raw A_raw] = LDN.check_loss(Xte,Yte);



