clear;

% Load USPS digits data
load('usps.mat');
X = bsxfun(@minus, X, mean(X, 2));
X = single(bsxfun(@rdivide, X, max(abs(X),[],2)));
Y = single(LDNet.class_inds(Y));
[Xtr Ytr Xte Yte] = trte_split(X,Y,0.8);

% Split training data into supervised and unsupervised components
sup_count = 1000;
Xtr_u = Xtr((sup_count+1):end,:);
Ytr_u = Ytr((sup_count+1):end,:);
Xtr = Xtr(1:sup_count,:);
Ytr = Ytr(1:sup_count,:);

% Setup SGD-related training options
opts = struct();
opts.lam_l2 = 0;
opts.rounds = 15000;
opts.decay_rate = 0.1^(1 / opts.rounds);
opts.start_rate = 0.2;
opts.batch_size = 100;
opts.dev_reps = 2;
opts.momentum = 0.9;
opts.do_validate = 1;
opts.Xv = Xte;
opts.Yv = Yte;

% Create a network to train with DEV regularization
LDN_dev = LDNet([size(Xtr,2) 400 400 size(Ytr,2)], @LDLayer.rehu_trans, @LDNet.loss_mcl2h);
LDN_dev.lam_l2a = [1e-5 1e-5 1e-5];
LDN_dev.dev_lams = [1.0 1.0 10.0];
LDN_dev.dev_types = [1 1 2];
LDN_dev.drop_input = 0.2;
LDN_dev.drop_hidden = 0.5;
LDN_dev.drop_undrop = 0.0;
LDN_dev.do_dev = 1;
LDN_dev.dev_pre = 0;
LDN_dev.wt_bnd = 2.0;
LDN_dev.init_weights(0.1,0.01);
Ws = LDN_dev.struct_weights(); % All nets will start with these weights
LDN_dev.train_ss(Xtr,Ytr,[Xtr; Xtr_u],opts);
[L_dev A_dev] = LDN_dev.check_loss(Xte,Yte);
F_dev = LDN_dev.feedforward(Xte);

% Create a network to train with SDE regularization
LDN_sde = LDNet([size(Xtr,2) 400 400 size(Ytr,2)], @LDLayer.rehu_trans, @LDNet.loss_mcl2h);
LDN_sde.lam_l2a = [1e-5 1e-5 1e-5];
LDN_sde.dev_lams = [1.0 1.0 10.0];
LDN_sde.dev_types = [1 1 2];
LDN_sde.drop_input = 0.2;
LDN_sde.drop_hidden = 0.5;
LDN_sde.drop_undrop = 0.0;
LDN_sde.do_dev = 0;
LDN_sde.dev_pre = 0;
LDN_sde.wt_bnd = 2.0;
LDN_sde.set_weights(Ws);
LDN_sde.train(Xtr,Ytr,opts);
[L_sde A_sde] = LDN_sde.check_loss(Xte,Yte);
F_sde = LDN_sde.feedforward(Xte);

% Create a network to train with basically no regularization
LDN_raw = LDNet([size(Xtr,2) 400 400 size(Ytr,2)], @LDLayer.rehu_trans, @LDNet.loss_mcl2h);
LDN_raw.lam_l2a = [1e-5 1e-5 1e-5];
LDN_raw.dev_lams = [1.0 1.0 10.0];
LDN_raw.dev_types = [1 1 2];
LDN_raw.drop_input = 0.0;
LDN_raw.drop_hidden = 0.0;
LDN_raw.drop_undrop = 0.0;
LDN_raw.do_dev = 0;
LDN_raw.dev_pre = 0;
LDN_raw.wt_bnd = 2.0;
LDN_raw.set_weights(Ws);
LDN_raw.train(Xtr,Ytr,opts);
[L_raw A_raw] = LDN_raw.check_loss(Xte,Yte);
F_raw = LDN_raw.feedforward(Xte);



