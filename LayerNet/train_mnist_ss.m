function [ LDN ] = train_mnist_ss( Xtr, Ytr, Xtr_u, Xte, Yte, train_type )
% Train a layer net using a particular rate/momentum schedule

% Setup SGD-related training options
opts = struct();
opts.do_draw = 0;
opts.lam_l2 = 1e-5;
opts.batch_size = 100;
opts.dev_reps = 2;
opts.do_validate = 1;
opts.Xv = Xte;
opts.Yv = Yte;

% Create a network to train with 
LDN = LDNet([size(Xtr,2) 800 800 size(Ytr,2)], @LDLayer.relu_trans, @LDNet.loss_mcl2h);
LDN.lam_l2a = [1e-6 1e-6 1e-3];
LDN.dev_lams = [1.0 1.0 20.0];
LDN.dev_types = [1 1 2];
if (strcmp(train_type, 'dev') || strcmp(train_type, 'sde'))
    LDN.drop_input = 0.2;
    LDN.drop_hidden = 0.5;
end
if strcmp(train_type, 'dev')
    LDN.do_dev = 1;
else
    LDN.do_dev = 0;
end
LDN.drop_undrop = 0.0;
LDN.dev_pre = 0;
LDN.wt_bnd = 2.0;

% Initialize the network weights
LDN.init_weights(0.1,0.01);

% Train through several phases of decreasing aggresion
opts.rounds = 5000;
momentums =[0.75 0.9];
rates = [0.2 0.1];
for i=1:numel(momentums),
    opts.momentum = momentums(i);
    opts.start_rate = rates(i);
    opts.decay_rate = 1.0;
    if strcmp(train_type,'dev')
        LDN.train_ss(Xtr, Ytr, [Xtr; Xtr_u], opts);
    else
        LDN.train(Xtr,Ytr,opts);
    end
end

% Train a final "top-off" phase
opts.rounds = 20000;
opts.momentum = 0.95;
opts.start_rate = 0.1;
opts.decay_rate = 0.1^(1 / opts.rounds);
if strcmp(train_type,'dev')
    LDN.train_ss(Xtr, Ytr, [Xtr; Xtr_u], opts);
else
    LDN.train(Xtr, Ytr, opts);
end

return
end

