clear LDN;
LDN = LDNet([size(Xtr,2) 500 500 size(Ytr,2)], @LDLayer.relu_trans, @LDNet.loss_mclr);
LDN.dev_lams = [1 20 1];
LDN.dev_types = [1 1 1];
LDN.init_weights(0.1,0.01);
LDN.drop_input = 0.2;
LDN.drop_hidden = 0.5;
LDN.drop_undrop = 0.0;
LDN.do_dev = 1;
LDN.dev_pre = 0;
LDN.lam_l2a = 1e-5;

opts = struct();
opts.lam_l2 = 1e-5;
opts.rounds = 20000;
opts.start_rate = 1.0;
opts.batch_size = 50;
opts.dev_reps = 2;
opts.momentum = 0.9;
opts.do_validate = 1;
opts.Xv = Xte;
opts.Yv = Yte;

LDN.train(Xtr,Ytr,opts);



