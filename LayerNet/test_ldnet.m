clear LDN;
LDN = LDNet([size(Xtr,2) 500 500 size(Ytr,2)], @LDLayer.rehu_trans, @LDNet.loss_mcl2h);
LDN.dev_lams = [1 1 20];
LDN.dev_types = [1 1 2];
LDN.init_weights(0.1,0.01);
LDN.drop_input = 0.2;
LDN.drop_hidden = 0.5;
LDN.drop_undrop = 0.0;
LDN.do_dev = 1;
LDN.dev_pre = 0;
LDN.lam_l2a = 0;

opts = struct();
opts.lam_l2 = 1e-5;
opts.rounds = 25000;
opts.start_rate = 0.5;
opts.batch_size = 32;
opts.dev_reps = 4;
opts.momentum = 0.9;
opts.do_validate = 1;
opts.Xv = Xte;
opts.Yv = Yte;

LDN.train(Xtr,Ytr,opts);



