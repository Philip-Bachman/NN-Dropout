% Initialize a FD-smoothed network to train
clear LSN;
LSN = LSNet([size(Xtr,2) 500 500 size(Ytr,2)], @LDLayer.rehu_trans, @LSNet.loss_mcl2h);
LSN.init_weights(0.1,0.01);
LSN.drop_input = 0.0;
LSN.drop_hidden = 0.0;
LSN.drop_undrop = 0.0;
LSN.ord_lams = [0.5 1.0];

% Setup general options for training
opts = struct();
opts.lam_l2 = 1e-5;
opts.rounds = 25000;
opts.start_rate = 0.5;
opts.batch_size = 32;
opts.fd_len = 0.05;
opts.momentum = 0.9;
opts.do_validate = 1;
opts.Xv = Xte;
opts.Yv = Yte;

LSN.train(Xtr,Ytr,opts);



