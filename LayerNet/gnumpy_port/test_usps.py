import numpy as np
import numpy.random as npr
import LNFuncs as lnf
import LNLayer as lnl
import LNNet as lnn
from time import clock as clock


if __name__ == '__main__':
    # Load usps data
    X = np.load('usps_X_numpy.npy')
    Y = np.load('usps_Y_numpy.npy')
    X = X - np.reshape(np.mean(X,axis=1), (X.shape[0], 1))
    X = X / np.reshape(np.max(np.abs(X),axis=1), (X.shape[0], 1))
    # Split into random training/test portions
    [Xtr, Ytr, Xte, Yte] = lnf.trte_split(X, Y, 0.8)
    # Configure network layer sizes
    obs_dim = X.shape[1]
    out_dim = Y.shape[1]
    hidden_size = 300
    layer_sizes = [obs_dim, hidden_size, hidden_size, out_dim]
    # Set some training options
    opts = lnf.check_opts()
    opts['rounds'] = 15000
    opts['batch_size'] = 100
    opts['start_rate'] = 0.1
    opts['momentum'] = 0.9
    opts['decay_rate'] = 0.1**(1.0 / opts['rounds'])
    opts['dev_reps'] = 2
    opts['do_validate'] = 1
    opts['Xv'] = Xte
    opts['Yv'] = Yte
    # Initialize a network and set its parameters
    LN = lnn.LNNet(layer_sizes, lnf.rehu_trans, lnf.loss_mcl2h)
    LN.bias_val = 0.1
    LN.wt_bnd = 2.0
    LN.lam_l2 = 1e-5
    LN.init_weights(0.1, 0.01, 1)
    LN.dev_lams = [0.0, 1.0, 20.0]
    LN.dev_types = [1, 1, 2]
    LN.do_dev = 1
    LN.drop_input = 0.2
    LN.drop_hidden = 0.5
    LN.drop_undrop = 0.2
    # Train first with DEV regularization
    LN.train(Xtr,Ytr,opts)
    CL_dev = LN.check_loss(Xte,Yte)
    # Train with standard dropout
    LN.do_dev = 0
    LN.init_weights(0.1, 0.01, 1)
    LN.train(Xtr,Ytr,opts)
    CL_sde = LN.check_loss(Xte,Yte)
    # Train raw, with just L2 regularization
    LN.drop_input = 0.0
    LN.drop_hidden = 0.0
    LN.init_weights(0.1, 0.01, 1)
    LN.train(Xtr,Ytr,opts)
    CL_raw = LN.check_loss(Xte,Yte)
    # Display results, for posterity
    #print 'DEV: loss = {0:.4f}, acc = {1:.4f}'.format(CL_dev['loss'],CL_dev['acc'])
    #print 'SDE: loss = {0:.4f}, acc = {1:.4f}'.format(CL_sde['loss'],CL_sde['acc'])
    print 'RAW: loss = {0:.4f}, acc = {1:.4f}'.format(CL_raw['loss'],CL_raw['acc'])











##############
# EYE BUFFER #
##############
