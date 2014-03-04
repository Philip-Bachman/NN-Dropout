from sys import stdout as stdout
import numpy as np
import numpy.random as npr
import gnumpy as gp
import LNFuncs as lnf
import LNLayer as lnl


class LNNet:
    def __init__(self, layer_sizes, act_func=lnf.rehu_trans, \
            loss_func=lnf.loss_lsq):
        self.layer_count = len(layer_sizes) - 1
        self.layers = []
        for i in range(self.layer_count):
            dim_in = layer_sizes[i] + 1
            dim_out = layer_sizes[i+1]
            if (i  == (self.layer_count - 1)):
                afun = lnf.line_trans
            else:
                afun = act_func
            self.layers.append(lnl.LNLayer(dim_in, dim_out, afun))
        self.out_loss = loss_func
        # Set default weight regularization parameters
        self.lam_l2 = 0.0
        self.lam_l1 = 0.0
        self.wt_bnd = 4.0
        # Set default DEV regularization parameters
        self.do_dev = 0
        self.dev_lams = []
        self.dev_types = []
        for i in range(self.layer_count):
            self.dev_lams.append(0.0)
            self.dev_types.append(1)
        # Set basic dropout parameters
        self.drop_input = 0.0
        self.drop_hidden = 0.0
        self.drop_undrop = 0.0
        # Set bias magnitude (applied to all layers)
        self.bias_val = 0.0
        return

    def check_acc(self, X, Y, Ws=[]):
        """Check classification accuracy using inputs X with classes Y.

        Classes should be represented in Y as a +/- 1 indicator matrix.
        """
        if (len(Ws) == 0):
            Ws = self.layer_weights()
        obs_count = X.shape[0]
        Yc = lnf.class_cats(Y)
        M = self.get_drop_masks(obs_count, 0, 0)
        A = self.feedforward(X, M, Ws)
        Yh = A[-1]
        Yhc = lnf.class_cats(Yh)
        hits = sum([(int(a) == int(b)) for (a, b) in zip(Yhc, Yc)])
        return (float(hits) / float(obs_count))

    def check_loss(self, X, Y, Ws=[]):
        """Check loss at output layer for observations X/Y."""
        if (len(Ws) == 0):
            Ws = self.layer_weights()
        obs_count = X.shape[0]
        M = self.get_drop_masks(obs_count, 0, 0)
        A = self.feedforward(X, M)
        O = self.out_loss(A[-1], Y)
        Yc = lnf.class_cats(Y)
        Yhc = lnf.class_cats(A[-1])
        hits = sum([(int(a) == int(b)) for (a, b) in zip(Yhc, Yc)])
        acc = float(hits) / float(obs_count)
        return {'loss': O['L'], 'acc': acc, 'grad': O['dL']}

    def weight_count(self):
        """Count the number of weights/parameters underlying this network."""
        N = sum([layer.W.size for layer in self.layers])
        return N

    def init_weights(self, wt_scale, b_scale=0.0, do_kill=1):
        """Do random initialization of the weights in each layer.

        The parameter wt_scale sets the scale of the normal distribution for
        general weights, b_scale gives a constant to which all biases will be
        initialized, and do_kill controls soft sparsification.
        """
        for i in range(self.layer_count):
            if (i == 0):
                self.layers[i].init_weights(wt_scale, b_scale, 0)
            else:
                self.layers[i].init_weights(wt_scale, b_scale, do_kill)
        return

    def set_weights(self, Ws):
        """Set weights for this network to those in the array Ws.

        Each weight array Wi = Ws[i] should be of the proper size for
        parameterizing the LNLayer at self.layers[i].
        """
        for i in range(self.layer_count):
            if not gp.is_garray(Ws[i]):
                Ws[i] = gp.garray(Ws[i])
        for i in range(self.layer_count):
            self.layers[i].set_weights(Ws[i])
        return

    def layer_weights(self):
        """Get an array Ws such that Ws[i] is self.layers[i].W."""
        Ws = [layer.W for layer in self.layers]
        return Ws

    def vector_weights(self, Ws=[]):
        """Get vectorized form of weights in Ws (or current net weights)."""
        if (len(Ws) == 0):
            Ws = self.layer_weights()
        Wv = [W.reshape((W.size, 1)) for W in Ws]
        return gp.concatenate(Wv, axis=0)

    def get_drop_masks(self, mask_count, in_drop=0, hd_drop=0):
        """Get mask_count dropout masks shaped for each layer in self.layers.

        Dropout masks are computed based on drop rates self.drop_input and
        self.drop_hidden, and self.drop_undrop. Masks are scaled so that the
        sum of each mask for a given layer is the same. If in_drop == 1, we do
        dropping on input layer and if hd_drop == 1, we also drop hiddens.
        """
        M = []
        # Generate an 'undrop' mask, which sets some masks to be dropless
        u_mask = (gp.rand(mask_count,1) < self.drop_undrop)
        for i in range(self.layer_count):
            # Set drop_rate based on layer and in_drop/hd_drop
            drop_rate = 0.0
            if ((i == 0) and (in_drop == 1)):
                drop_rate = self.drop_input
            elif (hd_drop == 1):
                drop_rate = self.drop_hidden
            # Get mask dimension for this layer
            mask_dim = self.layers[i].dim_input
            # Generate random 'bit' mask
            d_mask = (gp.rand(mask_count, mask_dim) > drop_rate)
            # Compute bootleg 'or' with the undrop mask
            mask = ((d_mask + u_mask) > 0.1)
            # Rescale mask entries to have unit mean
            scales = 1.0 / gp.mean(mask, axis=1)
            scales = scales[:,gp.newaxis]
            mask = mask * scales
            # Record the generated mask
            M.append(mask)
        return M

    def feedforward(self, X, M=[], Ws=[]):
        """Feedforward for inputs X with drop masks M and layer weights Ws."""
        if (len(M) == 0):
            # If no masks are given, use drop-free feedforward
            M = self.get_drop_masks(X.shape[0],0,0)
        if (len(Ws) == 0):
            # Default to this network's current per-layer weights
            Ws = self.layer_weights()
        A = []
        for i in range(self.layer_count):
            if (i == 0):
                # First layer receives X as input
                Xi = M[i] * lnf.bias(X, self.bias_val)
            else:
                # Other layers receive previous layer's activations as input
                Xi = M[i] * lnf.bias(A[i-1], self.bias_val)
            # Perform feedforward through the i'th network layer
            Ai = self.layers[i].feedforward(Xi, Ws[i])
            A.append(Ai['post'])
        return A

    def backprop(self, dLdA, A, X, M, Ws=[]):
        """Run backprop for the activation gradients in dLdA.

        The lengths (i.e. len()) of the lists of arrays dLdA, A, and M should
        all be self.layer_count. The shapes of dLdA[i] and A[i] should be the
        same for all i. The shape of M[i] should match the shape of A[i-1] for
        i from 1 to (self.layer_count - 1). The shape of M[0] should match the
        shape of X. Weight array list Ws defaults to self.layer_weights().
        """
        if (len(Ws) == 0):
            Ws = self.layer_weights()
        dLdWs = []
        dLdX = []
        for i in range((self.layer_count-1),-1,-1):
            if (i == 0):
                # First layer receives X as input
                Xi = M[i] * lnf.bias(X, self.bias_val)
            else:
                # Other layers receive previous layer's activations as input
                Xi = M[i] * lnf.bias(A[i-1], self.bias_val)
            # BP current grads onto current layer's weights and inputs
            Bi = self.layers[i].backprop(dLdA[i], A[i], Xi, Ws[i])
            # Rescale BP-ed input grads to account for dropout mask
            Bi['dLdX'] = M[i] * Bi['dLdX']
            if (i == 0):
                # BP-ed input grads at first layer are grads on X
                dLdX = lnf.unbias(Bi['dLdX'])
            else:
                # BP-ed input grads at other layers should be addded to
                # whatever grads were already there (e.g. DEV gradients)
                dLdA[i-1] = dLdA[i-1] + lnf.unbias(Bi['dLdX'])
            # Record the BP-ed gradients on current layer's inbound weights
            dLdWs.append(Bi['dLdW'])
        dLdWs.reverse()
        return {'dLdWs': dLdWs, 'dLdX': dLdX}

    def reg_loss(self, Ws=[]):
        """Compute basic L1/L2 loss and gradient on weights in Ws."""
        if (len(Ws) == 0):
            Ws = self.layer_weights()
        L = 0.0
        dLdWs = []
        for i in range(self.layer_count):
            L = L + (self.lam_l2 * gp.sum(Ws[i]**2.0))
            dLdWs.append((2.0 * self.lam_l2) * Ws[i])
        return {'L': L, 'dLdWs': dLdWs}

    def sde_loss(self, X, Y, M, Ws=[], do_print=0):
        """Compute dropout loss for inputs X with target outputs Y.

        This loss function computes the standard dropout loss for some inputs
        X with target outputs Y, when dropout is applied following the masks
        in M, given the layer weights in Ws (default self.layer_weights()).
        """
        if (len(Ws) == 0):
            Ws = self.layer_weights()
        # Compute droppy activations for observations in X
        A = self.feedforward(X, M, Ws)
        # Compute loss and gradient for output-layer activations
        O = self.out_loss(A[-1], Y)
        # Make list of activation gradients
        dLdA = [gp.zeros(Ai.shape) for Ai in A]
        dLdA[-1] = O['dL']
        # Backprop the output loss gradient through network
        B = self.backprop(dLdA, A, X, M, Ws)
        # Compute parameter regularization loss and gradients
        R = self.reg_loss(Ws)
        # Combine output loss, DEV loss, and regularization loss
        L = [O['L'], 0.0, R['L']]
        # Combine output loss gradient and regularization gradient
        dLdWs = [(dWb + dWr) for (dWb, dWr) in zip(B['dLdWs'], R['dLdWs'])]
        return {'L': L, 'dLdWs': dLdWs}


    def dev_loss(self, X, Y, M, Ws=[]):
        """Compute DEV-regularized loss for inputs X with target outputs Y.

        This loss function computes a combination of standard output loss
        (e.g. for classification/regression) and Dropout Ensemble Variance
        regularization loss. X should be a list of 'dev_reps' input arrays,
        where 'dev_reps' is the number of times each input will be pushed
        through a droppy network when computing the DEV regularizer. M should
        be a list of lists of per-layer dropout masks, matched to size of the
        input arrays in X. Y should contain the target outputs for X[0], for
        which inputs will be pushed through a drop-free network.
        """
        if (len(Ws) == 0):
            Ws = self.layer_weights()
        dev_reps = len(X)
        # Compute activations for observations in X
        A = [self.feedforward(X[i], M[i], Ws) for i in range(dev_reps)]
        # Compute loss and gradient for output-layer activations, for the
        # (should be) drop free feedforward of X[0].
        O = self.out_loss(A[0][-1], Y)
        # Make list of activation gradients
        dLdA = [[gp.zeros(Aj.shape) for Aj in A[0]] \
                for i in range(dev_reps)]
        dLdA[0][-1] = O['dL']
        # Compute DEV regularizer loss and gradients
        Ld = 0.0
        for i in range(self.layer_count):
            dev_type = self.dev_types[i]
            dev_lam = self.dev_lams[i]
            if (dev_lam > 0.0000001):
                Ai = [A[j][i] for j in range(dev_reps)]
                Di = lnf.dev_loss(Ai, dev_type, 0)
                Ld = Ld + (dev_lam * Di['L'])
                for j in range(dev_reps):
                    dLdA[j][i] = dLdA[j][i] + (dev_lam * Di['dLdA'][j])
        # Backpropagate gradients for each DEV rep
        B = {'dLdWs': [gp.zeros(W.shape) for W in Ws]}
        for i in range(dev_reps):
            Bi = self.backprop(dLdA[i], A[i], X[i], M[i], Ws)
            for j in range(self.layer_count):
                B['dLdWs'][j] = B['dLdWs'][j] + Bi['dLdWs'][j]
        # Compute parameter regularization loss and gradients
        R = self.reg_loss(Ws)
        # Combine output loss, DEV loss, and regularization loss
        L = [O['L'], Ld, R['L']]
        # Combine output loss gradient and regularization gradient
        dLdWs = [(dWb + dWr) for (dWb, dWr) in zip(B['dLdWs'], R['dLdWs'])]
        return {'L': L, 'dLdWs': dLdWs}

    def train(self, X, Y, opts={}):
        """Train this network using observations X/Y and options 'opts'.

        This does SGD.
        """
        # Fill out opts with defaults, and adjust self if needed
        opts = lnf.check_opts(opts)
        if opts.has_key('lam_l2'):
            self.lam_l2 = opts['lam_l2']
        if opts.has_key('lam_l1'):
            self.lam_l1 = opts['lam_l1']
        if opts.has_key('wt_bnd'):
            self.wt_bnd = opts['wt_bnd']
        # Grab params that control minibatch SGD
        batch_size = opts['batch_size']
        dev_reps = opts['dev_reps']
        rate = opts['start_rate']
        decay = opts['decay_rate']
        momentum = opts['momentum']
        rounds = opts['rounds']
        # Get initial weights, and an initial set of momentus updates
        Ws = self.layer_weights()
        self.set_weights(Ws)
        dLdWs_mom = [gp.zeros(W.shape) for W in Ws]
        # Get arrays for holding training batches and batches for loss
        # checking on the training set.
        Xb = gp.zeros((batch_size, X.shape[1]))
        Yb = gp.zeros((batch_size, Y.shape[1]))
        Xv = gp.zeros((min(X.shape[0],2000), X.shape[1]))
        Yv = gp.zeros((min(Y.shape[0],2000), Y.shape[1]))
        # Loop-da-loop
        b_start = 0
        for i in range(rounds):
            # Grab a minibatch of training examples
            b_end = b_start + batch_size
            if (b_end >= X.shape[0]):
                b_start = 0
                b_end = b_start + batch_size
            Xb = X[b_start:b_end,:]
            Yb = Y[b_start:b_end,:]
            b_start = b_end
            if (self.do_dev == 1):
                # Make lists of inputs and drop masks for DEV regularization
                Xb_a = [Xb for j in range(dev_reps)]
                Mb_a = [self.get_drop_masks(Xb.shape[0],int(j>0),int(j>0)) \
                        for j in range(dev_reps)]
                # Compute loss and gradients subject to DEV regularization
                loss_info = self.dev_loss(Xb_a, Yb, Mb_a, Ws)
            else:
                # Get dropout masks for the minibatch
                Mb = self.get_drop_masks(Xb.shape[0], 1, 1)
                # Compute SDE loss for the minibatch
                loss_info = self.sde_loss(Xb, Yb, Mb, Ws)
            # Adjust momentus updates and apply to Ws
            gentle_rate = min(1.0, (i / 1000.0)) * rate
            for j in range(self.layer_count):
                dLdWs_mom[j] = (momentum * dLdWs_mom[j]) + \
                        ((1.0 - momentum) * loss_info['dLdWs'][j])
                Ws[j] = Ws[j] - (gentle_rate * dLdWs_mom[j])
            # Update learning rate
            rate = rate * decay
            # Bound L2 norm of weights based on self.wt_bnd
            for j in range(self.layer_count):
                Ws[j] = self.layers[j].bound_weights(Ws[j], self.wt_bnd)
            # Give some feedback, to quell impatience and fidgeting
            if ((i == 0) or (((i + 1) % 200) == 0)):
                self.set_weights(Ws)
                lnf.sample_obs(X, Y, Xv, Yv)
                CL_tr = self.check_loss(Xv, Yv)
                print 'Round {0:6d}:'.format((i + 1))
                print ' Lo: {0:.4f}, Ld: {1:.4f}, Lr: {2:.4f}'.format(\
                        loss_info['L'][0],loss_info['L'][1],loss_info['L'][2])
                if (opts['do_validate'] == 1):
                    # Compute accuracy on validation set
                    lnf.sample_obs(opts['Xv'], opts['Yv'], Xv, Yv)
                    CL_te = self.check_loss(Xv, Yv)
                    print '    Atr: {0:.4f}, Ltr: {1:.4f}, Ate: {2:.4f}, Lte: {3:.4f}'.\
                            format(CL_tr['acc'], CL_tr['loss'], CL_te['acc'], CL_te['loss'])
                else:
                    print '    Atr: {0:.4f}, Ltr: {1:.4f}'.\
                            format(CL_tr['acc'], CL_tr['loss'])
                #print "  Matrix data types: "
                #print "    dLdWs_mom[0]: " + str(dLdWs_mom[0].dtype)
                #print "    Ws[0]: " + str(Ws[0].dtype)
                stdout.flush()





if __name__ == '__main__':
    from time import clock as clock
    obs_dim = 784
    out_dim = 10
    obs_count = 10000
    hidden_size = 250
    layer_sizes = [obs_dim, hidden_size, hidden_size, out_dim]
    # Generate dummy training data
    X = gp.randn((obs_count, obs_dim))
    Y = gp.randn((obs_count, out_dim))
    # Get some training options
    opts = lnf.check_opts()
    opts['rounds'] = 201
    opts['batch_size'] = 100
    opts['dev_reps'] = 2
    # Train a network (on BS data)
    LN = LNNet(layer_sizes, lnf.kspr_trans, lnf.loss_lsq)
    LN.do_dev = 1
    LN.dev_lams = [1.0 for i in range(LN.layer_count)]
    # Time training
    t1 = clock()
    LN.train(X,Y,opts)
    t2 = clock()
    print "TIME PER UPDATE: " + str(float(t2 - t1) / float(opts['rounds']))










##############
# EYE BUFFER #
##############
