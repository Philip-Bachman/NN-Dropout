import numpy as np
import numpy.random as npr
import LNFuncs as lnf


class LNLayer:
    def __init__(self, in_dim, out_dim, afun=lnf.rehu_trans):
        self.dim_input = in_dim
        self.dim_output = out_dim
        self.W = np.float32(npr.randn(out_dim, in_dim))
        self.act_trans = afun
        self.ff_evals = 0
        self.bp_evals = 0
        return

    def weight_count(self):
        """Return the number of weights in this layer."""
        N = self.dim_input * self.dim_output
        return N

    def init_weights(self, wt_scale, b_scale=0, do_kill=0):
        """Randomly initialize the weights in this layer.

        Use wt_scale to scale the (normally-distributed) random initial values
        and set the bias column uniformly to b_scale. If do_kill is 1, then
        this rescales most weights for each node to smaller values.
        """
        Wm = wt_scale * npr.randn(self.dim_output, self.dim_input)
        Wm[:,-1] = b_scale
        if (do_kill == 1):
            for i in range(self.dim_output):
                keep_count = 50
                if (keep_count < self.dim_input):
                    keep_idx = npr.permutation(range(self.dim_input - 1))
                    kill_idx = keep_idx[51:]
                    Wm[i,kill_idx] = 0.1 * Wm[i,kill_idx]
        self.W = np.float32(Wm)
        return

    def set_weights(self, Wm):
        """Set weights in this layer to the given values."""
        if ((Wm.shape[0] != self.dim_output) or \
                (Wm.shape[1] != self.dim_input)):
            raise Exception('Wrong-sized Wm.')
        self.W = Wm
        return

    def vector_weights(self, Wm=np.array([0])):
        """Return the weights in Wm or self.W, vectorized."""
        if (Wm.size == 1):
            Wm = self.W
        Wv = np.reshape(Wm, (Wm.size, 1))
        return Wv

    def matrix_weights(self, Wv=np.array([0])):
        """Return the weights in Wv, or self.W, matrized."""
        if (Wv.size == 1):
            Wm = self.W
        else:
            if (Wv.size != self.weight_count()):
                raise Exception('Wrong-sized Wv.')
            Wm = np.reshape(Wv,self.dim_output,self.dim_input)
        return Wm

    def bound_weights(self, Wm, wt_bnd):
        """Bound L2 (row-wise) norm of the weights in Wm by wt_bnd."""
        EPS = 0.00000001
        # Compute L2 norm of weights inbound to each node in this layer
        w_norms = np.sqrt(np.sum(Wm**2,axis=1) + EPS)
        # Compute scales based on norms and the upperbound set by wt_bnd
        w_scales = wt_bnd / w_norms
        mask = np.float32((w_scales < 1.0))
        w_scales = (w_scales * mask) + (1.0 - mask)
        w_scales = np.reshape(w_scales, (Wm.shape[0], 1))
        # Rescale weights to meet the bound set by wt_bnd
        Wm = Wm * w_scales
        return Wm

    def feedforward(self, X, Wm=np.array([0])):
        """Run feedforward for this layer, with inputs X and weights Wm."""
        if (Wm.size == 1):
            Wm = self.W
        A = {}
        true_count = X.shape[0]
        if (true_count < 100000):
            X2 = np.tile(X, (2, 1))
            A2 = np.dot(X2, Wm.T)
            A['pre'] = A2[:true_count,:]
        else:
            A['pre'] = np.dot(X, Wm.T)
        if (X.shape[0] == 15000000):
            A1 = np.dot(X, Wm.T)
            X2 = np.tile(X, (2,1))
            A2 = np.dot(X2, Wm.T)
            print 'A1 shape: ({0:d}, {1:d})'.format(A1.shape[0],A1.shape[1])
            for i in range(A1.shape[0]):
                for j in range(A1.shape[1]):
                    v1 = float(A1[i,j]) #float(np.dot(X2[i,:], Wm[j,:]))
                    v2 = float(A2[i,j])
                    if (abs(v1 - v2) > 1e-4):
                        v3 = np.dot(X[(i%X.shape[0]),:],Wm[j,:])
                        print '({0:3d}, {1:3d}) is BAD: v1={2:.4f}, v2={3:.4f}, true={4:.4f}'.format(\
                                i, j, v1, v2, v3)
                    else:
                        print '({0:3d}, {1:3d}) is OK: v1={2:.4f}, v2={3:.4f}'.format(\
                                i, j, v1, v2)
        # Transform pre-activations into post-activations
        A['post'] = self.act_trans(A['pre'], 'ff')
        # Track total number of feedforward evals (for timing info)
        self.ff_evals = self.ff_evals + X.shape[0]
        return A

    def backprop(self, dLdA, A, X, Wm=np.array([0])):
        """Backprop through the activation process defined by this layer.

        dLdA gives the gradient on the activations in A, which are assumed to
        have been produced by self.feedforward(X, Wm). If values meeting this
        assumption are not provided, the result will be wrong. We return both
        the gradients on Wm and the gradients on X, for easy layer chaining.
        """
        if (Wm.size == 1):
            Wm = self.W
        true_count = X.shape[0]
        # Backpropagate dLdA through the activation transform, which was
        # applied to the linear-form pre-activations X*Wm'.
        dLdF = self.act_trans({'A': A, 'dLdA': dLdA}, 'bp')
        if (true_count < -1):
            dLdF = np.concatenate((dLdF, dLdF), axis=0) / 2.0
            X = np.concatenate((X, X), axis=0)
            dLdW = np.dot(dLdF.T, X)
            dLdX = np.dot(dLdF, Wm)
            dLdX = dLdX[:true_count,:]
        else:
            # Use basic linear-form gradient to bp through pre-activation
            dLdW = np.dot(dLdF.T, X)
            dLdX = np.dot(dLdF, Wm)
        # Track total number of bp evals (for timing purposes)
        self.bp_evals = self.bp_evals + X.shape[0]
        return {'dLdW': dLdW, 'dLdX': dLdX}



if __name__ == '__main__':
    from time import clock
    in_dim = 500
    out_dim = 500
    obs_count = 100
    X = np.float32(npr.randn(obs_count, in_dim))
    layer = LNLayer(in_dim, out_dim)
    layer.init_weights(0.1, 0.0, 1)
    t1 = clock()
    print "Bounding weights to norm 0.1:",
    for i in range(10):
        Wm = layer.bound_weights(layer.W, 0.1)
        print ".",
    print " "
    t2 = clock()
    print "Total time: " + str(t2 - t1)
    t1 = clock()
    print "Performing 10 feedforward passes:",
    for i in range(10):
        A = layer.feedforward(X, layer.W)
        print ".",
    print " "
    t2 = clock()
    print "Total time: " + str(t2 - t1)
    t1 = clock()
    dLdA = np.float32(npr.randn(A.shape[0], A.shape[1]))
    print "Performing 10 backprop passes:",
    for i in range(10):
        bp_grads = layer.backprop(dLdA, A, X, Wm)
        print ".",
    print " "
    t2 = clock()
    print "Total time: " + str(t2 - t1)

##############
# EYE BUFFER #
##############
