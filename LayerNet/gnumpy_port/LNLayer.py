import numpy as np
import numpy.random as npr
import gnumpy as gp
import LNFuncs as lnf


class LNLayer:
    def __init__(self, in_dim, out_dim, afun=lnf.rehu_trans):
        self.dim_input = in_dim
        self.dim_output = out_dim
        self.W = gp.randn((out_dim, in_dim))
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
        self.W = gp.garray(Wm)
        return

    def set_weights(self, Wm):
        """Set weights in this layer to the given values.

        This performs a copy, so modifications of the given Wm, e.g. during
        network training, won't affect the values set for self.W.
        """
        if ((Wm.shape[0] != self.dim_output) or \
                (Wm.shape[1] != self.dim_input)):
            raise Exception('Wrong-sized Wm.')
        if not gp.is_garray(Wm):
            Wm = gp.garray(Wm)
        self.W = Wm.copy()
        return

    def vector_weights(self, Wm=gp.garray(())):
        """Return the weights in Wm or self.W, vectorized."""
        if (Wm.size == 0):
            Wm = self.W
        if not gp.is_garray(Wm):
            Wm = gp.garray(Wm)
        Wv = Wm.reshape((Wm.size, 1))
        return Wv

    def matrix_weights(self, Wv=gp.garray(())):
        """Return the weights in Wv, or self.W, matrized."""
        if (Wv.size == 0):
            Wm = self.Wm
        else:
            if not gp.is_garray(Wv):
                Wv = gp.garray(Wv)
            if (Wv.size != self.weight_count()):
                raise Exception('Wrong-sized Wv.')
            Wm = Wv.reshape((self.dim_output,self.dim_input))
        return Wm

    def bound_weights(self, Wm, wt_bnd):
        """Bound L2 (row-wise) norm of the weights in Wm by wt_bnd.

        This returns a garray if passed a garray, and performs all ops on the
        GPU if that is the case. Otherwise, it returns a numpy array, or if
        something besides an ndarray/garray was passed, it crashes (probably).
        """
        EPS = 0.00000001
        # Compute L2 norm of weights inbound to each node in this layer
        w_norms = gp.sqrt(gp.sum(Wm**2,axis=1) + EPS)
        # Compute scales based on norms and the upperbound set by wt_bnd
        w_scales = wt_bnd / w_norms
        mask = (w_scales < 1.0)
        w_scales = (w_scales * mask) + (1.0 - mask)
        w_scales = w_scales[:,gp.newaxis]
        # Rescale weights to meet the bound set by wt_bnd
        Wm = Wm * w_scales
        return Wm

    def feedforward(self, X, Wm=gp.garray(())):
        """Run feedforward for this layer, with inputs X and weights Wm."""
        if (Wm.size == 0):
            Wm = self.W
        A = {}
        A['pre'] = gp.dot(X, Wm.T)
        # Transform pre-activations into post-activations
        A['post'] = self.act_trans(A['pre'], 'ff')
        # Track total number of feedforward evals (for timing info)
        self.ff_evals = self.ff_evals + X.shape[0]
        return A

    def backprop(self, dLdA, A, X, Wm=gp.garray(())):
        """Backprop through the activation process defined by this layer.

        dLdA gives the gradient on the activations in A, which are assumed to
        have been produced by self.feedforward(X, Wm). If values meeting this
        assumption are not provided, the result will be wrong. We return both
        the gradients on Wm and the gradients on X, for easy layer chaining.
        """
        if (Wm.size == 0):
            Wm = self.W
        # Backpropagate dLdA through the activation transform, which was
        # applied to the linear-form pre-activations X*Wm'.
        dLdF = self.act_trans({'A': A, 'dLdA': dLdA}, 'bp')
        # Use basic linear-form gradient to bp through pre-activation
        dLdW = gp.dot(dLdF.T, X)
        dLdX = gp.dot(dLdF, Wm)
        # Track total number of bp evals (for timing purposes)
        self.bp_evals = self.bp_evals + X.shape[0]
        return {'dLdW': dLdW, 'dLdX': dLdX}



if __name__ == '__main__':
    from time import clock
    in_dim = 800
    out_dim = 800
    obs_count = 1000
    X = gp.randn((obs_count, in_dim))
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
    A = A['post']
    dLdA = gp.randn((A.shape[0], A.shape[1]))
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
