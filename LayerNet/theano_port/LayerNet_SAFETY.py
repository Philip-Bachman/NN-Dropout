import numpy as np
import cPickle
import gzip
import os
import sys
import time
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.ifelse import ifelse
import theano.printing
import theano.tensor.shared_randomstreams

import utils as utils
#from logistic_sgd import LogisticRegression
from output_losses import LogisticRegression
from load_data import load_umontreal_data, load_mnist


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out,
                 activation, W=None, b=None,
                 use_bias=False):

        self.input = input
        self.activation = activation

        if W is None:
            W_values = np.asarray(0.01 * rng.standard_normal(
                size=(n_in, n_out)), dtype=theano.config.floatX)
            W = theano.shared(value=W_values, name='W')

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX) + 0.001
            b = theano.shared(value=b_values, name='b')

        self.W = W
        self.b = b

        # Compute linear "pre-activation" for this layer
        if use_bias:
            self.linear_output = T.dot(input, self.W) + self.b
        else:
            self.linear_output = T.dot(input, self.W)
        # Apply desired transform to compute "activation" for this layer
        if activation is None:
            self.output = self.linear_output
        else:
            self.output = activation(self.linear_output)

        # Compute expected sum of squared activations, for regularization
        self.act_sq_sum = T.sum(self.output**2.0) / self.output.shape[0]

        # parameters of the model
        if use_bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]


def _dropout_from_layer(rng, layer, p):
    """p is the probablity of dropping a unit
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output

class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out,
                 activation, use_bias, W=None, b=None):
        super(DropoutHiddenLayer, self).__init__( \
                rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b, \
                activation=activation, use_bias=use_bias)
        undropped_output = self.output
        self.output = _dropout_from_layer(rng, undropped_output, p=0.5)


class DEV_MLP(object):
    """A multilayer perceptron with all the trappings required to do dropout
    training.

    """
    def __init__(self,
            rng,
            input,
            params,
            use_bias=True):

        # Setup simple activation function for this net
        rectified_linear_activation = lambda x: T.maximum(0.0, x)
        # Grab some of the parameters for this net
        layer_sizes = params['layer_sizes']
        lam_l2a = params['lam_l2a']
        dev_clones = params['dev_clones']
        # Make a dict to tell which parameters are norm-boundable
        self.clip_params = {}
        # Set up all the hidden layers
        weight_matrix_sizes = zip(layer_sizes, layer_sizes[1:])
        self.layers = []
        self.drop_nets = [[] for i in range(dev_clones)]
        # Initialize "next inputs", to be piped into new layers
        next_raw_input = input
        next_drop_inputs = [_dropout_from_layer(rng, input, p=0.2) \
                for i in range(dev_clones)]
        # Iteratively append layers to the RAW net and each of some number
        # of droppy DEV clones.
        first_layer = True
        for n_in, n_out in weight_matrix_sizes:
            # Add a new layer to the RAW (i.e. undropped) net
            self.layers.append(HiddenLayer(rng=rng,
                    input=next_raw_input,
                    activation=rectified_linear_activation,
                    n_in=n_in, n_out=n_out,
                    use_bias=use_bias))
            next_raw_input = self.layers[-1].output
            self.clip_params[self.layers[-1].W] = 1
            self.clip_params[self.layers[-1].b] = 0
            # Add a new dropout layer to each DEV clone, using the previous
            # layer in the corresponding DEV clone as input. The new DEV clone
            # layers all share parameters with the new RAW layer.
            W_drop = ((1.0/0.8) if first_layer else (1.0/0.5)) * self.layers[-1].W
            b_drop = self.layers[-1].b
            for i in range(dev_clones):
                self.drop_nets[i].append(DropoutHiddenLayer(rng=rng, \
                        input=next_drop_inputs[i], \
                        activation=rectified_linear_activation, \
                        W=W_drop, \
                        b=b_drop, \
                        n_in=n_in, n_out=n_out, use_bias=use_bias))
                next_drop_inputs[i] = self.drop_nets[i][-1].output
            first_layer = False
        # Use the negative log likelihood of the logistic regression layer of
        # the first DEV clone as dropout optimization objective.
        self.sde_out_func = LogisticRegression(self.drop_nets[0][-1])
        self.sde_class_loss = self.sde_out_func.loss_func
        self.sde_reg_loss = lam_l2a * T.sum([lay.act_sq_sum for lay in self.drop_nets[0]])
        self.sde_errors = self.sde_out_func.errors
        # Use the negative log likelihood of the logistic regression layer of
        # the RAW net as the standard optimization objective.
        self.raw_out_func = LogisticRegression(self.layers[-1])
        self.raw_class_loss = self.raw_out_func.loss_func
        self.raw_reg_loss = lam_l2a * T.sum([lay.act_sq_sum for lay in self.layers])
        self.raw_errors = self.raw_out_func.errors
        # Compute DEV loss based on the classification performance of the RAW
        # net and the "Dropout Ensemble Variance"
        self.dev_out_func = self.raw_out_func
        self.dev_class_loss = self.raw_class_loss
        self.dev_reg_loss = self.raw_reg_loss + self.dev_loss()

        # Grab all the parameters together.
        self.params = [ param for layer in self.layers for param in layer.params ]

    def dev_loss(self):
        """Compute Dropout Ensemble Variance regularizer term."""
        F1 = self.layers[-1].linear_output
        F2 = self.drop_nets[0][-1].linear_output
        L = T.sum((T.tanh(F1) - T.tanh(F2))**2.0) / F1.shape[0]
        return L


def test_mlp(
        initial_learning_rate,
        learning_rate_decay,
        wt_norm_bound,
        n_epochs,
        batch_size,
        dropout,
        results_file_name,
        mlp_params,
        dataset,
        use_bias):
    """
    The dataset is the one from the mlp demo on deeplearning.net.  This training
    function is lifted from there almost exactly.

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


    """
    datasets = load_mnist(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    print "train batches: {0:d}, valid batches: {1:d}, test_batches: {2:d}".format( \
            n_train_batches, n_valid_batches, n_test_batches)

    ######################
    # BUILD ACTUAL MODEL #
    ######################

    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    epoch = T.scalar()
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as a vector of [int] labels
    learning_rate = theano.shared(np.asarray(initial_learning_rate,
        dtype=theano.config.floatX))

    rng = np.random.RandomState(1234)

    # construct the MLP class
    NET = DEV_MLP(rng=rng, input=x, params=mlp_params, use_bias=use_bias)

    # Build the expressions for the cost functions.
    raw_cost = NET.raw_class_loss(y) + NET.raw_reg_loss
    sde_cost = NET.sde_class_loss(y) + NET.sde_reg_loss
    dev_cost = NET.dev_class_loss(y) + NET.dev_reg_loss

    ############################################################################
    # Compile testing and validation models. These models are evaluated on     #
    # batches of the same size as used in training. Trying to jam a large      #
    # validation or test set through the net may take too much memory.         #
    ############################################################################
    test_model = theano.function(inputs=[index],
            outputs=NET.raw_errors(y),
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]})
    #theano.printing.pydotprint(test_model, outfile="test_file.png",
    #        var_with_name_simple=True)
    validate_model = theano.function(inputs=[index],
            outputs=NET.raw_errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]})
    #theano.printing.pydotprint(validate_model, outfile="validate_file.png",
    #        var_with_name_simple=True)

    # Setup gradient variables (i.e. symbolic receptacles for symbolic grads)
    gparams = []
    for param in NET.params:
        # Use the right cost function here to train with or without dropout.
        if dropout:
            gparam = T.grad(sde_cost, param)
        else:
            gparam = T.grad(raw_cost, param)
        gparams.append(gparam)

    # Setup momentum variables
    gparams_mom = []
    for param in NET.params:
        gparam_mom = theano.shared(np.zeros(param.get_value(borrow=True).shape,
            dtype=theano.config.floatX))
        gparams_mom.append(gparam_mom)

    # Compute momentum for the current epoch
    mom = ifelse(epoch < 500,
            0.5*(1. - epoch/500.) + 0.99*(epoch/500.),
            0.99)

    # Update the step direction using momentum
    updates = OrderedDict()
    for gparam_mom, gparam in zip(gparams_mom, gparams):
        updates[gparam_mom] = mom * gparam_mom + (1. - mom) * gparam

    # ... and take a step along that direction
    for param, gparam_mom in zip(NET.params, gparams_mom):
        new_param = param - learning_rate * updates[gparam_mom]
        # Clip the updated param to bound its norm (where applicable)
        if (NET.clip_params.has_key(param) and \
                (NET.clip_params[param] == 1)):
            squared_norms = T.sum(new_param**2, axis=1).reshape((new_param.shape[0],1))
            scale = T.clip(T.sqrt(wt_norm_bound / squared_norms), 0., 1.)
            updates[param] = new_param * scale
        else:
            updates[param] = new_param

    # Compile theano function for training.  This returns the training cost and
    # updates the model parameters.
    output = dev_cost if dropout else raw_cost
    train_model = theano.function(inputs=[epoch, index], outputs=output,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})
    #theano.printing.pydotprint(train_model, outfile="train_file.png",
    #        var_with_name_simple=True)

    # Theano function to decay the learning rate, this is separate from the
    # training function because we only want to do this once each epoch instead
    # of after each minibatch.
    decay_learning_rate = theano.function(inputs=[], outputs=learning_rate,
            updates={learning_rate: learning_rate * learning_rate_decay})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    best_params = None
    best_validation_errors = np.inf
    best_iter = 0
    test_score = 0.
    epoch_counter = 0
    start_time = time.clock()

    results_file = open(results_file_name, 'wb')

    e_time = time.clock()
    while epoch_counter < n_epochs:
        # Train this epoch
        epoch_counter = epoch_counter + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_raw_cost = train_model(epoch_counter, minibatch_index)

        # Compute classification errors on validation set
        validation_errors = [validate_model(i) for i in xrange(n_valid_batches)]
        this_validation_errors = np.sum(validation_errors)

        # Report and save progress.
        print "epoch {}, test error {}, learning_rate={}{}".format(
                epoch_counter, this_validation_errors,
                learning_rate.get_value(borrow=True),
                " **" if this_validation_errors < best_validation_errors else "")
        print "--time: {0:.4f}".format((time.clock() - e_time))
        e_time = time.clock()

        best_validation_errors = min(best_validation_errors,
                this_validation_errors)
        results_file.write("{0}\n".format(this_validation_errors))
        results_file.flush()

        new_learning_rate = decay_learning_rate()

        # Save first layer weights to an image locally
        utils.visualize(NET, 0, 'net_weights.png')

    end_time = time.clock()

    # Compute loss on test set
    test_scores = [test_model(i) for i in xrange(n_test_batches)]
    test_score = np.sum(test_scores)
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_errors * 100., best_iter, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    import sys

    initial_learning_rate = 0.5
    learning_rate_decay = 0.998
    wt_norm_bound = 3.75
    n_epochs = 3000
    batch_size = 100
    mlp_params = {}
    mlp_params['layer_sizes'] = [28*28, 500, 500, 10]
    mlp_params['lam_l2a'] = 1e-5
    mlp_params['dev_clones'] = 1
    dataset = 'data/mnist_batches.npz'
    #dataset = 'data/mnist.pkl'

    if len(sys.argv) < 2:
        print "Usage: {0} [dropout|backprop]".format(sys.argv[0])
        exit(1)

    elif sys.argv[1] == "dropout":
        dropout = True
        results_file_name = "results_dropout.txt"

    elif sys.argv[1] == "backprop":
        dropout = False
        results_file_name = "results_backprop.txt"

    else:
        print "I don't know how to '{0}'".format(sys.argv[1])
        exit(1)

    test_mlp(initial_learning_rate=initial_learning_rate,
             learning_rate_decay=learning_rate_decay,
             wt_norm_bound=wt_norm_bound,
             n_epochs=n_epochs,
             batch_size=batch_size,
             mlp_params=mlp_params,
             dropout=dropout,
             dataset=dataset,
             results_file_name=results_file_name,
             use_bias=False)

