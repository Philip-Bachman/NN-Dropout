"""
This code implements a (currently) single-layer k-sparse autoencoder, with
the option for dropout in training and "marginalized dropout" at encoding.
"""

import cPickle
import gzip
import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T


from load_data import load_umontreal_data
from utils import tile_raster_images
from sys import stdout as stdout

import PIL.Image

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

class KSAE(object):
    """k-sparse autoencoder class.

    """

    def __init__(self, numpy_rng, input=None, n_visible=784, n_hidden=100,
                 W=None, bhid=None, bvis=None):
        """Initialize the KSAE class by specifying the number of visible units
        (the dimension d of the input), the number of hidden units (the dimension
        d' of the latent or hidden space) and the contraction level. The
        constructor also receives symbolic variables for the input, weights and
        bias.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given
                     one is generated based on a seed drawn from `rng`

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for
                      standalone KSAE

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None

        :type bhid: theano.tensor.TensorType
        :param bhid: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None

        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None

        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = np.asarray(numpy_rng.uniform(
                      low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                      high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                      size=(n_visible, n_hidden)),
                                      dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(value=np.zeros(n_visible,dtype=theano.config.floatX),
                                 name='b_vis',
                                 borrow=True)

        if not bhid:
            bhid = theano.shared(value=np.zeros(n_hidden,dtype=theano.config.floatX),
                                 name='b_hid',
                                 borrow=True)

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T

        # if no input is given, generate a variable representing the input
        if input == None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    def get_hidden_values(self, input, k_sparse):
        """Computes the activations of the hidden layer, without dropout.
        """
        A = T.dot(input, self.W) + self.b
        A_abs = abs(A)
        A_srt = A_abs.sort(axis=1)
        A_ks = A * (A_abs >= A_srt[:,-k_sparse].reshape(A.shape[0],1))
        return A_ks

    def get_hidden_values_droppy(self, input):
        """Computes the activations of the hidden layer, subject to dropout.
        """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input from hidden layer activations.
        """
        return  T.dot(hidden, self.W_prime) + self.b_prime

    def get_cost_updates(self, learning_rate=0.1, k_sparse=20, do_dropout=0):
        """Compute the cost and the updates for one training step of the KSAE.
        """

        y = self.get_hidden_values(self.x, k_sparse)
        x_rec = self.get_reconstructed_input(y)
        self.L_rec = T.sum((self.x - x_rec)**2.0)
        cost = T.mean(self.L_rec)

        # compute the gradients of the cost of the `KSAE` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))

        return (cost, updates)


def test_ksae(start_rate=0.01, decay_rate=1.0, training_epochs=20,
            dataset='./data/mnist.pkl.gz',
            batch_size=50, output_folder='CAE_plots'):
    """
    This demo is tested on MNIST

    :type learning_rate: float
    :param learning_rate: learning rate used for training the contracting
                          AutoEncoder

    :type training_epochs: int
    :param training_epochs: number of epochs used for training

    :type dataset: string
    :param dataset: path to the picked dataset

    """
    datasets = load_umontreal_data(dataset)
    train_set_x, train_set_y = datasets[0]

    learning_rate = theano.shared(np.asarray(start_rate,
        dtype=theano.config.floatX))

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)
    ####################################
    #        BUILDING THE MODEL        #
    ####################################

    rng = np.random.RandomState(123)

    ksae = KSAE(numpy_rng=rng, input=x, n_visible=(28 * 28), n_hidden=500)

    cost, updates = ksae.get_cost_updates(learning_rate=learning_rate)

    print '... building Theano functions:',
    # Build a function for updating the CAE parameters
    train_ksae = theano.function([index], [T.mean(ksae.L_rec)],
                               updates=updates,
                               givens={x: train_set_x[index * batch_size:
                                                    (index + 1) * batch_size]})
    # Build a function for decaying the learning rate
    set_learning_rate = theano.function(inputs=[], outputs=learning_rate, \
            updates={learning_rate: learning_rate * decay_rate})
    print 'done'

    start_time = time.clock()

    ############
    # TRAINING #
    ############

    # go through training epochs
    for epoch in xrange(training_epochs):
        c = np.zeros((1,n_train_batches))
        # Cycle over all training minibatches for this epoch...
        print "Epoch {0:d}:".format(epoch),
        stdout.flush()
        for batch_index in xrange(n_train_batches):
            c[batch_index] = train_ksae(batch_index)
            if ((batch_index % (n_train_batches / 30)) == 0):
                print ".",
                stdout.flush()
        print " "
        update_the_rate = set_learning_rate()
        # Display diagnostics for the most recent epoch of training
        print "-- reconstruction: {0:.4f}".format( np.mean(c) )
        # Visualize filters in their current state
        image = PIL.Image.fromarray(tile_raster_images(
            X=ksae.W.get_value(borrow=True).T,
            img_shape=(28, 28), tile_shape=(10, 10),
            tile_spacing=(1, 1)))
        image.save('ksae_filters.png')
        # Save the CAE parameters to disk
        W = ksae.W.get_value(borrow=False)
        b_encode = ksae.W.get_value(borrow=False)
        b_decode = ksae.b_prime.get_value(borrow=False)
        np.save('ksae_W.npy',W)
        np.save('ksae_b_encode.npy',b_encode)
        np.save('ksae_b_decode.npy',b_decode)

    # Record total training time, just for kicks
    end_time = time.clock()
    training_time = (end_time - start_time)
    # Print some jibber-jabber
    print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((training_time) / 60.))
    # Eject, eJect, EJECT!
    os.chdir('../')


if __name__ == '__main__':
    test_ksae(start_rate=0.01, decay_rate=0.97, training_epochs=50, \
            dataset='./data/mnist.pkl.gz', \
            batch_size=50, output_folder='CAE_results')


