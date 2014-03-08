#########################################
# Testing scripts for MNIST experiments #
#########################################

import numpy as np
import theano
import theano.tensor as T
import theano.tensor.shared_randomstreams

from FrankeNet import SS_DEV_NET
from load_data import load_udm, load_udm_ss
import NetTrainers as NT #import train_ss_mlp, train_dae

def train_ss_mlp(NET, mlp_params, sgd_params, rng, su_count=1000):
    """Run semisupervised DEV-regularized test."""

    # Load some data to train/validate/test with
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm_ss(dataset, su_count, rng)

    # Run training on the given NET
    NT.train_ss_mlp(NET=NET, \
        mlp_params=mlp_params, \
        sgd_params=sgd_params, \
        datasets=datasets)
    return 1


def train_dae(NET, dae_layer, mlp_params, sgd_params):
    """Run DAE training test."""

    # Load some data to train/validate/test with
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm(dataset)

    # Run denoising autoencoder training on the given layer of NET
    NT.train_dae(NET=NET, \
        dae_layer=dae_layer, \
        mlp_params=mlp_params, \
        sgd_params=sgd_params, \
        datasets=datasets)
    return 1

def test_dae(dae_layer, mlp_params, sgd_params):
    """Setup basic test for semisupervised DEV-regularized MLP."""

    sgd_params['batch_size'] = 50
    sgd_params['start_rate'] = 0.002

    # Initialize a random number generator for this test
    rng = np.random.RandomState(12345)

    # Construct the SS_DEV_NET object that we will be training
    x_in = T.matrix('x_in')
    NET = SS_DEV_NET(rng=rng, input=x_in, params=mlp_params)

    # Initialize biases in each net layer (except final layer) to zero
    for layer in NET.mlp_layers:
        b_init = layer.b.get_value(borrow=False)
        b_const = np.zeros(b_init.shape, dtype=theano.config.floatX)
        layer.b.set_value(b_const)

    # Run semisupervised training on the given MLP
    train_dae(NET, dae_layer, mlp_params, sgd_params)
    return 1

def test_ss_mlp(mlp_params, sgd_params, su_count=1000):
    """Setup basic test for semisupervised DEV-regularized MLP."""

    sgd_params['start_rate'] = 0.1
    sgd_params['batch_size'] = 128
    mlp_params['dev_mix_rate'] = 0.5

    # Initialize a random number generator for this test
    rng = np.random.RandomState(13579)

    # Construct the SS_DEV_NET object that we will be training
    x_in = T.matrix('x_in')
    NET = SS_DEV_NET(rng=rng, input=x_in, params=mlp_params)

    # Initialize biases in each net layer (except final layer) to small
    # positive constants (rather than their default zero initialization)
    for (num, layer) in enumerate(NET.mlp_layers):
        b_init = layer.b.get_value(borrow=False)
        b_const = np.zeros(b_init.shape, dtype=theano.config.floatX)
        if (num < (len(NET.mlp_layers)-1)):
            b_const = b_const + 0.1
        layer.b.set_value(b_const)

    # Run semisupervised training on the given MLP
    train_ss_mlp(NET, mlp_params, sgd_params, rng, su_count)
    return 1

def test_ss_mlp_pt(mlp_params, sgd_params, su_count=1000):
    """Setup basic test for semisupervised DEV-regularized MLP."""

    # Initialize a random number generator for this test
    rng = np.random.RandomState(13579)

    # Construct the SS_DEV_NET object that we will be training
    mlp_params['dev_mix_rate'] = 0.5
    x_in = T.matrix('x_in')
    NET = SS_DEV_NET(rng=rng, input=x_in, params=mlp_params)

    ######################################################
    # First, pretrain each hidden layer in NET as a DAE. #
    ######################################################
    sgd_params['batch_size'] = 100
    sgd_params['start_rate'] = 0.01
    sgd_params['epochs'] = 30
    for i in range(len(NET.mlp_layers)-1):
        print("==================================================")
        print("Pretraining hidden layer {0:d}".format(i+1))
        print("==================================================")
        train_dae(NET, i, mlp_params, sgd_params)

    #########################################################
    # Now, run supervised finetuning on the pretrained mlp. #
    #########################################################
    sgd_params['batch_size'] = 128
    sgd_params['start_rate'] = 0.02
    sgd_params['epochs'] = 500

    # Run semisupervised training on the given MLP
    train_ss_mlp(NET, mlp_params, sgd_params, rng, su_count)

    return 1

def set_params(cmd_args):
    """Construct default mlp and sgd parameter dicts."""
    # Set SGD-related parameters (and bound on net weights)
    sgd_params = {}
    sgd_params['start_rate'] = 0.1
    sgd_params['decay_rate'] = 0.998
    sgd_params['wt_norm_bound'] = 3.5
    sgd_params['epochs'] = 1000
    sgd_params['batch_size'] = 100
    # Set parameters for the network to be trained
    mlp_params = {}
    mlp_params['layer_sizes'] = [28*28, 250, 250, 11]
    mlp_params['lam_l2a'] = 1e-3
    mlp_params['dev_clones'] = 1
    mlp_params['dev_types'] = [1, 1, 2]
    mlp_params['dev_lams'] = [0.1, 0.1, 2.0]
    mlp_params['dev_mix_rate'] = 0.5
    mlp_params['use_bias'] = 1
    # Set the type of network to train, based on user input
    if (len(cmd_args) != 3):
        print "Usage: {0} [raw|sde|dev] [result_tag]".format(cmd_args[0])
        exit(1)
    elif cmd_args[1] == "raw":
        sgd_params['mlp_type'] = 'raw'
        sgd_params['result_tag'] = cmd_args[2]
        mlp_params['dev_lams'] = [0. for l in mlp_params['dev_lams']]
    elif cmd_args[1] == "sde":
        sgd_params['mlp_type'] = 'sde'
        sgd_params['result_tag'] = cmd_args[2]
    elif cmd_args[1] == "dev":
        sgd_params['mlp_type'] = 'dev'
        sgd_params['result_tag'] = cmd_args[2]
    else:
        print "I don't know how to '{0}'".format(cmd_args[1])
        exit(1)
    return (mlp_params, sgd_params)

if __name__ == '__main__':
    import sys

    # Grab command line arguments
    cmd_args = [arg for arg in sys.argv]

    # Construct MLP and SGD parameter dicts
    mlp_params, sgd_params = set_params(cmd_args)

    # Run some sort of test
    test_ss_mlp(mlp_params, sgd_params, 1000)
    #test_dae(0, mlp_params, sgd_params)
    #test_ss_mlp_pt(mlp_params, sgd_params, 1000)









##############
# EYE BUFFER #
##############
