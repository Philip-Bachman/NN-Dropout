NN-Dropout
==========

The code in this repository implements batch gradient-descent training for simple feedforward neural networks with dropout. The code is reasonably efficient, permits selection of various activation functions among hidden/output nodes, and selection of various loss functions at the output layer. In addition to the form of dropout currently promulgated by Hinton et. al, the SimpleNet class allows structured "block" dropout, in which non-overlapping adjacent blocks of hidden nodes are either dropped or retained during each round of training. This generalizes the existing approach to dropout, which occurs when each drop block comprises a single hidden node.

The code is designed to be somewhat modular, with activation and loss function implementations both separated from the actual neural network implementation. The implemented activation and loss functions support use of the SimpleNet class for training either classifiers or regressors. For multiclass data, binomial deviance loss with linear activations at the output layer currently provides the best results. Training/testing inputs should usually be processed via ZMUV prior to use, which normalizes them to zero-mean and unit-variance.

Currently, parameters accepted by the training method are not fully documented, but are generally self-explanatory for those already familiar with common approaches to training feedforward neural networks. The only novel parameter here is SimpleNet.drop_stride, which is an instance variable controlling the size of drop blocks during training.
