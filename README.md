NN-Dropout
==========

The code in this repository implements batch gradient-descent training for simple feedforward neural networks with dropout and a collection of other regularizers. The code is reasonably efficient, but should generally be considered more of a "proof-of-concept" than something "production ready". In addition to both dropout and its close sibling "DropConnect", the SmoothNet class allows regularization by first-order and second-order gradient functional norms measuring smoothness of the function induced by the network and also regularization based on Gaussian fuzzing of the network parameters. The regularizers other than dropout/dropconnect constitute research in progress.

The code is designed to be somewhat readable by people who aren't me, though the new regularizers may prove inscrutable for those not familiar with the general use of regularization in machine learning. People other than me who are not familiar with the low-level details of neural nets and machine learning in general should probably stick to working with the SmoothNet class, as the classes in the other directories are far from finished and their comments are currently lacking.

The file test_smoothnet.m in this directory will fully initialize a SmoothNet instance and run it on some synthetic data. The main purpose of this file is to show all the options/parameters that are exposed by the class. Hopefully I'll get around to explaining the various parameters at some point.
