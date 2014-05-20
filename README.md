NN-Dropout
==========

*** 
Note:
This should be considered deprecated, or whatever.
Everything I'm actually actively working with is in my NN-Python repository.
*** 



The code in this repository implements batch gradient-descent training for feedforward neural networks with dropout and a collection of other regularizers. The code is reasonably efficient, but should generally be considered more of a "proof-of-concept" than something "production ready". In addition to dropout and its close sibling "DropConnect", the SmoothNet class allows regularization by finite differences estimates of higher-order curvature in the function induced by the network, and regularization based on Gaussian fuzzing of the network parameters. The regularizers other than dropout/dropconnect constitute research in progress. A SmoothNet instance will train faster if curvature regularization for each layer is turned off (see test_smoothnet.m for more info). I've also written a "rectified Huber" activation function, which provides a smooth alternative to the rectified-linear activation function. The smooth ReHu is better suited to curvature regularization than the herky-jerky ReLU.

The code is designed to be somewhat readable by people who aren't me, though the new regularizers may prove inscrutable to those not familiar with the general use of regularization in machine learning. People who are not familiar with the low-level details of neural nets and machine learning in general should probably stick to working with the SmoothNet class, as classes in the other directories are far from finished and their comments/documentation are currently lacking.

The file test_smoothnet.m in this directory will fully initialize a SmoothNet instance and run it on some synthetic data. The main purpose of this file is to show all the options/parameters that are exposed by the class. The file includes documentation describing all parameters relevant to initializing and training a SmoothNet instance.

I'm now working actively on this project, so updates should be more frequent.
