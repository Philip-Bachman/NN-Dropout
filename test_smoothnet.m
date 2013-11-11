clear;

% Generate data to test basic network training
[X Y] = data_grid(5000, 0.25, 0.3, 4, 4);
[Xtr Ytr Xte Yte] = trte_split(X, Y, 250/5000);
% Note: SmoothNet expects class labels to be in +1/-1 indicator matrix form.
%       class_cats() and class_inds() convert indicator and categorical labels.
Ytr = class_inds(Ytr);
Yte = class_inds(Yte);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create the SmoothNet instance %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% layer_dims: This sets the number of nodes in each layer of the SmoothNet. The 
%             first element is the dimension of the input and the last element
%             is the dimension of the output (number of classes, usually).
%
% SmoothNet(A1, A2, A3):
%   A1: array describing the desired dimension of each network layer
%   A2: ActFunc instance to use in hidden layers (6=ReHu, a smoothed ReLU)
%   A3: ActFunc instance to use in output layer (1=linear, don't change it)
%
% NET.out_loss: Set the loss function to optimize at the output layer of the
%               SmoothNet instance NET. (Various classification and regression
%               losses are implemented in SmoothNet.m).
%
% NET.init_weights(s): Init weights using zero-mean Gaussian with stdev s. For
%                      details on weight initialization, see SmoothNet.m
%
layer_dims = [size(Xtr,2) 150 150 size(Ytr,2)];
NET = SmoothNet(layer_dims, ActFunc(6), ActFunc(1));
NET.out_loss = @(yh, y) SmoothNet.loss_mcl2h(yh, y);
NET.init_weights(0.1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set whole-network regularization parameters %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% NET.weight_noise: Scale parameter for "Gaussian fuzzing" regularizer, which
%                   corresponds to stochastically-approximated convolution of
%                   the energy surface induced by a given training set with an
%                   isotropic Gaussian distribution with the given scale. Just
%                   set this to zero if the preceding words made no sense.
%
% NET.drop_rate: Drop rate parameter for nodes in hidden layers of NET. Nodes
%                are dropped from each hidden layer with probability drop_rate
%                prior to computing gradients for each mini-batch. I plan to
%                update dropping to use different drop masks per-example rather
%                than per-batch, to improve sample coverage of subnetworks.
%
%   Note: Some code is still sitting in SmoothNet.m that lets you apply
%         DropConnect instead of DropOut, if desired. Also, when NET.drop_rate
%         is set sufficiently close to 0.5, weight-halving will be used to
%         approximate model averaging when NET.evaluate() is called.
%
% NET.drop_input: Drop rate parameter for nodes in input layer.
%
NET.weight_noise = 0.00;
NET.drop_rate = 0.10;
NET.drop_input = 0.00;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set per-layer regularization parameters %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Each layer has a structure storing regularization parameters that control its
% regularization. NET.layer_lams(i) gives the structure for layer i.
%
% layer_lams(i).lam_l1: L1 penalty applied to activations at layer i
% layer_lams(i).lam_l2: L2 penalty applied to activations at layer i
% layer_lams(i).wt_bnd: Bound on L2 norm of incoming weight vectors for each
%                       node in layer i.
% layer_lams(i).ord_lams: Vector of weights for higher-order curvature
%                         regularization, where ord_lams(i) is the weight for
%                         i'th order curvature. Either leave these unset, or set
%                         all values in each ord_lams vector to zero to apply
%                         standard DropOut training. This will also
%                         significantly accelerate training, due to a reduced
%                         number of feedforward and backprop computations.
%
% To use DropOut as in the original paper, set all regularization parameters
% here to 0, except for NET.layer_lams(i).wt_bnd, which should be selected by
% some sort of cross-validation.
%
for i=1:numel(layer_dims),
    NET.layer_lams(i).lam_l1 = 0;
    NET.layer_lams(i).lam_l2 = 0;
    NET.layer_lams(i).wt_bnd = 5.0;
end
NET.layer_lams(numel(layer_dims)).ord_lams = [0.01 0.001 0.0001];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup param struct for training the net %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% rounds: number of training rounds to perform
% start_rate: initial learning rate
% decay_rate: geometric decay to apply to learning rate
% momentum: momentum parameter for mixing gradients across updates
% batch_size: size of batches to use in batch SGD
% lam_l2: weight shrinkage (i.e. L2 regularization on inter-layer weights)
% do_validate: whether or not to perform validation
% Xv: source inputs for validation
% Yv: target outputs for validation
%
params = struct();
params.rounds = 5000;
params.start_rate = 0.1;
params.decay_rate = 0.1^(1 / params.rounds);
params.momentum = 0.95;
params.batch_size = 100;
params.lam_l2 = 1e-5;
params.do_validate = 1;
params.Xv = Xte;
params.Yv = Yte;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Train the network on training data using the given parameters %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NET.train(Xtr,Ytr,params);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot function learned by the network %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plot_netfunc(Xte(1:200,:), Yte(1:200,:), NET, 150, 0.25);








%%%%%%%%%%%%%%
% EYE BUFFER %  
%%%%%%%%%%%%%%