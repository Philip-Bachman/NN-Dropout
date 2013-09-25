classdef LMNNet < handle
    % This class performs training of a LMNNN multi-layer neural-net.
    %
    
    properties
        % act_func is an ActFunc instance for computing feed-forward activation
        % levels in hidden layers and backpropagating gradients
        act_func
        % out_func is an ActFunc instance for computing feed-forward activation
        % levels at the output layer, given activations of penultimate layer.
        out_func
        % depth is the number of layers (including in/out) in this neural-net
        depth
        % layer_sizes gives the number of nodes in each layer of this net
        %   note: these sizes do _not_ include the bias
        layer_sizes
        % layer_weights is a cell array such that layer_weights{l} contains a
        % matrix in which entry (i,j) contains the weights between node i in
        % layer l and node j in layer l+1. The number of weight matrices in
        % layer_weights (i.e. its length) is self.depth - 1.
        %   note: due to biases each matrix in layer_weights has an extra row
        layer_weights
        % layer_lams is an array of structs, with each struct holding several
        % regularization parameters for each layer:
        %   lam_l1: L1 regularization on activations
        %   lam_l2: L2 regularization on activations
        %   lam_grad: L2 regularization on gradients of activations
        %   lam_hess: L2 regularization on Hessian of activations
        %   lam_lmnn: strength of "LMNN" regularization
        layer_lams
        % lam_out weights the desired loss effect at the output layer
        lam_out
        % out_loss gives the loss function to apply at output layer
        out_loss
        % out_type == 0, use X at output, out_type == 1, use Y at output
        out_type
        % weight_noise gives the standard deviation for additive weight noise
        weight_noise
        % drop_rate gives the rate for DropOut/DropConnect regularization
        drop_rate
        % drop_input gives a separate drop rate for the input layer
        drop_input
        % const_layer determines the layer to use when computing neighborhood
        % constraints for LMNN regularization.
        const_layer
    end
    
    methods
        function [self] = LMNNet(X, Y, layer_dims, act_func, out_func)
            % Constructor for LMNNet class
            if ~exist('out_func','var')
                % Default to using linear activation transform at output layer
                out_func = ActFunc(1);
            end
            self.act_func = act_func;
            self.out_func = out_func;
            self.depth = numel(layer_dims);
            self.layer_sizes = reshape(layer_dims,1,numel(layer_dims));
            % Set loss at output layer (LSQ is a good general choice)
            self.out_loss = @(yh, y) LMNNet.loss_hsq(yh, y);
            self.out_type = 0;
            % Initialize inter-layer weights
            self.layer_weights = [];
            self.init_weights(0.1);
            % Initialize per-layer activation regularization weights
            self.layer_lams = struct();
            for i=1:self.depth,
                self.layer_lams(i).lam_l1 = 0.0;
                self.layer_lams(i).lam_l2 = 0.0;
                self.layer_lams(i).lam_grad = 0.0;
                self.layer_lams(i).lam_hess = 0.0;
                self.layer_lams(i).lam_lmnn = 0.0;
            end
            % Set general global regularization weights
            self.lam_out = 1.0;
            self.weight_noise = 0.0;
            self.drop_rate = 0.0;
            self.drop_input = 0.0;
            % Set constraint layer (default to penultimate layer)
            self.const_layer = self.depth-1;
            return
        end
        
        function [ result ] = init_weights(self, weight_scale)
            % Initialize the connection weights for this neural net.
            %
            if ~exist('weight_scale','var')
                weight_scale = 0.1;
            end
            self.layer_weights = cell(1,self.depth-1);
            for i=1:(self.depth-1),
                % Add one to each outgoing layer weight count, for biases.
                pre_dim = self.layer_sizes(i)+1;
                post_dim = self.layer_sizes(i+1);
                weights = randn(pre_dim,post_dim);
                weights(end,:) = 0.1 * weights(end,:);
                if (i > 1)
                   for j=1:size(weights,2),
                       keep_count = min(25, pre_dim-1);
                       keep_idx = randsample(1:(pre_dim-1), keep_count);
                       drop_idx = setdiff(1:(pre_dim-1),keep_idx);
                       weights(drop_idx,j) = 0;
                   end
                end
                self.layer_weights{i} = weights .* weight_scale;
            end
            result = 0;
            return
        end
        
        function [ drop_weights ] = get_drop_weights(self, add_noise)
            % Effect random edge and/or node dropping by zeroing randomly
            % selected weights between each adjacent pair of layers. Also, add
            % some small white noise to weights prior to keeping/dropping.
            if ~exist('add_noise','var')
                add_noise = 0;
            end
            drop_weights = self.layer_weights;
            for i=1:(self.depth-1),
                post_weights = drop_weights{i};
                pre_nodes = size(post_weights,1);
                post_nodes = size(post_weights,2);
                if (i == 1)
                    if (self.drop_input > 1e-3)
                        %node_mask = (rand(pre_nodes,1) > self.drop_input);
                        %edge_mask = bsxfun(@times,node_mask,ones(1,post_nodes));
                        edge_mask = rand(size(post_weights)) > self.drop_input;
                        edge_mask(end,:) = 1;                        
                    else
                        edge_mask = ones(pre_nodes,post_nodes);
                    end
                else
                    if (self.drop_rate > 1e-3)
                        % Do dropout at hidden node layers
                        %node_mask = (rand(pre_nodes,1) > self.drop_rate);
                        %edge_mask = bsxfun(@times,node_mask,ones(1,post_nodes));
                        edge_mask = rand(size(post_weights)) > self.drop_rate;
                        edge_mask(end,:) = 1;
                    else
                        edge_mask = ones(size(post_weights));
                    end
                end
                if ((add_noise == 1) && (self.weight_noise > 1e-8))
                    post_noise = self.weight_noise * randn(size(post_weights));
                    drop_weights{i} = (post_weights + post_noise) .* edge_mask;
                else
                    drop_weights{i} = post_weights .* edge_mask;
                end
            end
            return
        end
        
        function [ l_acts ] = feedforward(self, X, l_weights)
            % Get per-layer activations for the observations in X, given the
            % weights in l_weights.
            %
            l_acts = cell(1,self.depth);
            l_acts{1} = X;
            for i=2:self.depth,
                if (i == self.depth)
                    func = self.out_func;
                else
                    func = self.act_func;
                end
                W = l_weights{i-1};
                A = l_acts{i-1};
                %A = A + (exprnd(0.1,size(A)) .* (A > 1e-8));
                l_acts{i} = func.feedforward(LMNNet.bias(A), W);
            end
            return
        end
        
        function [ dW dN ] = backprop(self, l_acts, l_weights, l_grads)
            % Get per-layer weight gradients, based on the given per-layer
            % activations, per-layer weights, and loss gradients at the output
            % layer (i.e., perform backprop). SUPER IMPORTANT FUNCTION!!
            %
            % Parameters:
            %   l_acts: per-layer post-transform activations
            %   l_weights: inter-layer weights w.r.t. which to gradient
            %   l_grads: per layer gradients on post-transform activations
            %            note: for backpropping basic loss on net output, only
            %                  l_grads{self.depth} will be non-zero.
            %
            % Outputs:
            %   dW: grad on each inter-layer weight (size of l_weights)
            %   dN: grad on each pre-transform activation (size of l_grads)
            %       note: dN{1} will be grads on inputs to network
            %
            dW = cell(1,self.depth-1);
            dN = cell(1,self.depth);
            for i=0:(self.depth-1),
                l_num = self.depth - i;
                if (l_num == self.depth)
                    % BP for final layer (which has no post layer)
                    func = self.out_func;
                    act_grads = l_grads{l_num};
                    nxt_weights = zeros(size(act_grads,2),1);
                    nxt_grads = zeros(size(act_grads,1),1);
                    prv_acts = LMNNet.bias(l_acts{l_num-1});
                    prv_weights = l_weights{l_num-1};
                end
                if ((l_num < self.depth) && (l_num > 1))
                    % BP for internal layers (with post and pre layers)
                    func = self.act_func;
                    act_grads = l_grads{l_num};
                    nxt_weights = l_weights{l_num};
                    nxt_weights(end,:) = [];
                    nxt_grads = dN{l_num+1};
                    prv_acts = LMNNet.bias(l_acts{l_num-1});
                    prv_weights = l_weights{l_num-1};
                end
                if (l_num == 1)
                    % BP for first layer (which has no pre layer)
                    func = self.out_func;
                    act_grads = l_grads{l_num};
                    nxt_weights = l_weights{l_num};
                    nxt_weights(end,:) = [];
                    nxt_grads = dN{l_num+1};
                    prv_acts = zeros(size(act_grads,1),1);
                    prv_weights = zeros(1,size(act_grads,2));
                end
                % Compute pre-transform bp-ed gradients for each node in the 
                % current layer.
                cur_grads = func.backprop(nxt_grads, nxt_weights, ...
                    prv_acts, prv_weights, act_grads);
                dN{l_num} = cur_grads;
                if (l_num > 1)
                    % Compute gradients w.r.t. inter-layer connection weights
                    obs_count = size(prv_acts,1);
                    dW{l_num-1} = (prv_acts' * (cur_grads ./ obs_count));
                end
            end
            return
        end
        
        function [ out_acts ] = evaluate(self, X)
            % Do a simple feed-forward computation for the inputs in X. This
            % computation performs neither dropout nor weight fuzzing. For some
            % drop rates, it approximates ensemble averaging using Hinton's
            % suggestion of weight-halving.
            %
            % Note: the "feedforward(X, weights)" function can be used to
            %       evaluate this network with droppy/fuzzy weights.
            out_acts = X;
            for i=1:(self.depth-1),
                % Select activation function for the current layer
                if (i == self.depth-1)
                    func = self.out_func;
                else
                    func = self.act_func;
                end
                % Get weights connecting current layer to previous layer
                W = self.layer_weights{i};
                if ((abs(self.drop_rate - 0.5) < 0.1) && (i > 1))
                    % Halve the weights when net was trained with dropout rate
                    % near 0.5 for hidden nodes, to approximate sampling from
                    % the implied distribution over network architectures.
                    % Weights for first layer are not halved, as they modulate
                    % inputs from observed rather than hidden nodes.
                    W = W .* 0.5;
                end
                % Compute activations at the current layer via feedforward
                out_acts = func.feedforward(LMNNet.bias(out_acts), W);
            end
            return
        end
        
        function [ dN ] = bprop_gnrl(self, A, Y, smpl_wts)
            % Do a backprop computation for the "general loss" incurred by the
            % per-layer, per-node activations A, weighted by smpl_wts. Y is a
            % matrix of targets for the output layer. By choosing Y properly,
            % either a classification or encoding loss can be applied.
            %
            dN = cell(1,self.depth);
            for i=1:self.depth,
                ll = self.layer_lams(i);
                % Compute weighted l1 and l2 regularization on the per-layer,
                % per-node activations.
                dN{i} = zeros(size(A{i}));
                if (i > 1)
                    if (ll.lam_l1 > 1e-10)
                        dN_l1 = ll.lam_l1 * bsxfun(@times,sign(A{i}),smpl_wts);
                        dN{i} = dN{i} + dN_l1;
                    end
                    if (ll.lam_l2 > 1e-10)
                        dN_l2 = ll.lam_l2 * bsxfun(@times,A{i},smpl_wts);
                        dN{i} = dN{i} + dN_l2;
                    end
                    if ((i == self.depth) && (self.lam_out > 1e-10))
                        % Compute a special loss/grad at output, if desired
                        [Lo dLo] = self.out_loss(A{end}, Y);
                        dN_out = self.lam_out * bsxfun(@times, dLo, smpl_wts);
                        dN{i} = dN{i} + dN_out;
                    end
                end
            end
            return
        end
        
        function [ dNc dNn dNf ] = bprop_lmnn(self, Ac, An, Af, smpl_wts)
            % Compute the per-layer gradients on post-transform activations
            % using the activations in Ac/An/Af, which comprise triplets of
            % activations for LMNN constraints.
            %
            dNc = cell(1,self.depth);
            dNn = cell(1,self.depth);
            dNf = cell(1,self.depth);
            for i=1:self.depth,
                ll = self.layer_lams(i);
                if ((ll.lam_lmnn > 1e-10) && (i > 1))
                    [Ll dAc dAn dAf] = LMNNet.lmnn_grads_euc(...
                        Ac{i}, An{i}, Af{i}, 1.0);
                    smpl_wts = ll.lam_lmnn * smpl_wts;
                    dNc{i} = bsxfun(@times, dAc, smpl_wts);
                    dNn{i} = bsxfun(@times, dAn, smpl_wts);
                    dNf{i} = bsxfun(@times, dAf, smpl_wts);
                else
                    dNc{i} = zeros(size(Ac{i}));
                    dNn{i} = zeros(size(An{i}));
                    dNf{i} = zeros(size(Af{i}));
                end
            end
            return
        end

        function [ dNl dNr ] = bprop_grad(self, Al, Ar, smpl_wts)
            % Compute gradients at all nodes, starting with loss values and
            % gradients for each observation at output layer. Do this for
            % each point underlying the fd-estimated gradient functionals.
            %
            % Return relavant for left and right points independently.
            %
            dNl = cell(1,self.depth);
            dNr = cell(1,self.depth);
            for i=1:self.depth,
                ll = self.layer_lams(i);
                if ((ll.lam_grad > 1e-10) && (i > 1))
                    [L dLl dLr] = LMNNet.loss_grad(Al{1},Ar{1},Al{i},Ar{i});
                    % Reweight grads, e.g. for (possibly) importance sampling
                    smpl_wts = ll.lam_grad * smpl_wts;
                    dNl{i} = bsxfun(@times, dLl, smpl_wts);
                    dNr{i} = bsxfun(@times, dLr, smpl_wts);
                else
                    dNl{i} = zeros(size(Al{i}));
                    dNr{i} = zeros(size(Ar{i}));
                end 
            end
            return
        end
        
        function [ dNl dNc dNr ] = bprop_hess(self, Al, Ac, Ar, smpl_wts)
            % Compute gradients at all nodes, starting with loss values and
            % gradients for each observation at output layer. Do this for
            % each point underlying the fd-estimated Hessian functionals.
            dNl = cell(1,self.depth);
            dNc = cell(1,self.depth);
            dNr = cell(1,self.depth);
            for i=1:self.depth,
                ll = self.layer_lams(i);
                if ((ll.lam_hess > 1e-10) && (i > 1))
                    [L dLl dLc dLr] = LMNNet.loss_hess(Al{1}, Ac{1}, Ar{1},...
                            Al{i}, Ac{i}, Ar{i});
                    % Reweight grads, e.g. for (possibly) importance sampling
                    smpl_wts = ll.lam_hess * smpl_wts;
                    dNl{i} = bsxfun(@times, dLl, smpl_wts);
                    dNc{i} = bsxfun(@times, dLc, smpl_wts);
                    dNr{i} = bsxfun(@times, dLr, smpl_wts);
                else
                    dNl{i} = zeros(size(Al{i}));
                    dNc{i} = zeros(size(Ac{i}));
                    dNr{i} = zeros(size(Ar{i}));
                end 
            end
            return
        end
        
        function [ L ] = check_gnrl_loss(self, A, Y, smpl_wts)
            % Do a loss computation for the "general loss" incurred by the
            % per-layer, per-node activations A, weighted by smpl_wts.
            %
            obs_count = size(A{1},1);
            Ln = zeros(1,self.depth);
            for i=2:self.depth,
                ll = self.layer_lams(i);
                % Compute weighted l1 and l2 loss on the per-layer, per-node
                % activations.
                A_i = A{i};
                L_l1 = ll.lam_l1 * bsxfun(@times, abs(A_i), smpl_wts);
                L_l2 = (ll.lam_l2 / 2) * bsxfun(@times, A_i.^2, smpl_wts);
                Ln(i) = sum(L_l1(:)) + sum(L_l2(:));
                if ((i == self.depth) && (self.lam_out > 1e-10))
                    % Compute a special loss at output, if desired
                    L_out = self.lam_out * self.out_loss(A{end},Y);
                    Ln(i) = Ln(i) + sum(L_out(:));
                end
            end
            L = sum(Ln) / obs_count;
            return
        end
        
        function [ L ] = check_lmnn_loss(self, Ac, An, Af, smpl_wts)
            % Compute the per-layer gradients on post-transform activations
            % using the activations in Ac/An/Af, which comprise triplets of
            % activations for LMNN constraints.
            %
            obs_count = size(Ac{1},1);
            L = 0;
            for i=2:self.depth,
                ll = self.layer_lams(i);
                if (ll.lam_lmnn > 1e-10)
                    Ll = LMNNet.lmnn_grads_euc(Ac{i}, An{i}, Af{i}, 1.0);
                    Ll = bsxfun(@times, Ll, smpl_wts);
                    L = L + ll.lam_lmnn * sum(Ll(:));
                end
            end
            L = L / obs_count;
            return
        end
        
        function [ L ] = check_grad_loss(self, Al, Ar, smpl_wts)
            % Compute grad-loss at all nodes, starting with loss values and
            % gradients for each observation at output layer. Do this for
            % each point underlying the fd-estimated gradient functionals.
            %
            obs_count = size(Al{1},1);
            Ln = zeros(1,self.depth);
            for i=2:self.depth,
                ll = self.layer_lams(i);
                if (ll.lam_grad > 1e-10)
                    L_l = LMNNet.loss_grad(Al{1},Ar{1},Al{i},Ar{i});
                    % Reweight loss, e.g. for (possibly) importance sampling
                    smpl_wts = ll.lam_grad * smpl_wts;
                    L_l = bsxfun(@times, L_l, smpl_wts);
                    Ln(i) = sum(L_l(:));
                end
            end
            L = sum(Ln) / obs_count;
            return
        end
        
        function [ L ] = check_hess_loss(self, Al, Ac, Ar, smpl_wts)
            % Compute hess-loss at all nodes, starting with loss values and
            % gradients for each observation at output layer. Do this for
            % each point underlying the fd-estimated gradient functionals.
            %
            obs_count = size(Al{1},1);
            Ln = zeros(1,self.depth);
            for i=2:self.depth,
                ll = self.layer_lams(i);
                if (ll.lam_hess > 1e-10)
                    L_l = LMNNet.loss_hess(Al{1},Ac{1},Ar{1},Al{i},Ac{i},Ar{i});
                    % Reweight loss, e.g. for (possibly) importance sampling
                    smpl_wts = ll.lam_hess * smpl_wts;
                    L_l = bsxfun(@times, L_l, smpl_wts);
                    Ln(i) = sum(L_l(:));
                end
            end
            L = sum(Ln) / obs_count;
            return
        end
        
        function [L_gnrl L_lmnn L_grad L_hess] = ...
                check_losses(self, X, Y, nc_lmnn, grad_len)
            % Check the various losses being optimized for this LMNNet
            l_weights = self.layer_weights;
            nc_count = size(nc_lmnn,1);
            % Sample a set of constraint point triples for this batch
            smpl_wts = ones(2000,1);
            nc_batch = nc_lmnn(randsample(nc_count,2000),:);
            Xc = X(nc_batch(:,1),:);
            Xl = X(nc_batch(:,2),:);
            Xr = X(nc_batch(:,3),:);
            % Compute activations for the sampled triples
            acts_c = self.feedforward(Xc, l_weights);
            acts_l = self.feedforward(Xl, l_weights);
            acts_r = self.feedforward(Xr, l_weights);
            % Get LMNN loss for this batch
            L_lmnn = self.check_lmnn_loss(acts_c,acts_l,acts_r,smpl_wts);
            % Sample triples for the non-LMNN losses
            [Xl Xc Xr Yc smpl_wts] = ...
                LMNNet.sample_points(X, Y, 2000, grad_len);
            if (self.out_type == 0)
                Yc = Xc;
            else
                Yc = Y(nc_batch(:,1),:);
            end
            % Compute activations for left/center/right points
            acts_l = self.feedforward(Xl, l_weights);
            acts_c = self.feedforward(Xc, l_weights);
            acts_r = self.feedforward(Xr, l_weights);
            % Get general loss for this batch
            L_gnrl = self.check_gnrl_loss(acts_c, Yc, smpl_wts);
            % Get gradient norm loss for this batch
            L_grad = self.check_grad_loss(acts_l, acts_r, smpl_wts);
            % Get Hessian norm loss for this batch
            L_hess = self.check_hess_loss(acts_l, acts_c, acts_r, smpl_wts);
            return
        end
            
        
        function [ result ] =  train(self, X, Y, params)
            % Do fully parameterized training for a LMNNet
            if ~exist('params','var')
                params = struct();
            end
            params = LMNNet.process_params(params);
            % Compute a length scale for gradient regularization.
            grad_len = LMNNet.compute_grad_len(X, 100);
            % Setup parameters for gradient updates (rates and momentums)
            all_mom = cell(1,self.depth-1);
            for i=1:(self.depth-1),
                all_mom{i} = zeros(size(self.layer_weights{i}));
            end
            rate = params.start_rate;
            batch_size = params.batch_size;
            % Run update loop
            fprintf('Updating weights (%d rounds):\n', params.rounds);
            nc_lmnn = zeros(1000000,3);
            nc_count = size(nc_lmnn,1);
            for i=1:3,
                nc_lmnn(:,i) = randsample(1:size(X,1),nc_count,true);
            end
            for e=1:params.rounds,
                % Get the droppy/fuzzy weights to use with this round
                l_weights = self.get_drop_weights(1);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Sample points and do feedforward/backprop for general loss %
                % on output layer and internal layers, and functional        %
                % gradient/hessian loss.                                     %
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                [Xl Xc Xr Yc smpl_wts] = ...
                    LMNNet.sample_points(X, Y, batch_size, grad_len);
                if (self.out_type == 0)
                    Yc = Xc;
                end
                % Compute activations for left/center/right points
                acts_l = self.feedforward(Xl, l_weights);
                acts_c = self.feedforward(Xc, l_weights);
                acts_r = self.feedforward(Xr, l_weights);
                % Get general per-node loss and grads for the "true" points
                dNc_gnrl = self.bprop_gnrl(acts_c, Yc, smpl_wts);
                % Compute gradients from gradient norm penalty
                [dNl_grad dNr_grad] = ...
                    self.bprop_grad(acts_l, acts_r, smpl_wts);
                % Compute gradients from Hessian norm penalty
                [dNl_hess dNc_hess dNr_hess] = ...
                    self.bprop_hess(acts_l, acts_c, acts_r, smpl_wts);
                for l=1:self.depth,
                    dNl_hess{l} = dNl_grad{l} + dNl_hess{l};
                    dNc_hess{l} = dNc_gnrl{l} + dNc_hess{l};
                    dNr_hess{l} = dNr_grad{l} + dNr_hess{l};
                end
                % Backprop per-node gradients with source activations/weights
                dLdWc = self.backprop(acts_c, l_weights, dNc_hess);
                dLdWl = self.backprop(acts_l, l_weights, dNl_hess);
                dLdWr = self.backprop(acts_r, l_weights, dNr_hess);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % First perform a series of computations to get gradients    %
                % derived from the (optional) LMNN penalties on each layer   %
                % of the network and from general losses on all layers       %
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if (e >= params.lmnn_start)
                    if ((e == params.lmnn_start) || (mod(e, 1000) == 0))
                        % Compute neighbor constraints from current embedding
                        idx = randsample(1:size(X,1),params.lmnn_count);
                        X_nc = X(idx,:);
                        Y_nc = Y(idx,:);
                        A = self.feedforward(X_nc, self.layer_weights);
                        nc_lmnn = LMNNet.neighbor_constraints(...
                            A{self.const_layer}, Y_nc, 10, 20, 0);
                        clear('A','X_nc','Y_nc');
                        nc_lmnn(:,1) = idx(nc_lmnn(:,1));
                        nc_lmnn(:,2) = idx(nc_lmnn(:,2));
                        nc_lmnn(:,3) = idx(nc_lmnn(:,3));
                        nc_count = size(nc_lmnn,1);
                    end
                    % Sample a set of constraint point triples for this batch
                    smpl_wts = ones(batch_size,1);
                    smpl_idx = randsample(nc_count,batch_size,false);
                    nc_batch = nc_lmnn(smpl_idx,:);
                    Xc = X(nc_batch(:,1),:);
                    Xl = X(nc_batch(:,2),:);
                    Xr = X(nc_batch(:,3),:);
                    % Compute activations for the sampled triples
                    acts_c = self.feedforward(Xc, l_weights);
                    acts_l = self.feedforward(Xl, l_weights);
                    acts_r = self.feedforward(Xr, l_weights);
                    % Get per-node gradients derived from LMNN losses
                    [dNc_lmnn dNl_lmnn dNr_lmnn] = ...
                        self.bprop_lmnn(acts_c, acts_l, acts_r, smpl_wts);
                    % Backprop per-node gradients with source activations/weights
                    dLdWc_lmnn = self.backprop(acts_c, l_weights, dNc_lmnn);
                    dLdWl_lmnn = self.backprop(acts_l, l_weights, dNl_lmnn);
                    dLdWr_lmnn = self.backprop(acts_r, l_weights, dNr_lmnn);
                    dLdW_lmnn = cell(1,(self.depth-1));
                    for l=1:(self.depth-1),
                        dLdW_lmnn{l} = ...
                            dLdWc_lmnn{l} + dLdWl_lmnn{l} + dLdWr_lmnn{l};
                    end
                    % Add LMNN gradients to the general and regularizer grads
                    for l=1:(self.depth-1),
                        dLdWl{l} = dLdWl{l} + dLdWl_lmnn{l};
                        dLdWc{l} = dLdWc{l} + dLdWc_lmnn{l};
                        dLdWr{l} = dLdWr{l} + dLdWr_lmnn{l};
                    end
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
                % Compact the per-weight gradients into a single array.      %
                % Then, use the array for weight updating.                   %
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                for l=1:(self.depth-1),
                    dLdWc{l} = dLdWc{l} + dLdWl{l} + dLdWr{l};
                end
                % Update inter-layer weights using the merged gradients in dWc
                for l=1:(self.depth-1),
                    % Get relevant current weights
                    l_weights = self.layer_weights{l};
                    % Add L2 per-weight gradients
                    dLdWc{l} = dLdWc{l} + (params.lam_l2 * l_weights);
                    % Mix gradient using momentum
                    dW = (params.momentum * all_mom{l}) + ...
                        ((1 - params.momentum) * dLdWc{l});
                    all_mom{l} = dW;
                    if ((sum(isnan(dW(:))) > 0) || (sum(isinf(dW(:))) > 0))
                        error('BROKEN GRADIENTS');
                    end
                    % Update weights using momentum-blended gradients
                    self.layer_weights{l} = l_weights - (rate * dW);
                end
                % Decay the learning rate after performing update
                rate = rate * params.decay_rate;
                % Occasionally recompute and display the loss and accuracy
                if ((e == 1) || (mod(e, 100) == 0))
                    [L_gnrl L_lmnn L_grad L_hess] = ...
                        self.check_losses(X, Y, nc_lmnn, grad_len);
                    fprintf('    %d: t=(%.4f, %.4f, %.4f, %.4f)\n',...
                        e, L_gnrl, L_grad, L_hess, L_lmnn);
                end
            end
            fprintf('\n');
            result = 1;
            return
        end
        
        function [ L steps ] = test_lmmnn_grads(self, Xc, Xl, Xr, W, dLdW)
            % Check the effect of the grads in dLdW on the LMNN loss for the
            % collection of LMNN observation triples in Xc/Xl/Xr. Assume the
            % weights in W were used to compute dLdW.
            %
            Ws = cell(size(W));
            steps = [0 logspace(-4,0,17)];
            L = zeros(1,numel(steps));
            for i=1:numel(steps),
                for j=1:(self.depth-1),
                    Ws{j} = W{j} - (steps(i) * dLdW{j});
                end
                Ac = self.feedforward(Xc,Ws);
                Al = self.feedforward(Xl,Ws);
                Ar = self.feedforward(Xr,Ws);
                Ls = 0;
                for k=1:(self.depth),
                    ll = self.layer_lams(k);
                    if ((ll.lam_lmnn > 1e-10) && (k > 1))
                        Lk = LMNNet.lmnn_grads_euc(Ac{k},Al{k},Ar{k},1.0);
                        Ls = Ls + mean(Lk);
                    end
                    
                end
                L(i) = Ls;
            end
            return
        end
        
        function [ L ] = test_hess(self, X, sample_count)
            % Check hessian loss near the points in X.
            %
            grad_len = LMNNet.compute_grad_len(X, 100);
            % Get sample points for computing hessian penalty
            Y = zeros(obs_count,1);
            [Xl Xc Xr] = LMNNet.sample_points(X, Y, sample_count, grad_len);
            % Compute per-layer activations for the observation pairs
            acts_l = self.feedforward(Xl, self.layer_weights);
            acts_c = self.feedforward(Xc, self.layer_weights);
            acts_r = self.feedforward(Xr, self.layer_weights);
            % Compute gradients at all nodes, starting with loss values and
            % gradients for each observation at output layer. Do this for
            % both end points of the fd-estimated functional gradients.
            Fl = acts_l{self.depth};
            Fc = acts_c{self.depth};
            Fr = acts_r{self.depth};
            L = LMNNet.loss_hess(Xl,Xc,Xr,Fl,Fc,Fr);
            return
        end
        
        function [ pl_l1_norms ] = test_sparse(self, X)
            % Check per-node activation sparsity at all layers of network.
            %
            obs_count = size(X,1);
            pl_l1_norms = zeros(self.depth,1);
            layer_acts = self.feedforward(X, self.layer_weights);
            for i=1:self.depth,
                acts = layer_acts{i};
                pl_l1_norms(i) = sum(sum(abs(acts))) / obs_count;
            end
            return
        end
    
    end
    
    %%%%%%%%%%%%%%%%%%
    % STATIC METHODS %
    %%%%%%%%%%%%%%%%%%
    
    methods (Static = true)
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Output layer loss functions %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [ L dL ] = loss_mclr(Yh, Y)
            % Compute a multiclass logistic regression loss and its gradients,
            % w.r.t. the proposed outputs Yh, given the true values Y.
            %
            cl_count = size(Y,2);
            obs_count = size(Yh,1);
            [Y_max Y_idx] = max(Y,[],2);
            P = bsxfun(@rdivide, exp(Yh), sum(exp(Yh),2));
            % Compute classification loss (deviance)
            p_idx = sub2ind(size(P), (1:obs_count)', Y_idx);
            L = -log(P(p_idx));
            if (nargout > 1)
                % Make a binary class indicator matrix
                Yi = bsxfun(@eq, Y_idx, 1:cl_count);
                % Compute the gradient of classification loss
                dL = P - Yi;
            end
            return
        end
        
        function [ L dL ] = loss_mcl2h(Yh, Y)
            % Compute a multiclass L2 hinge loss and its gradients, w.r.t. the
            % proposed outputs Yh, given the true values Y.
            %
            cl_count = size(Y,2);
            [Y_max Y_idx] = max(Y,[],2);
            % Make a class indicator matrix using +1/-1
            Yc = bsxfun(@(y1,y2) (2*(y1==y2))-1, Y_idx, 1:cl_count);
            % Compute current L2 hinge loss given the predictions in Yh
            margin_lapse = max(0, 1 - (Yc .* Yh));
            L = 0.5 * margin_lapse.^2;
            if (nargout > 1)
                % For L2 hinge loss, dL is equal to the margin intrusion
                dL = -Yc .* margin_lapse;
            end
            return
        end       
        
        function [ L dL ] = loss_lsq(Yh, Y)
            % Compute a least-sqaures regression loss and its gradients, for
            % each of the predicted outputs in Yh with true values Y.
            %
            R = Yh - Y;
            L = 0.5 * R.^2;
            if (nargout > 1)
                % Compute the gradient of least-squares loss
                dL = R;
            end
            return
        end
        
        function [ L dL ] = loss_hsq(Yh, Y)
            % Compute a "huberized" least-sqaures regression loss and its
            % gradients, for each of the predicted outputs in Yh with true
            % values Y. This loss simply transitions from L2 to L1 loss for
            % element residuals greater than 1. This helps avoid descent
            % breaking oversized gradients.
            %
            R = Yh - Y;
            L_lsq = 0.5 * R.^2;
            L_abs = abs(R) - 0.5;
            h_mask = (L_abs < 1);
            L = (L_lsq .* h_mask) + (L_abs .* (1 - h_mask));
            if (nargout > 1)
                % Compute the gradient of huberized least-squares loss
                dL = max(-2, min(2, R));
            end
            return
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Functional norm accessory functions %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [ grad_len ] = compute_grad_len(X, sample_count)
            % Compute a length scale for gradient/hessian regularization. 
            % Sample random pairs of observations and set the length to a
            % function of some approximate quantile of the pairwise
            % inter-observation distances.
            %
            obs_count = size(X,1);
            dists = zeros(sample_count,1);
            for i=1:sample_count,
                idx = randi(obs_count);
                x1 = X(idx,:);
                dx = sqrt(sum(bsxfun(@minus,X,x1).^2,2));
                dx(idx) = max(dx) + 1;
                dists(i) = min(dx);
            end
            grad_len = median(dists) / 2;
            return
        end
        
        function [ L dLl dLr ] = loss_grad(Xl, Xr, Fl, Fr)
            % Compute approximate functional gradient loss for the point tuples
            % (Xl, Xr) with outputs (Fl, Fr). Loss is equal to the square of the
            % finite-differences-estimated first-order (directional)
            % derivatives, with gradients computed accordingly.
            %
            d_norms = sqrt(sum((Xl - Xr).^2,2));
            d_scale = d_norms.^2;
            f_diffs = Fl - Fr;
            % Compute squared (fd-based) directional derivatives
            L = bsxfun(@rdivide, f_diffs.^2, 2*d_scale);
            if (nargout > 1)
                % Compute gradients
                dLl = bsxfun(@rdivide, f_diffs, d_scale);
                dLr = bsxfun(@rdivide, -f_diffs, d_scale);
                % Clip gradients
                dLl = max(-2, min(dLl, 2));
                dLr = max(-2, min(dLr, 2));
            end
            return
        end
        
        function [ L dLl dLc dLr ] = loss_hess(Xl, Xc, Xr, Fl, Fc, Fr)
            % Compute approximate functional gradient loss for the point tuples
            % (Xl, Xc, Xr) with outputs (Fl, Fc, Fr). Loss is equal to the
            % square of the finite-differences-estimated second-order
            % (directional) derivatives, with gradients computed accordingly.
            %
            d_norms = 0.5 * (sqrt(sum((Xl-Xc).^2,2)) + sqrt(sum((Xr-Xc).^2,2)));
            d_scale = (d_norms.^2).^2; % weirdly written, for semantics
            f_diffs = (Fl + Fr) - (2 * Fc);
            % Compute squared (fd-based) directional derivatives
            L = bsxfun(@rdivide, f_diffs.^2, 2*d_scale);
            if (nargout > 1)
                % Compute gradients
                dLl = bsxfun(@rdivide, f_diffs, d_scale);
                dLr = bsxfun(@rdivide, f_diffs, d_scale);
                dLc = bsxfun(@rdivide, -2*f_diffs, d_scale);
                % Clip gradients
                dLl = max(-2, min(dLl, 2));
                dLr = max(-2, min(dLr, 2));
                dLc = max(-2, min(dLc, 2));
            end
            return
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % LMNN accessory Functions %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [ I_nn ] = knn_ind( Xte, Xtr, k, do_loo, do_dot )
            % Find indices of the knn for points in Xte measured with respect to
            % the points in Xtr.
            %
            % (do_loo == 1) => do "leave-one-out" knn, assuming Xtr == Xte
            % (do_dot == 1) => use max dot-products instead of min euclideans.
            %
            if ~exist('do_loo','var')
                do_loo = 0;
            end
            if ~exist('do_dot','var')
                do_dot = 0;
            end
            obs_count = size(Xte,1);
            I_nn = zeros(obs_count,k);
            fprintf('Computing knn:');
            for i=1:obs_count,
                if (mod(i,round(obs_count/50)) == 0)
                    fprintf('.');
                end
                if (do_dot == 1)
                    d = Xtr * Xte(i,:)';
                    if (do_loo == 1)
                        d(i) = min(d) - 1;
                    end
                    [d_srt i_srt] = sort(d,'descend');
                else
                    d = sum(bsxfun(@minus,Xtr,Xte(i,:)).^2,2);
                    if (do_loo == 1)
                        d(i) = max(d) + 1;
                    end
                    [d_srt i_srt] = sort(d,'ascend');
                end
                I_nn(i,:) = i_srt(1:k);
            end
            fprintf('\n');
            return
        end
        
        function [ n_const ] = neighbor_constraints( X, Y, k_in, k_out, do_dot )
            % Generate "neighbor constraints" with which to train a metric.
            if ~exist('do_dot','var')
                do_dot = 0;
            end
            [xxx Y] = max(Y,[],2);
            I_nn = LMNNet.knn_ind(X, X, (3*(k_in+k_out)), 1, do_dot);
            o_count = size(X,1);
            n_const = zeros(o_count*k_in*k_out,3);
            idx_c = 1;
            for i=1:size(X,1),
                % Get indices of in-class and out-class nns for X(i,:)
                knn_i = I_nn(i,:);
                idx_i = knn_i(Y(knn_i) == Y(i));
                idx_o = knn_i(Y(knn_i) ~= Y(i));
                % Use the in/out nn indices to compute neighbor constraints
                for j_i=1:min(k_in,numel(idx_i)),
                    for j_o=1:min(k_out,numel(idx_o)),
                        n_const(idx_c,1) = i;
                        n_const(idx_c,2) = idx_i(j_i);
                        n_const(idx_c,3) = idx_o(j_o);
                        idx_c = idx_c + 1;
                    end
                end
            end
            n_const = n_const(1:(idx_c-1),:);
            % For diagnostic purposes, compute knn error for points in X/Y
            Y_nn = zeros(size(I_nn));
            for i=1:size(I_nn,2),
                Y_nn(:,i) = Y(I_nn(:,i));
            end
            knn_err = mean(bsxfun(@eq, Y_nn ,Y));
            fprintf('    knn error: %.4f, %.4f, %.4f, %.4f, %.4f\n', ...
                knn_err(1), knn_err(2), knn_err(3), knn_err(4), knn_err(5));
            return
        end

        function [ L dXc dXn dXf ] = lmnn_grads_euc( Xc, Xn, Xf, margin )
            % Compute gradients of standard LMNN using Euclidean distance.
            %
            % In addition to the standard penalty on margin transgression by
            % impostor neighbors, impose an attractive penalty on distance between
            % true neighbors and a repulsive penalty between false neighbors.
            %
            % Parameters:
            %   Xc: central/source points
            %   Xn: points that should be closer to those in Xc
            %   Xf: points that should be further from those in Xc
            %   margin: desired margin between near/far distances w.r.t. Xc
            % Outputs:
            %   L: loss for each LMNN triplet (Xc(i,:),Xn(i,:),Xf(i,:))
            %   dXc: gradient of L w.r.t. Xc
            %   dXn: gradient of L w.r.t. Xn
            %   dXf: gradient of L w.r.t. Xf
            %

            On = Xn - Xc;
            Of = Xf - Xc;
            % Compute (squared) norms of the offsets On/Of
            Dn = sum(On.^2,2);
            Df = sum(Of.^2,2);
            % Add a penalty on neighbor distances > margin.
            yes_pen = max(0, Dn - margin);
            % Add a penalty on non-neighbor distances < margin.
            non_pen = max(0, margin - Df);
            % Get losses and indicators for violated LMNN constraints
            m_viol = max(0, (Dn - Df) + margin);
            L = (0.5 * m_viol) + (0.5 * yes_pen) + (0.5 * non_pen);
            % Compute gradients for violated constraints
            dXn = bsxfun(@times,On,(m_viol > 1e-10)) + ...
                bsxfun(@times,On,(yes_pen > 1e-10));
            dXf = bsxfun(@times,-Of,(m_viol > 1e-10)) + ...
                bsxfun(@times,-Of,(non_pen > 1e-10));
            dXc = -dXn - dXf;
            % Clip gradients
            dXc = max(-2,min(dXc, 2));
            dXn = max(-2,min(dXn, 2));
            dXf = max(-2,min(dXf, 2));
            return
        end
        
        function [ L dXc dXn dXf ] = lmnn_grads_dot( Xc, Xn, Xf, margin )
            % Compute gradients of standard LMNN using dot-product distance.
            %
            % Parameters:
            %   Xc: central/source points
            %   Xn: points that should be closer to those in Xc
            %   Xf: points that should be further from those in Xc
            %   margin: desired margin between near/far distances w.r.t. Xc
            % Outputs:
            %   L: loss for each LMNN triplet (Xc(i,:),Xn(i,:),Xf(i,:))
            %   dXc: gradient of L w.r.t. Xc
            %   dXn: gradient of L w.r.t. Xn
            %   dXf: gradient of L w.r.t. Xf
            %
            % Compute dot-products between center/near and center/far points
            Dn = sum(Xc .* Xn,2);
            Df = sum(Xc .* Xf,2);
            % Get losses and indicators for violated LMNN constraints
            m_viol = max(0, (Df - Dn) + margin);
            m_mask = m_viol > 1e-10;
            L = m_viol;
            % Compute gradients for violated constraints
            dXn = bsxfun(@times, -Xc, m_mask);
            dXf = bsxfun(@times, Xc, m_mask);
            dXc = bsxfun(@times, (Xf - Xn), m_mask);
            % Clip gradients
            dXc = max(-2, min(dXc, 2));
            dXn = max(-2, min(dXn, 2));
            dXf = max(-2, min(dXf, 2));
            return
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % HAPPY FUN BONUS FUNCTIONS %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [ Xb ] = bias(X)
            % Add a column of constant bias to the observations in X
            Xb = [X ones(size(X,1),1)];
            return
        end
        
        function [ Xl Xc Xr Yc smpl_wts ] = sample_points(X, Y, smpls, grd_len)
            % Sample a batch of training observations and compute gradients
            % for gradient and hessian parts of loss.
            obs_count = size(X,1);
            idx = randsample(1:obs_count, smpls, true);
            Xc = X(idx,:);
            Yc = Y(idx,:);
            Xd = randn(size(Xc));
            Xd = bsxfun(@rdivide, Xd, sqrt(sum(Xd.^2,2))+1e-8);
            Xd = Xd .* grd_len;
            Xl = Xc - (Xd ./ 2);
            Xr = Xc + (Xd ./ 2);
            smpl_wts = ones(smpls,1);
            return
        end
        
        function [ params ] = process_params(params)
            % Process parameters to use in training of some sort.
            if ~isfield(params, 'rounds')
                params.rounds = 10000;
            end
            if ~isfield(params, 'lmnn_start')
                params.lmnn_start = 1000;
            end
            if ~isfield(params, 'lmnn_count')
                params.lmnn_start = 5000;
            end
            if ~isfield(params, 'start_rate')
                params.start_rate = 1.0;
            end
            if ~isfield(params, 'decay_rate')
                params.decay_rate = 0.995;
            end
            if ~isfield(params, 'momentum')
                params.momentum = 0.5;
            end
            if ~isfield(params, 'lam_l2')
                params.lam_l2 = 1e-5;
            end
            if ~isfield(params, 'batch_size')
                params.batch_size = 100;
            end
            if ~isfield(params, 'do_validate')
                params.do_validate = 0;
            end
            if (params.do_validate == 1)
                if (~isfield(params, 'Xv') || ~isfield(params, 'Yv'))
                    error('Validation set required for doing validation.');
                end
            end
            % Clip momentum to be in range [0...1]
            params.momentum = min(1, max(0, params.momentum));
            return
        end
        
    end 
    
end










%%%%%%%%%%%%%%
% EYE BUFFER %
%%%%%%%%%%%%%%

