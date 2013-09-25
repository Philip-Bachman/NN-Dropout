classdef SmoothNet < handle
    % This class performs training of a smoothed multi-layer neural-net.
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
        % layer_bsizes gives the block size for each hidden layer
        layer_bsizes
        % layer_bcounts gives the number of blocks in each hidden layer
        layer_bcounts
        % layer_bmembs tells the members of each block in each hidden layer
        layer_bmembs
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
        layer_lams
        % out_loss gives the loss function to apply at output layer
        out_loss
        % weight_noise gives the standard deviation for additive weight noise
        weight_noise
        % drop_rate gives the rate for DropOut/DropConnect regularization
        drop_rate
        % drop_input gives a separate drop rate for the input layer
        drop_input
    end
    
    methods
        function [self] = SmoothNet(X, Y, layer_dims, act_func, out_func)
            % Constructor for SmoothNet class
            if ~exist('out_func','var')
                % Default to using linear activation transform at output layer
                out_func = ActFunc(1);
            end
            self.act_func = act_func;
            self.out_func = out_func;
            self.depth = numel(layer_dims);
            self.layer_sizes = reshape(layer_dims,1,numel(layer_dims));
            % Set loss at output layer (HSQ is a good general choice)
            self.out_loss = @(yh, y) SmoothNet.loss_mcl2h(yh, y);
            % Initialize inter-layer weights
            self.layer_weights = [];
            self.layer_bsizes = [];
            self.layer_bcounts = [];
            self.layer_bmembs = [];
            % Set blocks to contain individual nodes for now.
            b_sizes = ones(size(self.layer_sizes));
            b_counts = self.layer_sizes;
            self.init_blocks(b_sizes, b_counts, 0.1);
            % Initialize per-layer activation regularization weights
            self.layer_lams = struct();
            for i=1:self.depth,
                self.layer_lams(i).lam_l1 = 0.0;
                self.layer_lams(i).lam_l2 = 0.0;
                self.layer_lams(i).lam_grad = 0.0;
                self.layer_lams(i).lam_hess = 0.0;
            end
            % Set general global regularization weights
            self.weight_noise = 0.0;
            self.drop_rate = 0.0;
            self.drop_input = 0.0;
            return
        end
        
        function [ result ] = init_blocks(self, bsizes, bcounts, weight_scale)
            % Do a full init of the network, including block parameters and
            % edge weights.
            %
            self.set_blocks(bsizes, bcounts);
            self.init_weights(weight_scale);
            result = 1;
            return
        end
        
        function [ result ] = set_blocks(self, bsizes, bcounts)
            % Set the block sizes and counts for each layer in this net.
            % Currently, sizes other than 1 are not accepted for input layer.
            %
            self.depth = numel(bsizes);
            self.layer_sizes = zeros(1,self.depth);
            self.layer_bsizes = zeros(1,self.depth);
            self.layer_bcounts = zeros(1,self.depth);
            self.layer_bmembs = cell(1,self.depth);
            for i=1:self.depth,
                self.layer_bsizes(i) = bsizes(i);
                self.layer_bcounts(i) = bcounts(i);
                self.layer_sizes(i) = bsizes(i) * bcounts(i);
                % Compute sets of member indices for the blocks in this layer
                bmembs = zeros(bcounts(i), bsizes(i));
                for b=1:bcounts(i),
                    b_start = ((b - 1) * bsizes(i)) + 1;
                    b_end = b_start + (bsizes(i) - 1);
                    bmembs(b,:) = b_start:b_end;
                end
                self.layer_bmembs{i} = bmembs;
            end
            % Check to make sure the layer sizes implied by bsizes and bcounts
            % are concordant with previous layer sizes if they exist). If they
            % don't exist, then initialize the layer weights.
            if isempty(self.layer_weights)
                self.init_weights();
            else
                if (length(self.layer_weights) ~= (self.depth-1))
                    warning('set_blocks: contradiction with previous depth.');
                    self.init_weights();
                end
                for i=1:(self.depth-1),
                    lw = self.layer_weights{i};
                    if (((size(lw,1) - 1) ~=  self.layer_sizes(i)) || ...
                            (size(lw,2) ~= self.layer_sizes(i+1)))
                        warning('set_blocks: contradiction with layer sizes.');
                        self.init_weights();
                    end
                end
            end
            result = 1;
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
                l_acts{i} = func.feedforward(SmoothNet.bias(A), W);
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
                    prv_acts = SmoothNet.bias(l_acts{l_num-1});
                    prv_weights = l_weights{l_num-1};
                end
                if ((l_num < self.depth) && (l_num > 1))
                    % BP for internal layers (with post and pre layers)
                    func = self.act_func;
                    act_grads = l_grads{l_num};
                    nxt_weights = l_weights{l_num};
                    nxt_weights(end,:) = [];
                    nxt_grads = dN{l_num+1};
                    prv_acts = SmoothNet.bias(l_acts{l_num-1});
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
                out_acts = func.feedforward(SmoothNet.bias(out_acts), W);
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
                    if (i == self.depth)
                        % Compute a special loss/grad at output, if desired
                        [Lo dLo] = self.out_loss(A{end}, Y);
                        dN_out = bsxfun(@times, dLo, smpl_wts);
                        dN{i} = dN{i} + dN_out;
                    end
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
                    [L dLl dLr] = SmoothNet.loss_grad(Al{1},Ar{1},Al{i},Ar{i});
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
                    [L dLl dLc dLr] = SmoothNet.loss_hess(Al{1}, Ac{1}, Ar{1},...
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
                if (i == self.depth)
                    % Compute a special loss at output, if desired
                    L_out = self.out_loss(A{end},Y);
                    Ln(i) = Ln(i) + sum(L_out(:));
                end
            end
            L = sum(Ln) / obs_count;
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
                    L_l = SmoothNet.loss_grad(Al{1},Ar{1},Al{i},Ar{i});
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
                    L_l = SmoothNet.loss_hess(Al{1},Ac{1},Ar{1},Al{i},Ac{i},Ar{i});
                    % Reweight loss, e.g. for (possibly) importance sampling
                    smpl_wts = ll.lam_hess * smpl_wts;
                    L_l = bsxfun(@times, L_l, smpl_wts);
                    Ln(i) = sum(L_l(:));
                end
            end
            L = sum(Ln) / obs_count;
            return
        end
        
        function [L_gnrl L_grad L_hess L_out] = check_losses(self,X,Y,grad_len)
            % Check the various losses being optimized for this SmoothNet
            l_weights = self.layer_weights;
            % Sample triples for which to compute losses
            [Xl Xc Xr Yc smpl_wts] = ...
                SmoothNet.sample_points(X, Y, 2000, grad_len);
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
            % Compute either classification or encoding loss at the network
            % output. Determine type using the output dimension.
            out_dim = size(acts_c{end},2);
            in_dim = size(X,2);
            if (out_dim == in_dim)
                % Compute encoding error (using least squares)
                L_out = SmoothNet.loss_lsq(acts_c{end},acts_c{1});
                L_out = mean(L_out(:));
            else
                % Compute misclassification rate
                [xxx Y_i] = max(Yc,[],2);
                [xxx Yh_i] = max(acts_c{end},[],2);
                L_out = sum(Y_i ~= Yh_i) / numel(Y_i);
            end
            return
        end
            
        
        function [ result ] =  train(self, X, Y, params)
            % Do fully parameterized training for a SmoothNet
            if ~exist('params','var')
                params = struct();
            end
            params = SmoothNet.process_params(params);
            % Compute a length scale for gradient regularization.
            grad_len = SmoothNet.compute_grad_len(X, 100);
            % Setup parameters for gradient updates (rates and momentums)
            all_mom = cell(1,self.depth-1);
            for i=1:(self.depth-1),
                all_mom{i} = zeros(size(self.layer_weights{i}));
            end
            rate = params.start_rate;
            batch_size = params.batch_size;
            % Run update loop
            fprintf('Updating weights (%d rounds):\n', params.rounds);
            for e=1:params.rounds,
                % Get the droppy/fuzzy weights to use with this round
                l_weights = self.get_drop_weights(1);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Sample points and do feedforward/backprop for general loss %
                % on output layer and internal layers. Then, do feedforward  %
                % and backprop for the gradient/Hessian regularizers.        %
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                [Xl Xc Xr Yc smpl_wts] = ...
                    SmoothNet.sample_points(X, Y, batch_size, grad_len);
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
                    [L_gnrl L_grad L_hess L_class] = ...
                        self.check_losses(X, Y, grad_len);
                    if (params.do_validate == 1)
                        [Lv_gnrl Lv_grad Lv_hess Lv_class] = ...
                            self.check_losses(params.Xv,params.Yv,grad_len);
                        fprintf('    %d: tr=(%.4f, %.4f, %.4f, %.4f), te=(%.4f, %.4f, %.4f, %.4f)\n',...
                            e,L_gnrl,L_grad,L_hess,L_class,Lv_gnrl,Lv_grad,Lv_hess,Lv_class);                        
                    else
                        fprintf('    %d: t=(%.4f, %.4f, %.4f, %.4f)\n',...
                            e, L_gnrl, L_grad, L_hess, L_class);
                    end
                end
            end
            fprintf('\n');
            result = 1;
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
                dL = max(-2, min(2, dL));
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
                dL = max(-1, min(1, R));
            end
            return
        end
        
        function [ L dL ] = loss_hinge(Yh, Y)
            % Compute a multiclass L2 hinge loss and its gradients, w.r.t. the
            % proposed outputs Yh, given the true values Y.
            %
            cl_count = size(Y,2);
            [Y_max Y_idx] = max(Y,[],2);
            % Make a mask based on class membership
            c_mask = zeros(size(Y));
            for c=1:cl_count,
                c_mask(Y_idx==c,c) = 1;
            end
            Fc = Yh;
            Fp = sum(Yh .* c_mask, 2);
            margin_trans = max(bsxfun(@minus, Fc, Fp) + 1, 0);
            margin_trans = margin_trans .* (1 - c_mask);
            L = sum(margin_trans,2);
            if (nargout > 1)
                dL = double(margin_trans > 0);
                for c=1:cl_count,
                    not_c = setdiff(1:cl_count,c);
                    dL(Y_idx==c,c) = -sum(dL(Y_idx==c,not_c),2);
                end
            end
            return
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Functional norm accessory functions %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [ grad_len ] = compute_grad_len(X, sample_count)
            % Compute a length scale for gradient/hessian regularization. 
            % Sample observations at random and compute the distance to their
            % nearest neighbor. Use these nearest neighbor distances to compute
            % grad_len.
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

