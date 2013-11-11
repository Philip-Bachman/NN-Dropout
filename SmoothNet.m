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
        %   ord_lams: regularization on curvature of orders >= 1
        %   wt_bnd: L2 ball bound on norms of incoming weights
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
        function [self] = SmoothNet(layer_dims, act_func, out_func)
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
                self.layer_lams(i).ord_lams = [0];
                self.layer_lams(i).wt_bnd = 10;
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
                if ((i > 1) && (i < (self.depth - 1)))
                   for j=1:size(weights,2),
                       keep_count = min(50, pre_dim-1);
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
                        % Do dropout for input layer
                        drop_nodes = randsample(pre_nodes, ...
                            floor(pre_nodes * self.drop_input));
                        node_mask = ones(pre_nodes,1);
                        node_mask(drop_nodes) = 0;
                        %node_mask = (rand(pre_nodes,1) > self.drop_input);
                        edge_mask = bsxfun(@times,node_mask,ones(1,post_nodes));
                        %edge_mask = rand(size(post_weights)) > self.drop_input;
                        edge_mask(end,:) = 1;                        
                    else
                        edge_mask = ones(pre_nodes,post_nodes);
                    end
                else
                    if (self.drop_rate > 1e-3)
                        % Do dropout for hidden layer
                        drop_nodes = randsample(pre_nodes, ...
                            floor(pre_nodes * self.drop_rate));
                        node_mask = ones(pre_nodes,1);
                        node_mask(drop_nodes) = 0;
                        %node_mask = (rand(pre_nodes,1) > self.drop_rate);
                        edge_mask = bsxfun(@times,node_mask,ones(1,post_nodes));
                        %edge_mask = rand(size(post_weights)) > self.drop_rate;
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
        
        function [ max_ord ] = max_order(self)
            % Determine the maximum order of penalized curvature based on
            % current self.layer_lams.
            max_ord = 0;
            for i=1:self.depth,
                max_ord = max(max_ord, length(self.layer_lams(i).ord_lams));
            end
        end
        
        function [ max_olam ] = max_olam(self)
            % Determine the maximum order of penalized curvature based on
            % current self.layer_lams.
            max_olam = 0;
            for i=1:self.depth,
                max_olam = max(max_olam, max(self.layer_lams(i).ord_lams));
            end
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
                A_pre = l_acts{i-1};
                A_cur = func.feedforward(SmoothNet.bias(A_pre), W);
                l_acts{i} = A_cur;
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
        
        function [ dN L ] = bprop_gnrl(self, A, Y, smpl_wts)
            % Do a backprop computation for the "general loss" incurred by the
            % per-layer, per-node activations A, weighted by smpl_wts. Y is a
            % matrix of targets for the output layer. By choosing Y properly,
            % either a classification or encoding loss can be applied.
            %
            dN = cell(1,self.depth);
            L = zeros(size(A{1},1),1);
            for i=1:self.depth,
                ll = self.layer_lams(i);
                % Compute weighted l1 and l2 regularization on the per-layer,
                % per-node activations.
                dN{i} = zeros(size(A{i}));
                if (i > 1)
                    if (ll.lam_l1 > 1e-10)
                        dN_l1 = ll.lam_l1 * bsxfun(@times,sign(A{i}),smpl_wts);
                        dN{i} = dN{i} + dN_l1;
                        L = L + (ll.lam_l1 * (sum(sign(A{i}),2) .* smpl_wts));
                    end
                    if (ll.lam_l2 > 1e-10)
                        dN_l2 = ll.lam_l2 * bsxfun(@times,A{i},smpl_wts);
                        dN{i} = dN{i} + dN_l2;
                        L = L + ((ll.lam_l2 / 2) * ...
                            (sum(A{i}.^2,2) .* smpl_wts));
                    end
                    if (i == self.depth)
                        % Compute a special loss/grad at output, if desired
                        [Lo dLo] = self.out_loss(A{end}, Y);
                        dN_out = bsxfun(@times, dLo, smpl_wts);
                        dN{i} = dN{i} + dN_out;
                        L = L + sum(Lo,2);
                    end
                end
            end
            return
        end

        function [ dN_fd L_fd ] = bprop_fd(self, A_fd, fd_lens, smpl_wts)
            % Compute gradients at all nodes, starting with loss values and
            % gradients for each observation at output layer. Do this for
            % each point underlying the fd-estimated gradient functionals.
            %
            % Return cell array of gradients on activations for each point in
            % each fd chain independently.
            %
            % length(A_fd): self.max_order() + 1 (well, it should be)
            % length(A_fd{1}): self.depth (well, it should be)
            %
            L_fd = cell(1,self.max_order());
            for j=1:length(L_fd),
                L_fd{j} = zeros(size(A_fd{1}{1},1),1);
            end
            dN_fd = cell(1,length(A_fd));
            for j=1:length(A_fd),
                dN_fd{j} = cell(1,length(A_fd{j}));
                for i=1:self.depth,
                    dN_fd{j}{i} = zeros(size(A_fd{j}{i}));
                end
            end
            for i=1:self.depth,
                olams = self.layer_lams(i).ord_lams;
                if ((max(olams) > 1e-8) && (i > 1))
                    A_out = cell(1,length(A_fd));
                    for j=1:length(A_fd),
                        A_out{j} = A_fd{j}{i};
                    end
                    [L_fdi dN_fdi] = ...
                        SmoothNet.loss_fd_huber(A_out,fd_lens,olams,smpl_wts);
                    for j=1:length(A_fd),
                        dN_fd{j}{i} = dN_fdi{j};
                    end
                    for j=1:numel(olams),
                        L_fd{j} = L_fd{j} + L_fdi{j};
                    end
                end
            end
            return
        end
        
        function [L_gnrl L_curv L_out] = check_losses(self, X, Y, nn_len)
            % Check the various losses being optimized for this SmoothNet
            l_weights = self.layer_weights;
            % Sample points for general loss
            [Xg Yg smpl_wts] = SmoothNet.sample_points(X, Y, 2000);
            % Compute activations for sampled points
            acts_g = self.feedforward(Xg, l_weights);
            % Compute general loss for sampled points
            [dN_gnrl L_gnrl] = self.bprop_gnrl(acts_g, Yg, smpl_wts);
            L_gnrl = sum(L_gnrl(:)) / numel(L_gnrl);
            if (self.max_olam() > 1e-10)
                % Sample fd chains for cuvrvature loss
                [X_fd fd_lens] = SmoothNet.sample_fd_chains(X,...
                    2000, self.max_order(), nn_len, (nn_len/4));
                % Compute activations for points in each FD chain
                acts_fd = cell(1,length(X_fd));
                for i=1:length(X_fd),
                    acts_fd{i} = self.feedforward(X_fd{i}, l_weights);
                end
                % Compute multi-order curvature gradients for the FD chains
                [dN_fd L_fd] = self.bprop_fd(acts_fd, fd_lens, smpl_wts);
                L_curv = zeros(size(L_fd{1},1),1);
                for i=1:length(L_fd),
                    L_curv = L_curv + L_fd{i};
                end
                L_curv = sum(L_curv(:)) / numel(L_curv);
            else
                L_curv = 0;
            end
            % Compute either classification or encoding loss at the network
            % output. Determine type using the output dimension.
            Fg = self.evaluate(Xg);
            out_dim = size(Fg,2);
            in_dim = size(Xg,2);
            if (max(Y(:)) ~= 1)
                % Compute encoding error (using least squares)
                L_out = SmoothNet.loss_lsq(Fg,Xg);
                L_out = mean(L_out(:));
            else
                % Compute misclassification rate
                [xxx Y_i] = max(Yg,[],2);
                [xxx Yh_i] = max(Fg,[],2);
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
            nn_len = SmoothNet.compute_nn_len(X, 500);
            nn_len = max(nn_len,0.1);
            % Setup parameters for gradient updates (rates and momentums)
            all_mom = cell(1,self.depth-1);
            for i=1:(self.depth-1),
                all_mom{i} = zeros(size(self.layer_weights{i}));
            end
            rate = params.start_rate;
            batch_size = params.batch_size;
            dldw_norms = zeros(self.depth-1,params.rounds);
            lwts_norms = zeros(self.depth-1,params.rounds);
            max_sratio = zeros(self.depth-1,params.rounds);
            max_ratios = zeros(1,params.rounds);
            % Run update loop
            fprintf('Updating weights (%d rounds):\n', params.rounds);
            for e=1:params.rounds,
                % Get the droppy/fuzzy weights to use with this round
                l_weights = self.get_drop_weights(1);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Sample points and do feedforward/backprop for general loss %
                % on output layer and internal layers.                       %
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                [Xg Yg smpl_wts] = SmoothNet.sample_points(X, Y, batch_size);
                % Compute activations for center points
                acts_g = self.feedforward(Xg, l_weights);
                % Get general per-node loss and grads for the "true" points
                dNc_gnrl = self.bprop_gnrl(acts_g, Yg, smpl_wts);
                % Backprop per-node gradients for general loss
                dLdW = self.backprop(acts_g, l_weights, dNc_gnrl);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Sample FD chains and do feedforward/backprop for curvature  %
                % regularization across multiple orders of curvature.         %
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if (self.max_olam() > 1e-10)
                    l_weights = self.layer_weights;
                    [X_fd fd_lens] = SmoothNet.sample_fd_chains(X,...
                        batch_size, self.max_order(), nn_len, (nn_len/4));
                    smpl_wts = ones(batch_size,1);
                    % Compute activations for points in each FD chain
                    acts_fd = cell(1,length(X_fd));
                    for o=1:length(X_fd),
                        acts_fd{o} = self.feedforward(X_fd{o}, l_weights);
                    end
                    % Compute multi-order curvature gradients for the FD chains
                    dN_fd = self.bprop_fd(acts_fd, fd_lens, smpl_wts);
                    % Backprop multi-order curvature gradients at each FD point
                    dLdW_fd = cell(1,length(acts_fd));
                    for o=1:length(acts_fd),
                        dLdW_fd{o} = ...
                            self.backprop(acts_fd{o}, l_weights, dN_fd{o});
                    end
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
                    % Compact the per-weight gradients into a single array.   %
                    % Then, use the array for weight updating.                %
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    for l=1:(self.depth-1),
                        for o=1:length(dLdW_fd),
                            dLdW{l} = dLdW{l} + dLdW_fd{o}{l};
                        end
                    end
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Compute updates for inter-layer weights using grads in dLdW %
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                layer_dW = cell(1,(self.depth-1));
                for l=1:(self.depth-1),
                    % Get relevant current weights and proposed update
                    l_weights = self.layer_weights{l};
                    l_dLdW = dLdW{l};
                    % Add L2 per-weight gradients
                    l_dLdW = l_dLdW + (params.lam_l2 * l_weights);
                    % Apply momentum transform
                    l_dLdW = (params.momentum * all_mom{l}) + ...
                        ((1 - params.momentum) * l_dLdW);
                    all_mom{l} = l_dLdW;
                    % Compute norm of weights and gradient, for diagnostics
                    lwts_norms(l,e) = sqrt(sum(sum(l_weights.^2)));
                    dldw_norms(l,e) = sqrt(sum(sum(l_dLdW.^2)));
                    max_sratio(l,e) = max(...
                        sqrt(sum(l_dLdW.^2,1)) ./ sqrt(sum(l_weights.^2,1))+1e-3);
                    layer_dW{l} = l_dLdW;
                    if ((sum(isnan(l_dLdW(:)))>0) || (sum(isinf(l_dLdW(:)))>0))
                        error('BROKEN GRADIENTS');
                    end
                end
                max_ratios(e) = max(max_sratio(:,e));
                % Apply updates to inter-layer weights
                for l=1:(self.depth-1),
                    dW = layer_dW{l};
                    scale = min(1, (1/max_ratios(e)));
                    dW = dW .* scale;
                    l_weights = self.layer_weights{l};
                    % Update weights using momentum-blended gradients
                    l_weights = l_weights - (rate * dW);
                    % Clip incoming weights for each node to bounded L2 ball.
                    wt_bnd = self.layer_lams(l+1).wt_bnd;
                    wt_scales = wt_bnd ./ sqrt(sum(l_weights.^2,1));
                    l_weights = bsxfun(@times, l_weights, min(wt_scales,1));
                    % Store updated weights
                    self.layer_weights{l} = l_weights;
                end
                % Decay the learning rate after performing update
                rate = rate * params.decay_rate;
                % Occasionally recompute and display the loss and accuracy
                if ((e == 1) || (mod(e, 100) == 0))
                    [L_gnrl L_curv L_class] = ...
                        self.check_losses(X, Y, nn_len);
                    if (params.do_validate == 1)
                        [Lv_gnrl Lv_curv Lv_class] = ...
                            self.check_losses(params.Xv,params.Yv,nn_len);
                        fprintf('    %d: tr=(%.4f, %.4f, %.4f), te=(%.4f, %.4f, %.4f)\n',...
                            e,L_gnrl,L_curv,L_class,Lv_gnrl,Lv_curv,Lv_class);                        
                    else
                        fprintf('    %d: t=(%.4f, %.4f, %.4f)\n',...
                            e, L_gnrl, L_curv, L_class);
                    end
                end
            end
            fprintf('\n');
            result = struct();
            result.lwts_norms = lwts_norms;
            result.dldw_norms = dldw_norms;
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
            % Compute a multiclass L1 hinge loss and its gradients, w.r.t. the
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
        
        function [ L_fd dL_fd ] = loss_fd(A_out,fd_lens,ord_lams,smpl_wts)
            % This computes gradients for curvature loss of multiple orders,
            % given the fd chain points in A_in, and their associated output
            % values in A_out. 
            %
            % length(A_out) should equal numel(ord_lams)+1
            %
            % A_out{i} should be give sampled outputs for the ith link in some
            % fd chain.
            %
            % ord_lams gives the weightings for curvature order penalties
            %
            % fd_lens gives the step size of the underlying fd chains
            %
            % smpl_wts gives the relative importance of each fd chain
            %
            % L_fd{i} holds the vector of losses for the fd subchain of order i
            %
            % dL_fd{i} holds the matrix of gradients on the function outputs in
            % A_out{i}.
            %
            L_fd = cell(1,numel(ord_lams));
            dL_fd = cell(1,length(A_out));
            for i=1:length(A_out),
                dL_fd{i} = zeros(size(A_out{i}));
            end
            % Compute loss and gradient for each fd chain, for each order
            for i=1:numel(ord_lams),
                olam = ord_lams(i);
                fd_coeffs = zeros(1,(i+1));
                for j=0:i,
                    fd_coeffs(j+1) = (-1)^j * nchoosek(i,j);
                end
                fd_diffs = zeros(size(A_out{1}));
                for j=1:(i+1),
                    fd_diffs = fd_diffs + (fd_coeffs(j) * A_out{j});
                end
                fd_ls = fd_lens.^i; %ones(size(fd_lens.^i));
                L_fd{i} = olam * ...
                    (sum(bsxfun(@rdivide,fd_diffs.^2,fd_ls.^2),2) .* smpl_wts);
                for j=1:(i+1),
                    dL_j = bsxfun(@rdivide,(fd_coeffs(j)*fd_diffs),(fd_ls.^2));
                    dL_fd{j} = dL_fd{j} + ...
                        ((2 * olam) * bsxfun(@times, dL_j, smpl_wts));
                end
            end
            return
        end
        
        function [L_fd dL_fd] = loss_fd_huber(A_out,fd_lens,ord_lams,smpl_wts)
            % This computes gradients for curvature loss of multiple orders,
            % given the fd chain points in A_in, and their associated output
            % values in A_out. 
            %
            % General idea is same as for SmoothNet.loss_fd, except a Huberized
            % loss is applied to finite differences, rather than a pure squared
            % loss. This helps mitigate the strong "outlier" effect that occurs
            % for FDs of higher-order curvature, due to tiny denominators.
            %
            if ~exist('hub_thresh','var')
                hub_thresh = 2.0;
            end 
            L_fd = cell(1,numel(ord_lams));
            dL_fd = cell(1,length(A_out));
            for i=1:length(A_out),
                dL_fd{i} = zeros(size(A_out{i}));
            end
            % Compute loss and gradient for each fd chain, for each order
            for i=1:numel(ord_lams),
                olam = ord_lams(i);
                fd_coeffs = zeros(1,(i+1));
                for j=0:i,
                    fd_coeffs(j+1) = (-1)^j * nchoosek(i,j);
                end
                fd_diffs = zeros(size(A_out{1}));
                for j=1:(i+1),
                    fd_diffs = fd_diffs + (fd_coeffs(j) * A_out{j});
                end
                fd_ls = fd_lens.^i;
                fd_vals = bsxfun(@rdivide,fd_diffs,fd_ls);
                % Get masks for FD values subject to quadratic/linear losses
                quad_mask = bsxfun(@lt, abs(fd_vals), hub_thresh);
                line_mask = bsxfun(@ge, abs(fd_vals), hub_thresh);
                % Compute quadratic and linear parts of FD loss
                quad_loss = fd_vals.^2;
                quad_loss = quad_loss .* quad_mask;
                line_loss = (((2*hub_thresh) * abs(fd_vals)) - hub_thresh^2);
                line_loss = line_loss .* line_mask;
                L_fd{i} = sum(quad_loss,2) + sum(line_loss,2);
                L_fd{i} = olam * (L_fd{i} .* smpl_wts);
                for j=1:(i+1),
                    dL_quad = 2*bsxfun(@rdivide,(fd_coeffs(j)*fd_vals),fd_ls);
                    dL_line = 2*fd_coeffs(j)*sign(fd_vals);
                    dL_j = (dL_quad .* quad_mask) + (dL_line .* line_mask);
                    dL_fd{j} = dL_fd{j} + (olam * bsxfun(@times,dL_j,smpl_wts));
                end
            end
            return
        end
        
        function [ X_fd fd_lens ] = sample_fd_chains(X, chain_count,...
                max_order, fuzz_len, fd_len, bias)
            % Sample chains of points for forward FD estimates of directional 
            % higher-order derivatives. The anchor point for each sampled chain
            % is sampled from a "fuzzed" version of the empirical distribution
            % described by the points in X.
            %
            % Sample chain directions from a uniform distribution over the
            % surface of a hypersphere of dimension size(X,2). Then, rescale
            % these directions based on fd_len. If given, apply the "biasing"
            % transform (a matrix) to the directions prior to setting lengths.
            %
            if ~exist('bias','var')
                bias = eye(size(X,2));
            end
            if ~exist('strict_len','var')
                strict_len = 1;
            end
            chain_len = max_order + 1;
            % Sample points from the empirical distribution described by X,
            % convolved with the isotropic distribution with a length
            % distribution given by a Gaussian with std dev fuzz_len.
            s_idx = randsample(size(X,1),chain_count,true);
            Xs = X(s_idx,:);
            Xd = randn(size(Xs));
            Xd = bsxfun(@rdivide, Xd, max(sqrt(sum(Xd.^2,2)),1e-8));
            Xd = bsxfun(@times, Xd, (fuzz_len * abs(randn(size(Xd,1),1))));
            Xs = Xs + Xd;
            % Sample (biased & scaled) displacement directions for each chain.
            Xd = randn(size(Xs));
            Xd = Xd * bias;
            Xd = bsxfun(@rdivide, Xd, max(sqrt(sum(Xd.^2,2)),1e-8));
            if (strict_len ~= 1)
                % Sample fd lengths from a scaled abs(normal) distribution
                fd_lens = (fd_len/2) * abs(randn(size(Xd,1),1));
                fd_lens = fd_lens + fd_len;
            else
                % Set fd lengths strictly (i.e. to a fixed value)
                fd_lens = fd_len * ones(size(Xd,1),1);
            end
            % Scale sampled displacement directions using sampled fd lengths
            Xd = bsxfun(@times, Xd, fd_lens);
            % Construct forward chains by stepwise displacement
            X_fd = cell(1,chain_len);
            for i=0:(chain_len-1),
                X_fd{i+1} = Xs + (i * Xd);
            end
            return
        end
        
        function [ nn_len ] = compute_nn_len(X, sample_count)
            % Compute a length scale for curvature regularization. 
            % Sample observations at random and compute the distance to their
            % nearest neighbor. Use these nearest neighbor distances to compute
            % nn_len.
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
            nn_len = median(dists) / 2;
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
        
        function [ Xs Ys smpl_wts ] = sample_points(X, Y, smpls)
            % Sample a batch of training observations and compute gradients
            % for gradient and hessian parts of loss.
            obs_count = size(X,1);
            idx = randsample(1:obs_count, smpls, true);
            Xs = X(idx,:);
            Ys = Y(idx,:);
            smpl_wts = ones(smpls,1);
            return
        end
        
        
        function [ params ] = process_params(params)
            % Process parameters to use in training of some sort.
            if ~isfield(params, 'rounds')
                params.rounds = 10000;
            end
            if ~isfield(params, 'start_rate')
                params.start_rate = 0.1;
            end
            if ~isfield(params, 'decay_rate')
                params.decay_rate = 0.9999;
            end
            if ~isfield(params, 'momentum')
                params.momentum = 0.8;
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

