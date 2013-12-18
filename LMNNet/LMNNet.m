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
        %   lam_lmnn: strength of "LMNN" regularization
        layer_lams
        % weight_noise gives the standard deviation for additive weight noise
        weight_noise
        % drop_hidden gives the drop out rate for hidden layers
        drop_hidden
        % drop_input gives the drop out rate at the input layer
        drop_input
        % drop_output gives the drop out rate at the output layer
        drop_output
        % const_layer determines the layer to use when computing neighborhood
        % constraints for LMNN regularization.
        const_layer
    end
    
    methods
        function [self] = LMNNet(layer_dims, act_func, out_func)
            % Constructor for LMNNet class
            if ~exist('out_func','var')
                % Default to using linear activation transform at output layer
                out_func = ActFunc(1);
            end
            self.act_func = act_func;
            self.out_func = out_func;
            self.depth = numel(layer_dims);
            self.layer_sizes = reshape(layer_dims,1,numel(layer_dims));
            % Initialize inter-layer weights
            self.layer_weights = [];
            self.init_weights(0.1);
            % Initialize per-layer activation regularization weights
            self.layer_lams = struct();
            for i=1:self.depth,
                self.layer_lams(i).lam_l1 = 0.0;
                self.layer_lams(i).lam_l2 = 0.0;
                self.layer_lams(i).wt_bnd = 10;
                self.layer_lams(i).lam_lmnn = 0.0;
            end
            % Set general global regularization weights
            self.weight_noise = 0.0;
            self.drop_hidden = 0.0;
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
                weights(end,:) = 0.1 + (0.1 * weights(end,:));
                if ((i > 1) && (i < (self.depth - 1)))
                   for j=1:size(weights,2),
                       keep_count = min(50, pre_dim-1);
                       keep_idx = randsample(1:(pre_dim-1), keep_count);
                       drop_idx = setdiff(1:(pre_dim-1),keep_idx);
                       weights(drop_idx,j) = 0.1 * weights(drop_idx,j);
                   end
                end
                self.layer_weights{i} = weights .* weight_scale;
            end
            result = 0;
            return
        end
        
        function [ l_acts d_masks ] = feedforward(self, X, l_weights, m_drop)
            % Get per-layer activations for the observations in X, given the
            % weights in l_weights. If m_drop is 1, do mask-based dropping of
            % activations at each layer, and return the sampled masks.
            %
            if ~exist('m_drop','var')
                m_drop = 0;
            end
            l_acts = cell(1,self.depth);
            d_masks = cell(1,self.depth);
            l_acts{1} = X;
            d_masks{1} = ones(size(X));
            if ((m_drop == 1) && (self.drop_input > 1e-8))
                mask = rand(size(l_acts{1})) > self.drop_input;
                d_masks{1} = mask;
                l_acts{1} = l_acts{1} .* mask;
            end
            for i=2:self.depth,
                if (i == self.depth)
                    func = self.out_func;
                else
                    func = self.act_func;
                end
                W = l_weights{i-1};
                A_pre = l_acts{i-1};
                A_cur = func.feedforward(LMNNet.bias(A_pre), W);
                l_acts{i} = A_cur;
                d_masks{i} = ones(size(A_cur));
                if (m_drop == 1)
                    % Set drop rate based on the current layer type
                    if (i < self.depth)
                        drop_rate = self.drop_hidden;
                    else
                        drop_rate = self.drop_output;
                    end
                    if (drop_rate > 1e-8)
                        mask = rand(size(l_acts{i})) > drop_rate;
                        d_masks{i} = mask;
                        l_acts{i} = l_acts{i} .* mask;
                    end
                end
            end
            return
        end
        
        function [ dW dN ] = backprop(self, l_acts, l_weights, l_grads, l_masks)
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
            %   l_masks: masks for dropout on per-layer, per-node activations
            %
            % Outputs:
            %   dW: grad on each inter-layer weight (size of l_weights)
            %   dN: grad on each pre-transform activation (size of l_grads)
            %       note: dN{1} will be grads on inputs to network
            %
            if ~exist('l_masks','var')
                l_masks = cell(1,self.depth);
                for i=1:self.depth,
                    l_masks{i} = ones(size(l_acts{i}));
                end
            end
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
                cur_grads = cur_grads .* l_masks{l_num};
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
                if (i < (self.depth - 1))
                    drop_rate = self.drop_hidden;
                else
                    drop_rate = self.drop_output;
                end
                if ((i > 1) && (abs(drop_rate - 0.5) < 0.1))
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
        
        function [ dN ] = bprop_gnrl(self, A, smpl_wts)
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
                        % Apply L1 regularization to _normalized_ rows.
                        [An BP] = LMNNet.norm_rows(A{i});
                        dN_l1 = ll.lam_l1 * ...
                            bsxfun(@times,BP(ones(size(An))),smpl_wts);
                        dN{i} = dN{i} + dN_l1;
                    end
                    if (ll.lam_l2 > 1e-10)
                        % Apply L2 regularization
                        dN_l2 = ll.lam_l2 * bsxfun(@times,A{i},smpl_wts);
                        dN{i} = dN{i} + dN_l2;
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
                if (ll.lam_lmnn > 1e-10)
                    [Ll dAc dAn dAf] = LMNNet.lmnn_grads_dot(...
                        Ac{i}, An{i}, Af{i}, 0.2);
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
        
        function [ L ] = check_gnrl_loss(self, A, smpl_wts)
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
                L_l1 = ll.lam_l1 * bsxfun(@times,LMNNet.norm_rows(A_i),smpl_wts);
                L_l2 = (ll.lam_l2 / 2) * bsxfun(@times, A_i.^2, smpl_wts);
                Ln(i) = sum(L_l1(:)) + sum(L_l2(:));
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
                    Ll = LMNNet.lmnn_grads_dot(Ac{i}, An{i}, Af{i}, 0.2);
                    Ll = bsxfun(@times, Ll, smpl_wts);
                    L = L + ll.lam_lmnn * sum(Ll(:));
                end
            end
            L = L / obs_count;
            return
        end
        
        function [L_gnrl L_lmnn] = check_losses(self, X, Y, nc_lmnn)
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
            [Xs Ys smpl_wts] = ...
                LMNNet.sample_points(X, Y, 2000);
            % Compute activations for left/center/right points
            acts_s = self.feedforward(Xs, l_weights);
            % Get general loss for this batch
            L_gnrl = self.check_gnrl_loss(acts_s, smpl_wts);
            return
        end
            
        
        function [ result ] =  train(self, X, Y, params)
            % Do fully parameterized training for a LMNNet
            if ~exist('params','var')
                params = struct();
            end
            params = LMNNet.process_params(params);
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
            nc_lmnn = zeros(1000000,3);
            nc_count = size(nc_lmnn,1);
            for i=1:3,
                nc_lmnn(:,i) = randsample(1:size(X,1),nc_count,true);
            end
            for e=1:params.rounds,
                % Get the droppy/fuzzy weights to use with this round
                l_weights = self.layer_weights;
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Sample points and do feedforward/backprop for general loss %
                % on output layer and internal layers.                       %
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                [Xs Ys smpl_wts] = ...
                    LMNNet.sample_points(X, Y, batch_size);
                % Compute activations for left/center/right points
                [acts_g masks_g] = self.feedforward(Xs, l_weights, 1);
                % Get general per-node loss and grads for the "true" points
                dN_gnrl = self.bprop_gnrl(acts_g, smpl_wts);
                % Backprop per-node gradients with source activations/weights
                dLdW_gnrl = self.backprop(acts_g, l_weights, dN_gnrl, masks_g);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % First perform a series of computations to get gradients    %
                % derived from the (optional) LMNN penalties on each layer   %
                % of the network and from general losses on all layers       %
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if (e >= params.lmnn_start)
                    if ((e == params.lmnn_start) || (mod(e, 500) == 0))
                        % Compute neighbor constraints from current embedding
                        if (params.lmnn_count < size(X,1))
                            idx = randsample(1:size(X,1),params.lmnn_count);
                            X_nc = X(idx,:);
                            Y_nc = Y(idx,:);
                        else
                            idx = 1:size(X,1);
                            X_nc = X;
                            Y_nc = Y;
                        end
                        A = self.feedforward(X_nc, self.layer_weights);
                        nc_lmnn = LMNNet.neighbor_constraints(...
                            A{self.const_layer}, Y_nc, 5, 10, 1);
                        clear('A','X_nc','Y_nc');
                        nc_lmnn(:,1) = idx(nc_lmnn(:,1));
                        nc_lmnn(:,2) = idx(nc_lmnn(:,2));
                        nc_lmnn(:,3) = idx(nc_lmnn(:,3));
                        nc_count = size(nc_lmnn,1);
                        for l=1:(self.depth-1),
                            all_mom{l} = zeros(size(all_mom{l}));
                        end
                    end
                    % Sample a set of constraint point triples for this batch
                    smpl_wts = ones(batch_size,1);
                    smpl_idx = randsample(nc_count,batch_size,false);
                    nc_batch = nc_lmnn(smpl_idx,:);
                    Xc = X(nc_batch(:,1),:);
                    Xn = X(nc_batch(:,2),:);
                    Xf = X(nc_batch(:,3),:);
                    % Compute activations for the sampled triples
                    [acts_c masks_c] = self.feedforward(Xc, l_weights, 1);
                    [acts_n masks_n] = self.feedforward(Xn, l_weights, 1);
                    [acts_f masks_f] = self.feedforward(Xf, l_weights, 1);
                    % Get per-node gradients derived from LMNN losses
                    [dNc_lmnn dNn_lmnn dNf_lmnn] = ...
                        self.bprop_lmnn(acts_c, acts_n, acts_f, smpl_wts);
                    % Backprop per-node gradients for LMNN constraint losses
                    dLdWc = self.backprop(acts_c, l_weights, dNc_lmnn, masks_c);
                    dLdWn = self.backprop(acts_n, l_weights, dNn_lmnn, masks_n);
                    dLdWf = self.backprop(acts_f, l_weights, dNf_lmnn, masks_f);
                    dLdW_lmnn = cell(1,(self.depth-1));
                    for l=1:(self.depth-1),
                        dLdW_lmnn{l} = dLdWc{l} + dLdWn{l} + dLdWf{l};
                    end
                else
                    dLdW_lmnn = cell(1,(self.depth-1));
                    for l=1:(self.depth-1),
                        dLdW_lmnn{l} = zeros(size(dLdW_gnrl{l}));
                    end
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
                % Compact the per-weight gradients into a single array.      %
                % Then, use the array for weight updating.                   %
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                dLdW = cell(1,(self.depth-1));
                for l=1:(self.depth-1),
                    dLdW{l} = dLdW_gnrl{l} + dLdW_lmnn{l};
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
                    % Record gradients for weights at this layer
                    layer_dW{l} = l_dLdW;
                    % Compute some norms, for diagnostics
                    lwts_norms(l,e) = sqrt(sum(sum(l_weights.^2)));
                    dldw_norms(l,e) = sqrt(sum(sum(l_dLdW.^2)));
                    max_sratio(l,e) = max(...
                        sqrt(sum(l_dLdW.^2,1))./sqrt(sum(l_weights.^2,1))+1e-3);
                    if ((sum(isnan(l_dLdW(:)))>0) || (sum(isinf(l_dLdW(:)))>0))
                        error('BROKEN GRADIENTS');
                    end
                end
                max_ratios(e) = max(max_sratio(:,e));
                if (e > 50)
                    plot(1:e,max_ratios(1:e));
                    drawnow();
                end
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
                    [L_gnrl L_lmnn] = ...
                        self.check_losses(X, Y, nc_lmnn);
                    fprintf('    %d: t=(%.4f, %.4f)\n',...
                        e, L_gnrl, L_lmnn);
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
                        Lk = LMNNet.lmnn_grads_dot(Ac{k},Al{k},Ar{k},0.2);
                        Ls = Ls + mean(Lk);
                    end
                    
                end
                L(i) = Ls;
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

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % LMNN accessory Functions %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [ I_nn ] = knn_ind( Xte, Xtr, k, do_loo, do_cos )
            % Find indices of the knn for points in Xte measured with respect to
            % the points in Xtr.
            %
            % (do_loo == 1) => do "leave-one-out" knn, assuming Xtr == Xte
            % (do_cos == 1) => use max dot-products instead of min euclideans.
            %
            if ~exist('do_loo','var')
                do_loo = 0;
            end
            if ~exist('do_cos','var')
                do_cos = 0;
            end
            obs_count = size(Xte,1);
            I_nn = zeros(obs_count,k);
            fprintf('Computing knn:');
            for i=1:obs_count,
                if (mod(i,round(obs_count/50)) == 0)
                    fprintf('.');
                end
                if (do_cos == 1)
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
        
        function [ n_const ] = neighbor_constraints( X, Y, k_in, k_out, do_cos )
            % Generate "neighbor constraints" with which to train a metric.
            if ~exist('do_cos','var')
                do_cos = 0;
            end
            if (do_cos == 1)
                % Use a "cosine" distance measure
                X = LMNNet.norm_rows(X);
            end
            [xxx Y] = max(Y,[],2);
            I_nn = LMNNet.knn_ind(X, X, (3*(k_in+k_out)), 1, do_cos);
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
        
        function [ L dXc dXn dXf ] = lmnn_grads_tish( Xc, Xn, Xf, margin )
            % Compute gradients of standard LMNN using T-ish distance.
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
            d = 0.1;
            On = Xn - Xc;
            Of = Xf - Xc;
            % Compute (squared) norms of the offsets On/Of
            Dn = sqrt(sum(On.^2,2) + d);
            Df = sqrt(sum(Of.^2,2) + d);
            % Get losses and indicators for violated LMNN constraints
            m_viol = max(0, (Dn - Df) + margin);
            L = m_viol;
            % Compute gradients for violated constraints
            dXn = bsxfun(@times, bsxfun(@rdivide,On,Dn), (m_viol > 1e-10));
            dXf = bsxfun(@times, bsxfun(@rdivide,-Of,Df), (m_viol > 1e-10));
            dXc = -dXn - dXf;
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
            return
        end
        
        function [L dLdAc dLdAn dLdAf] = lmnn_grads_dot(Ac, An, Af, margin)
            % Compute gradients of standard LMNN using dot-product distance.
            %
            % Parameters:
            %   Ac: central/source activations
            %   An: activations that should be closer to those in Ac
            %   Af: activations that should be further from those in Ac
            %   margin: desired margin between near/far distances w.r.t. Ac
            % Outputs:
            %   L: loss for each LMNN triplet (Ac(i,:),An(i,:),Af(i,:))
            %   dLdAc: gradient of L w.r.t. Ac
            %   dLdAn: gradient of L w.r.t. An
            %   dLdAf: gradient of L w.r.t. Af
            %
            [Xc BPc] = LMNNet.norm_rows(Ac);
            [Xn BPn] = LMNNet.norm_rows(An);
            [Xf BPf] = LMNNet.norm_rows(Af);
            % Compute dot-products between center/(near|far) activations
            Dn = sum(Xc .* Xn,2);
            Df = sum(Xc .* Xf,2);
            % Get losses and indicators for violated LMNN constraints
            m_viol = max(0, (Df - Dn) + margin);
            m_mask = m_viol > 1e-10;
            L = m_viol;
            % Compute gradients for violated constraints
            dLdXc = bsxfun(@times, (Xf - Xn), m_mask);
            dLdXn = bsxfun(@times, -Xc, m_mask);
            dLdXf = bsxfun(@times, Xc, m_mask);
            dLdAc = BPc(dLdXc);
            dLdAn = BPn(dLdXn);
            dLdAf = BPf(dLdXf);
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
        
        function [ F BP ] = norm_rows(X)
            % L2 normalize X by rows, and return both the row-normalized matrix
            % and a function handle for backpropagating through normalization.
            N = sqrt(sum(X.^2,2) + 1e-6);
            F = bsxfun(@rdivide,X,N);
            % Backpropagate through normalization for unit norm
            BP = @( D ) ...
                (bsxfun(@rdivide,D,N) - bsxfun(@times,F,(sum(D.*X,2)./(N.^2))));
            return
        end
        
        function [ Xs Ys smpl_wts ] = sample_points(X, Y, smpls)
            % Sample a batch of training observations.
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
            if ~isfield(params, 'lmnn_start')
                params.lmnn_start = 1000;
            end
            if ~isfield(params, 'lmnn_count')
                params.lmnn_count = 5000;
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

