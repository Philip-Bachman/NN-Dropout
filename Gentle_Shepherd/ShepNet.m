classdef ShepNet < handle
    % This class performs "shepherded" training of a multi-layer neural-net.
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
        %   lam_shep: strength of "shepherd" regularization
        layer_lams
        % layer_sheps stores handles for the per-layer "shepherd" functions
        layer_sheps
        % lam_code weights the "encoding" loss effect at the output layer
        lam_code
        % weight_noise gives the standard deviation for additive weight noise
        weight_noise
        % drop_rate gives the rate for DropOut/DropConnect regularization
        drop_rate
        % drop_input gives a separate drop rate for the input layer
        drop_input
    end
    
    methods
        function [self] = ShepNet(X, Y, layer_dims, act_func, out_func)
            % Constructor for ShepNet class
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
                self.layer_lams(i).lam_grad = 0.0;
                self.layer_lams(i).lam_hess = 0.0;
            end
            % Initialize the shepherd function for each layer
            self.init_sheps(X, Y);
            % Set general global regularization weights
            self.lam_code = 1.0;
            self.weight_noise = 0.0;
            self.drop_rate = 0.0;
            self.drop_input = 0.0;
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
        
        function [ result ] = init_sheps(self, X, Y, shep_sizes, shep_weights)
            % Initialize shepherd functions associated with each layer.
            %
            if ~exist('shep_sizes','var')
                shep_sizes = 5 * ones(1,self.depth);
            end
            if ~exist('shep_weights','var')
                shep_weights = zeros(1,self.depth);
            end
            if (numel(shep_sizes) ~= self.depth)
                error('Incorrect number of shep sizes.');
            end
            if (numel(shep_weights) ~= self.depth)
                error('Incorrect number of shep weights.');
            end
            % Get current weights and compute per-layer activations for X
            l_weights = self.layer_weights;
            A = self.feedforward(X, l_weights);
            for i=1:self.depth,
                self.layer_lams(i).lam_shep = shep_weights(i);
                self.layer_sheps{i} = EucMOE(A{i}, Y, shep_sizes(i));
                self.layer_sheps{i}.init_weights(A{i}, Y, shep_sizes(i));
            end
            result = 1;
            return
        end
        
        function [ result ] = reset_sheps(self, X, Y, iters)
            % Initialize shepherd functions associated with each layer.
            %
            opts = struct();
            opts.Display = 'iter';
            opts.Method = 'lbfgs';
            opts.Corr = 10;
            opts.LS = 0;
            opts.LS_init = 0;
            opts.MaxIter = iters;
            opts.MaxFunEvals = 500;
            opts.TolX = 1e-10;
            % Get current weights and compute per-layer activations for X
            l_weights = self.layer_weights;
            A = self.feedforward(X, l_weights);
            for i=1:self.depth,
                shep = self.layer_sheps{i};
                shep_size = size(shep.moe_wts,2);
                shep.init_weights(A{i}, Y, shep_size, 0);
                shep.train(A{i}, Y, opts);
            end
            result = 1;
            return
        end
        
        function [ result ] = update_sheps(self, X, Y, iters)
            % Update the shepherd functions associated with each layer.
            %
            if ~exist('iters','var')
                iters = 25;
            end
            % Get the current net weights and compute activations for X
            l_weights = self.layer_weights;
            A = self.feedforward(X, l_weights);
            % Setup options structure for minFunc applied to shepherd funcs
            opts = struct();
            opts.Display = 'iter';
            opts.Method = 'lbfgs';
            opts.Corr = 10;
            opts.LS = 0;
            opts.LS_init = 0;
            opts.MaxIter = iters;
            opts.MaxFunEvals = 500;
            opts.TolX = 1e-10;
            % Update the shepherd function associated with each layer
            for i=1:self.depth,
                self.layer_sheps{i}.train(A{i}, Y, opts);
            end
            result = 1;
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
                l_acts{i} = func.feedforward(ShepNet.bias(A), W);
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
                    prv_acts = ShepNet.bias(l_acts{l_num-1});
                    prv_weights = l_weights{l_num-1};
                end
                if ((l_num < self.depth) && (l_num > 1))
                    % BP for internal layers (with post and pre layers)
                    func = self.act_func;
                    act_grads = l_grads{l_num};
                    nxt_weights = l_weights{l_num};
                    nxt_weights(end,:) = [];
                    nxt_grads = dN{l_num+1};
                    prv_acts = ShepNet.bias(l_acts{l_num-1});
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
                out_acts = func.feedforward(ShepNet.bias(out_acts), W);
            end
            return
        end
        
        function [ dN ] = bprop_gnrl(self, A, smpl_wts)
            % Do a backprop computation for the "general loss" incurred by the
            % per-layer, per-node activations A, weighted by smpl_wts.
            %
            dN = cell(1,self.depth);
            for i=1:self.depth,
                ll = self.layer_lams(i);
                % Compute weighted l1 and l2 regularization on the per-layer,
                % per-node activations.
                dN_l1 = ll.lam_l1 * bsxfun(@times, sign(A{i}), smpl_wts);
                dN_l2 = ll.lam_l2 * bsxfun(@times, A{i}, smpl_wts);
                dN{i} = dN_l1 + dN_l2;
                if ((i == self.depth) && (self.lam_code > 1e-10))
                    % Compute an "encoding" loss/grad at output, if desired
                    [Le dLe] = ShepNet.loss_lsq(A{end}, A{1});
                    dN_code = self.lam_code * bsxfun(@times, dLe, smpl_wts);
                    dN{i} = dN{i} + dN_code;
                end
            end
            return
        end
        
        function [ dN ] = bprop_shep(self, A, Y, smpl_wts)
            % Compute the per-layer gradients on post-transform activations
            % using the activations in A, with target class Y, w.r.t. sample
            % weights smpl_wts. Gradients result from losses produced by the
            % "Shepherd" functions associated with each layer.
            %
            dN = cell(1,self.depth);
            for i=1:self.depth,
                ll = self.layer_lams(i);
                if (ll.lam_shep > 1e-10)
                    dLdA = self.layer_sheps{i}.obs_grads(A{i},Y);
                    dN{i} = ll.lam_shep * bsxfun(@times, dLdA, smpl_wts);
                else
                    dN{i} = zeros(size(A{i}));
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
                [L dLl dLr] = ShepNet.loss_grad(Al{1},Ar{1},Al{i},Ar{i});
                % Reweight grads, e.g. for (possibly) importance sampling
                ll = self.layer_lams(i);
                dNl{i} = bsxfun(@times, dLl, (ll.lam_grad * smpl_wts));
                dNr{i} = bsxfun(@times, dLr, (ll.lam_grad * smpl_wts));
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
                [L dLl dLc dLr] = ShepNet.loss_hess(Al{1}, Ac{1}, Ar{1},...
                        Al{i}, Ac{i}, Ar{i});
                % Reweight grads, e.g. for (possibly) importance sampling
                ll = self.layer_lams(i);
                dNl{i} = bsxfun(@times, dLl, (ll.lam_hess * smpl_wts));
                dNc{i} = bsxfun(@times, dLc, (ll.lam_hess * smpl_wts));
                dNr{i} = bsxfun(@times, dLr, (ll.lam_hess * smpl_wts));
            end
            return
        end
        
        function [ L ] = check_gnrl_loss(self, A, smpl_wts)
            % Do a loss computation for the "general loss" incurred by the
            % per-layer, per-node activations A, weighted by smpl_wts.
            %
            obs_count = size(A{1},1);
            Ln = zeros(1,self.depth);
            for i=1:self.depth,
                ll = self.layer_lams(i);
                % Compute weighted l1 and l2 loss on the per-layer, per-node
                % activations.
                A_i = A{i};
                L_l1 = ll.lam_l1 * bsxfun(@times, abs(A_i), smpl_wts);
                L_l2 = (ll.lam_l2 / 2) * bsxfun(@times, A_i.^2, smpl_wts);
                Ln(i) = sum(L_l1(:)) + sum(L_l2(:));
                if ((i == self.depth) && (self.lam_code > 1e-10))
                    % Compute an "encoding" loss at output, if desired
                    L_enc = (self.lam_code / 2) * (A{end} - A{1}).^2;
                    Ln(i) = Ln(i) + sum(L_enc(:));
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
            for i=1:self.depth,
                L_l = ShepNet.loss_grad(Al{1},Ar{1},Al{i},Ar{i});
                % Reweight grads, e.g. for (possibly) importance sampling
                ll = self.layer_lams(i);
                L_l = bsxfun(@times, L_l, smpl_wts);
                Ln(i) = sum(L_l(:));
            end
            L = sum(Ln) / obs_count;
            return
        end
        
        function [ L ] = check_shep_loss(self, A, Y, smpl_wts)
            % Check loss coming from the shepherd functions
            %
            obs_count = size(A{1},1);
            Ln = zeros(1,self.depth);
            for i=1:self.depth,
                ll = self.layer_lams(i);
                L_l = self.layer_sheps{i}.obs_loss(A{i},Y);
                Ln(i) = ll.lam_shep * sum(sum(bsxfun(@times, L_l, smpl_wts)));
            end
            L = sum(Ln) / obs_count;
            return
        end
        
        function [ result ] =  train(self, X, Y, params)
            % Do fully parameterized training for a ShepNet
            if ~exist('params','var')
                params = struct();
            end
            params = ShepNet.process_params(params);
            % Compute a length scale for gradient regularization.
            grad_len = ShepNet.compute_grad_len(X, 10000);
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
                % Sample a batch of training observations..
                [Xl Xc Xr Yc smpl_wts] = ...
                    ShepNet.sample_points(X, Y, batch_size, grad_len);
                % Get the droppy/fuzzy weights to use with this batch
                l_weights = self.get_drop_weights(1);
                % Compute activations for left/center/right points
                acts_l = self.feedforward(Xl, l_weights);
                acts_c = self.feedforward(Xc, l_weights);
                acts_r = self.feedforward(Xr, l_weights);
                % Get per-node gradients for general all-purpose stuff, e.g.
                % encode penalty on output, and L1/L2 regularization on acts.
                dNc_gnrl = self.bprop_gnrl(acts_c, smpl_wts);
                % Get per-node gradients for the "Shepherd Functions"
                dNc_shep = self.bprop_shep(acts_c, Yc, smpl_wts);
                % Get per-node gradients for gradient regularization
                [dNl_grad dNr_grad] = ...
                    self.bprop_grad(acts_l, acts_r, smpl_wts);
                % Get per-node gradients for Hessian regularization
                [dNl_hess dNc_hess dNr_hess] = ...
                    self.bprop_hess(acts_l, acts_c, acts_r, smpl_wts);
                % Merge "center" gradients into dNc_gnrl, and left/right
                % gradients into dNl_grad/dNr_grad
                for l=1:self.depth,
                    dNc_gnrl{l} = dNc_gnrl{l} + dNc_shep{l} + dNc_hess{l};
                    dNl_grad{l} = dNl_grad{l} + dNl_hess{l};
                    dNr_grad{l} = dNr_grad{l} + dNr_hess{l};
                end
                % Backprop the left/center/right per-node gradients along with
                % the relevant weights and activations to get per-weight grads.
                dLdWc = self.backprop(acts_c, l_weights, dNc_gnrl);
                dLdWl = self.backprop(acts_l, l_weights, dNl_grad);
                dLdWr = self.backprop(acts_r, l_weights, dNr_grad);
                % Merge left/center/right per-weight grads for easy updating
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
                if ((e == 1) || (mod(e, 50) == 0))
                    % Sample a set of weights
                    l_weights = self.layer_weights;
                    % Sample some left/center/right points and their acts
                    [Xl Xc Xr Yc smpl_wts] = ...
                        ShepNet.sample_points(X, Y, 2500, grad_len);
                    acts_l = self.feedforward(Xl, l_weights);
                    acts_c = self.feedforward(Xc, l_weights);
                    acts_r = self.feedforward(Xr, l_weights);
                    % Compute relevant losses
                    L_gnrl = self.check_gnrl_loss(acts_c, smpl_wts);
                    L_grad = self.check_grad_loss(acts_l, acts_r, smpl_wts);
                    L_shep = self.check_shep_loss(acts_c, Yc, smpl_wts);
                    if (params.do_validate)
                        fprintf('    %d: t=(%.4f, %.4f, %.4f) v=( , )\n',...
                            e, L_gnrl, L_grad, L_shep);
                    else
                        fprintf('    %d: t=(%.4f, %.4f, %.4f)\n',...
                            e,L_gnrl, L_grad, L_shep);
                    end
                end
            end
            fprintf('\n');
            result = 1;
            return
        end
        
        function [ L ] = test_hess(self, X, sample_count)
            % Check hessian loss near the points in X.
            %
            grad_len = ShepNet.compute_grad_len(X, 10000);
            % Get sample points for computing hessian penalty
            Y = zeros(obs_count,1);
            [Xl Xc Xr] = ShepNet.sample_points(X, Y, sample_count, grad_len);
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
            L = ShepNet.loss_hess(Xl,Xc,Xr,Fl,Fc,Fr);
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
        
        function [ grad_len ] = compute_grad_len(X, sample_count)
            % Compute a length scale for gradient/hessian regularization. 
            % Sample random pairs of observations and set the length to a
            % function of some approximate quantile of the pairwise
            % inter-observation distances.
            %
            obs_count = size(X,1);
            dists = zeros(sample_count,1);
            for i=1:sample_count,
                x1 = X(randi(obs_count),:);
                x2 = X(randi(obs_count),:);
                dists(i) = sqrt(sum((x1 - x2).^2));
            end
            grad_len = (quantile(dists,0.01) / 2);
            return
        end
        
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
            end
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

