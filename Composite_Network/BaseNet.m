classdef BaseNet < handle
    % This is a class for managing a base-level multi-layer neural-net.
    %
    % This class is primarily for reuse by the CompNet class, which composes
    % multiple BaseNet instances for training and testing jointly.
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
        % layer_nsizes gives the number of nodes in each layer of this net
        %   note: these sizes do _not_ include the bias
        layer_nsizes
        % layer_weights is a cell array such that layer_weights{l} contains a
        % matrix in which entry (i,j) contains the weights between node i in
        % layer l and node j in layer l+1. The number of weight matrices in
        % layer_weights (i.e. its length) is self.depth - 1.
        %   note: due to biases each matrix in layer_weights has an extra row
        layer_weights
        % pl_lam_l1s gives the strength of L1 regularization to apply to the
        % per-node activations at each layer.
        pl_lam_l1s
        % out_loss gives the function to use for loss/grad on network output
        out_loss
        % weight_noise gives the standard deviation for additive weight noise
        weight_noise
        % drop_rate gives the rate for DropOut/DropConnect regularization
        drop_rate
        % drop_input gives a separate drop rate for the input layer
        drop_input
    end
    
    methods
        function [self] = BaseNet(layer_dims, act_func, out_func)
            % Constructor for BaseNet class
            if ~exist('out_func','var')
                % Default to using linear activation transform at output layer
                out_func = ActFunc(1);
            end
            self.act_func = act_func;
            self.out_func = out_func;
            self.depth = numel(layer_dims);
            self.layer_nsizes = reshape(layer_dims,1,numel(layer_dims));
            self.layer_weights = [];
            self.init_weights(0.1);
            self.pl_lam_l1s = zeros(1,self.depth);
            self.out_loss = @(Yh, Y) BaseNet.loss_mclr(Yh, Y);
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
                pre_dim = self.layer_nsizes(i)+1;
                post_dim = self.layer_nsizes(i+1);
                weights = randn(pre_dim,post_dim);
                weights(end,:) = 0;
                %if (i > 1)
                %    for j=1:size(weights,2),
                %        keep_idx = randsample((size(weights,1)-1), 50);
                %        drop_idx = setdiff(1:(size(weights,1)-1),keep_idx);
                %        weights(drop_idx,j) = 0;
                %    end
                %end
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
                l_acts{i} = func.feedforward(BaseNet.bias(A), W);
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
                    prv_acts = BaseNet.bias(l_acts{l_num-1});
                    prv_weights = l_weights{l_num-1};
                end
                if ((l_num < self.depth) && (l_num > 1))
                    % BP for internal layers (with post and pre layers)
                    func = self.act_func;
                    act_grads = l_grads{l_num};
                    nxt_weights = l_weights{l_num};
                    nxt_weights(end,:) = [];
                    nxt_grads = dN{l_num+1};
                    prv_acts = BaseNet.bias(l_acts{l_num-1});
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
                out_acts = func.feedforward(BaseNet.bias(out_acts), W);
            end
            return
        end
        
        function [ w_grads ] = bprop_out(self, weights, acts, Y, smpl_wts)
            % Do a backprop computation with dropout for the data in X/Y.
            [L dL] = self.out_loss(acts{self.depth}, Y);
            % Get gradients to inject at each node during backprop
            grads = cell(size(acts));
            for i=1:self.depth,
                if (i == self.depth)
                    grads{i} = bsxfun(@times, dL, smpl_wts);
                else
                    grads{i} = zeros(size(acts{i}));
                end
            end
            % Get full per-edge gradients by backpropping per-node gradients
            w_grads = self.backprop(acts, weights, grads);
            return
        end
        
        function [ w_grads ] = bprop_func(self, l_weights, l_acts, smpl_wts)
            % Backpropagate to compute gradients for L2 functional norm
            % regularization.
            %
            % Compute loss and gradients for L2 functional norm
            F = l_acts{self.depth};
            [L dL] = BaseNet.loss_lsq(F, zeros(size(F)));
            % Get gradients to inject at each node during backprop
            l_grads = cell(size(l_acts));
            for i=1:self.depth,
                if (i == self.depth)
                    l_grads{i} = bsxfun(@times, dL, smpl_wts);
                else
                    l_grads{i} = zeros(size(l_acts{i}));
                end
            end
            % Get full per-edge gradients by backpropping per-node gradients
            w_grads = self.backprop(l_acts, l_weights, l_grads);
            return
        end

        function [ dWl dWr dNl dNr L ] = bprop_grad_ind(self, l_weights,...
                acts_l, acts_r, Xl, Xr, smpl_wts)
            % Compute gradients at all nodes, starting with loss values and
            % gradients for each observation at output layer. Do this for
            % each point underlying the fd-estimated gradient functionals.
            %
            % Return relavant for left and right points independently.
            %
            l_grads_l = cell(size(acts_l));
            l_grads_r = cell(size(acts_r));
            for i=1:self.depth,
                if (i == self.depth)
                    [L dLl dLr] = BaseNet.loss_grad(Xl,Xr,acts_l{i},acts_r{i});
                else
                    dLl = zeros(size(acts_l{i}));
                    dLr = zeros(size(acts_r{i}));
                end
                l_grads_l{i} = bsxfun(@times, dLl, smpl_wts);
                l_grads_r{i} = bsxfun(@times, dLr, smpl_wts);
            end
            % Get full per-edge gradients by backpropping per-node gradients
            [dWl dNl] = self.backprop(acts_l, l_weights, l_grads_l);
            [dWr dNr] = self.backprop(acts_r, l_weights, l_grads_r);
            return
        end
        
        function [ dWj dNj L ] = bprop_grad_jnt(self, l_weights, acts_l,...
                acts_r, Xl, Xr, smpl_wts)
            % Compute gradients at all nodes, starting with loss values and
            % gradients for each observation at output layer. Do this for
            % each point underlying the fd-estimated gradient functionals.
            %
            % Return relavant for left and right points jointly.
            %
            [dWl dWr dNl dNr L] = self.bprop_grad_ind(l_weights, acts_l,...
                acts_r, Xl, Xr, smpl_wts);
            dNj = cell(1,self.depth);
            for i=1:self.depth,
                dNj{i} = (dNl{i} + dNr{i});
            end
            dWj = cell(1,self.depth-1);
            for i=1:(self.depth-1),
                dWj{i} = (dWl{i} + dWr{i});
            end
            return
        end
        
        function [ w_grads ] = bprop_hess(self, l_weights, acts_l, acts_c, ...
                acts_r, Xl, Xc, Xr, smpl_wts)
            % Compute gradients at all nodes, starting with loss values and
            % gradients for each observation at output layer. Do this for
            % each point underlying the fd-estimated Hessian functionals.
            l_grads_l = cell(size(acts_l));
            l_grads_c = cell(size(acts_c));
            l_grads_r = cell(size(acts_r));
            for i=1:self.depth,
                if (i == self.depth)
                    [L dLl dLc dLr] = BaseNet.loss_hess(Xl, Xc, Xr,...
                        acts_l{i}, acts_c{i}, acts_r{i});
                else
                    dLl = zeros(size(acts_l{i}));
                    dLc = zeros(size(acts_c{i}));
                    dLr = zeros(size(acts_r{i}));
                end
                l_grads_l{i} = bsxfun(@times, dLl, smpl_wts);
                l_grads_c{i} = bsxfun(@times, dLc, smpl_wts);
                l_grads_r{i} = bsxfun(@times, dLr, smpl_wts);
            end
            % Get full per-edge gradients by backpropping per-node gradients
            dWl = self.backprop(acts_l, l_weights, l_grads_l);
            dWc = self.backprop(acts_c, l_weights, l_grads_c);
            dWr = self.backprop(acts_r, l_weights, l_grads_r);
            w_grads = cell(1,self.depth-1);
            for i=1:(self.depth-1),
                w_grads{i} = (dWl{i} + dWc{i} + dWr{i});
            end
            return
        end
        
        function [ result ] =  train(self, X, Y, params)
            % Do fully parameterized training for a BaseNet
            if ~exist('params','var')
                params = struct();
            end
            params = BaseNet.process_params(params);
            obs_count = size(X,1);
            % Compute a length scale for gradient regularization. Sample random
            % pairs of observations and set the length to a function of some
            % approximate quantile of the pairwise inter-observation distances.
            dists = zeros(10000,1);
            for i=1:10000,
                x1 = X(randi(obs_count),:);
                x2 = X(randi(obs_count),:);
                dists(i) = sqrt(sum((x1 - x2).^2));
            end
            grad_len = (quantile(dists,0.01) / 2);
            % Setup parameters for gradient updates (rates and momentums)
            rate = params.start_rate;
            all_mom = cell(1,self.depth-1);
            for i=1:(self.depth-1),
                all_mom{i} = zeros(size(self.layer_weights{i}));
            end
            % Setup accuracy/validation tracking stuff
            train_accs = zeros(1,params.rounds);
            train_loss = zeros(1,params.rounds);
            if (params.do_validate)
                test_accs = zeros(1,params.rounds);
                test_loss = zeros(1,params.rounds);
            end
            batch_size = params.batch_size;
            fprintf('Updating weights (%d rounds):\n', params.rounds);
            for e=1:params.rounds,
                % Sample a batch of training observations..
                [Xl Xc Xr Yc smpl_wts] = ...
                    BaseNet.sample_points(X, Y, batch_size, grad_len);
                % Get the droppy/fuzzy weights to use with this batch
                l_weights = self.get_drop_weights(1);
                % Get central point activations
                acts_c = self.feedforward(Xc, l_weights);
                grd_obj = self.bprop_out(l_weights, acts_c, Yc, smpl_wts);
                if ((params.lam_grad > 1e-8) || (params.lam_hess > 1e-8))
                    acts_l = self.feedforward(Xl, l_weights);
                    acts_r = self.feedforward(Xr, l_weights);
                    if (params.lam_grad > 1e-8)
                        grd_grad = self.bprop_grad_jnt(l_weights, acts_l, acts_r,...
                            Xl, Xr, smpl_wts);
                    end
                    if (params.lam_hess > 1e-8)
                        grd_hess = self.bprop_hess(l_weights, acts_l, acts_c,...
                            acts_r, Xl, Xc, Xr, smpl_wts);
                    end
                end
                if (params.lam_grad <= 1e-8)
                    grd_grad = grd_obj;
                end
                if (params.lam_hess <= 1e-8)
                    grd_hess = grd_obj; 
                end
                % Update the weights at each layer using the computed gradients
                for i=1:(self.depth-1),
                    % Aggregate class/grad/hess gradients
                    l_grads = grd_obj{i} + ...
                        (params.lam_grad * grd_grad{i}) + ...
                        (params.lam_hess * grd_hess{i});
                    l_weights = self.layer_weights{i};
                    % Add L2 per-weight gradients
                    l_grads = l_grads + (params.lam_l2 * l_weights);
                    % Mix gradient using momentum
                    dW = (params.momentum * all_mom{i}) + ...
                        ((1 - params.momentum) * (rate * l_grads));
                    all_mom{i} = dW;
                    % Update weights using blended gradients
                    self.layer_weights{i} = l_weights - dW;
                end
                % Decay the learning rate after performing update
                rate = rate * params.decay_rate;
                % Occasionally recompute and display the loss and accuracy
                if ((e == 1) || (mod(e, 50) == 0))
                    [Lc Ac] = self.test_loss_acc(X, Y);
                    if (params.do_validate)
                        [Lc_v Ac_v] = ...
                            self.test_loss_acc(params.Xv, params.Yv);
                        fprintf('    %d: t=(%.4f, %.4f) v=(%.4f, %.4f)\n',...
                            e, Lc, Ac, Lc_v, Ac_v);
                    else
                        fprintf('    %d: %.4f, %.4f\n', e, Lc, Ac);
                    end
                end
                % Store validation/accuracy info
                train_loss(e) = Lc;
                train_accs(e) = Ac;
                if (params.do_validate)
                    test_loss(e) = Lc_v;
                    test_accs(e) = Ac_v;
                end
            end
            fprintf('\n');
            result = struct();
            result.Yh = self.evaluate(X);
            result.train_accs = train_accs;
            result.train_loss = train_loss;
            if (params.do_validate)
                result.test_accs = test_accs;
                result.test_loss = test_loss;
            end
            return
        end
        
        function [ Lc Ac ] = test_loss_acc(self, X, Y)
            % Check classification and encoding loss / accuracy
            if (size(X,1) > 5000)
                idx = randsample(size(X,1),5000);
            else
                idx = 1:size(X,1);
            end
            X = X(idx,:);
            Y = Y(idx,:);
            % Check classification loss/accuracy
            Yh = self.evaluate(X);
            Lc = self.out_loss(Yh, Y);
            Lc = mean(Lc(:));
            [xxx Y_idx] = max(Y,[],2);
            [xxx Yh_idx] = max(Yh,[],2);
            Ac = sum(Y_idx == Yh_idx) / numel(Y_idx);
            return
        end
        
        function [ L ] = test_hess(self, X, sample_count)
            % Check hessian loss near the points in X.
            %
            obs_count = size(X,1);
            D = squareform(pdist(X));
            D = D + (1e3 * eye(size(D)));
            D = sort(D,'ascend');
            grad_len = median(D(1,:)) + 1e-5;
            % Get sample points for computing hessian penalty
            Y = zeros(obs_count,1);
            [Xl Xc Xr] = BaseNet.sample_points(X, Y, sample_count, grad_len);
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
            L = BaseNet.loss_hess(Xl,Xc,Xr,Fl,Fc,Fr);
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
            if ~isfield(params, 'lam_grad')
                params.lam_grad = 0;
            end
            if ~isfield(params, 'lam_hess')
                params.lam_hess = 0;
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
            params.momentum = min(1, max(0, params.momentum));
            return
        end
        
    end 
    
end










%%%%%%%%%%%%%%
% EYE BUFFER %
%%%%%%%%%%%%%%

