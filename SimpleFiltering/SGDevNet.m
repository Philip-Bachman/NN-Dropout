classdef SGDevNet < handle
    % This class performs training of a LMNNN multi-layer neural-net.
    %
    
    properties
        % act_func is an ActFunc instance for computing feed-forward activation
        % levels in hidden layers and backpropagating gradients
        act_func
        % out_func is an ActFunc instance for computing feed-forward activation
        % levels at the output layer, given activations of penultimate layer.
        out_func
        % out_loss gives the function to apply at output layer, which should be
        % set based on whether this net is for classification or regression
        out_loss
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
        %   l2_bnd: L2 norm bound on incoming weights to each node
        %   lam_dev: dropout ensemble variance regularization strength
        %   dev_type: set the transform to apply prior to computing dropout
        %             ensemble variance.
        %             1: row-wise L2 normalization
        %             2: element-wise hypertangent
        %             3: none (i.e. regularize variance of raw activations)
        layer_lams
        % lam_l2 controls network-wide L2 weight regularization
        lam_l2
        % lam_l1 controls network-wide (soft) L1 weight regularization
        lam_l1
        % do_dev tells whether to train with "Dropout Ensemble Variance"
        % regularization or whether to apply standard dropout
        do_dev
        % do_cde tells whether to train the network using standard dropout
        % ensemble approach (i.e. train network to minimize expected loss at
        % output layer with respect to the dropout ensemble.
        do_sde
        % drop_hidden gives the drop out rate for hidden layers
        drop_hidden
        % drop_input gives the drop out rate at the input layer
        drop_input
    end
    
    methods
        function [self] = SGDevNet(layer_dims, act_func, out_func)
            % Constructor for SGDevNet class
            if ~exist('out_func','var')
                % Default to using linear activation transform at output layer
                out_func = ActFunc(1);
            end
            self.act_func = act_func;
            self.out_func = out_func;
            self.out_loss = @SGDevNet.loss_mcl2h;
            self.depth = numel(layer_dims);
            self.layer_sizes = reshape(layer_dims,1,numel(layer_dims));
            % Initialize inter-layer weights
            self.layer_weights = [];
            self.init_weights(0.1);
            % Initialize per-layer activation regularization weights
            self.layer_lams = struct();
            for i=1:self.depth,
                self.layer_lams(i).l2_bnd = 10;
                self.layer_lams(i).lam_dev = 0.0;
                self.layer_lams(i).dev_type = 1;
            end
            % Set network-wide weight regularization parameters
            self.lam_l2 = 0;
            self.lam_l1 = 0;
            % Set parameters for dropout regularization
            self.drop_hidden = 0.0;
            self.drop_input = 0.0;
            self.do_dev = 0;
            self.do_sde = 1;
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
                % Default to a smallish positive weight on bias.
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
        
        function [ acc ] = check_acc(self, X, Y)
            % Check classification performance on the given data
            if (size(Y,2) == 1)
                Y = SGDevNet.to_cats(Y);
            else
                Y = SGDevNet.class_cats(Y);
            end
            Yh = SGDevNet.class_cats(self.evaluate(X));
            acc = sum(Yh == Y) / numel(Y);
            return
        end
        
        function [ l_acts d_masks ] = feedforward(self, X, l_weights, do_drop)
            % Get per-layer activations for the observations in X, given the
            % weights in l_weights. If do_drop is 1, do mask-based dropping of
            % activations at each layer, and return the sampled masks.
            %
            if ~exist('do_drop','var')
                do_drop = 0;
            end
            l_acts = cell(1,self.depth);
            d_masks = cell(1,self.depth);
            l_acts{1} = X;
            d_masks{1} = ones(size(X));
            if ((do_drop == 1) && (self.drop_input > 1e-8))
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
                A_cur = func.feedforward(SGDevNet.bias(A_pre), W);
                l_acts{i} = A_cur;
                d_masks{i} = ones(size(A_cur));
                if (do_drop == 1)
                    % Set drop rate based on the current layer type. Don't do
                    % dropping at the output layer.
                    if ((i < self.depth) && (self.drop_hidden > 1e-8))
                        mask = rand(size(l_acts{i})) > self.drop_hidden;
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
                    prv_acts = SGDevNet.bias(l_acts{l_num-1});
                    prv_weights = l_weights{l_num-1};
                end
                if ((l_num < self.depth) && (l_num > 1))
                    % BP for internal layers (with post and pre layers)
                    func = self.act_func;
                    act_grads = l_grads{l_num};
                    nxt_weights = l_weights{l_num};
                    nxt_weights(end,:) = [];
                    nxt_grads = dN{l_num+1};
                    prv_acts = SGDevNet.bias(l_acts{l_num-1});
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
                    dW{l_num-1} = prv_acts' * cur_grads;
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
                if (i == 1)
                    drop_rate = self.drop_input;
                else
                    drop_rate = self.drop_hidden;
                end
                if ((abs(drop_rate-0.5) < 0.1) && (self.do_sde == 1))
                    % Halve the weights when net was trained with dropout rate
                    % near 0.5 for hidden nodes, to approximate sampling from
                    % the implied distribution over network architectures.
                    % Weights for first layer are not halved, as they modulate
                    % inputs from observed rather than hidden nodes.
                    % 
                    % Only do this when network was trained in standard dropout
                    % ensemble mode (i.e. self.do_sde == 1).
                    % 
                    W = W .* 0.5;
                end
                % Compute activations at the current layer via feedforward
                out_acts = func.feedforward(SGDevNet.bias(out_acts), W);
            end
            return
        end
        
        function [ dN L ] = bprop_output(self, A, Y)
            % Compute loss and gradients for the output-layer activations of
            % observations in A, with target outputs Y.
            %
            dN = cell(1,self.depth);
            L = 0;
            for i=1:self.depth,
                if (i < self.depth)
                    dN{i} = zeros(size(A{i}));
                else
                    [L_out dLdA_out] = self.out_loss(A{i}, Y);
                    dN{i} = dLdA_out;
                    L = L + L_out;
                end
            end
            return
        end
        
        function [ dN L ] = bprop_dev(self, A, batch_size, dev_reps)
            % Compute loss and gradients for dropout ensemble variance
            % regularization.
            %
            dN = cell(1,self.depth);
            L = 0;
            for i=1:self.depth,
                if (i == 1)
                    dN{i} = zeros(size(A{i}));
                else
                    lam_dev = self.layer_lams(i).lam_dev;
                    dev_type = self.layer_lams(i).dev_type;
                    [L_dev dLdA_dev] = ...
                        SGDevNet.drop_loss(A{i},batch_size,dev_reps,dev_type);
                    L = L + (lam_dev * L_dev);
                    dN{i} = (lam_dev * dLdA_dev);
                end
            end
            return
        end
        
        function [ dLdW L ] = bprop_weight_reg(self, l_weights)
            % Do a loss computation for the "general loss" incurred by the
            % per-layer, per-node activations A, weighted by smpl_wts.
            %
            dLdW = cell(1,length(l_weights));
            L = 0;
            for i=1:length(l_weights),
                W = l_weights{i};
                % Compute L2 regularization loss/gradient
                L = L + (self.lam_l2 * sum(sum(W.^2)));
                dLdW{i} = 2 * self.lam_l2 * W;
                % Compute (soft) L1 regularization loss/gradient
                L = L + (self.lam_l1 * sum(sum(sqrt(W.^2 + 1e-5))));
                dLdW{i} = dLdW{i} + (self.lam_l1 * (W ./ sqrt(W.^2 + 1e-5)));
            end
            return
        end
        
        function [L_out L_dev L_reg] = check_losses(self, X, Y, dev_reps)
            % Check the various losses being optimized for this SGDevNet
            if ~exist('dev_reps','var')
                dev_reps = 1;
            end
            l_weights = self.layer_weights;
            batch_size = 1000;
            % Compute loss at output layer
            [X_out Y_out] = SGDevNet.sample_points(X, Y, batch_size);
            if (self.do_sde == 1)
                A_out = self.feedforward(X_out, l_weights, 1);
            else
                A_out = self.feedforward(X_out, l_weights, 0);
            end
            [junk L_out] = self.bprop_output(A_out, Y_out);
            % Compute DEV loss
            if (self.do_dev == 1)
                X_dev = repmat(X_out,dev_reps,1);
                A_dev = self.feedforward(X_dev, l_weights, 1);
                [junk L_dev] = self.bprop_dev(A_dev, batch_size, dev_reps);
            else
                L_dev = 0;
            end
            % Compute network-wide weight regularization loss
            [junk L_reg] = self.bprop_weight_reg(l_weights);
            return
        end
            
        function [ result ] =  train(self, X, Y, params)
            % Do fully parameterized training for a SGDevNet
            if ~exist('params','var')
                params = struct();
            end
            params = SGDevNet.check_params(params);
            % Setup parameters for gradient updates (rates and momentums)
            all_mom = cell(1,self.depth-1);
            for i=1:(self.depth-1),
                all_mom{i} = zeros(size(self.layer_weights{i}));
            end
            rate = params.start_rate;
            batch_size = params.batch_size;
            dev_reps = params.dev_reps;
            do_validate = params.do_validate;
            dldw_norms = zeros(self.depth-1,params.rounds);
            lwts_norms = zeros(self.depth-1,params.rounds);
            max_sratio = zeros(self.depth-1,params.rounds);
            max_ratios = zeros(1,params.rounds);
            % Run update loop-a-doop
            fprintf('Updating weights (%d rounds):\n', params.rounds);
            for e=1:params.rounds,
                % Get the current network weights to use for this round
                l_weights = self.layer_weights;
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Sample points and do feedforward/backprop for general loss %
                % at the output layer (i.e. for classification/regression).  %
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                [X_out Y_out] = SGDevNet.sample_points(X, Y, batch_size);
                if (self.do_sde == 1)
                    [A_out M_out] = self.feedforward(X_out, l_weights, 1);
                else
                    [A_out M_out] = self.feedforward(X_out, l_weights, 0);
                end
                dN_out = self.bprop_output(A_out, Y_out);
                dLdW_out = self.backprop(A_out, l_weights, dN_out, M_out);
                if (self.do_dev == 1)
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % Compute DEV loss and gradients for the same points used %
                    % for loss/gradient computations at output layer.         %
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    X_dev = repmat(X_out,dev_reps,1);
                    [A_dev M_dev] = self.feedforward(X_dev, l_weights, 1);
                    dN_dev = self.bprop_dev(A_dev, batch_size, dev_reps);
                    dLdW_dev = self.backprop(A_dev, l_weights, dN_dev, M_dev);
                else
                    dLdW_dev = cell(1,length(dLdW_out));
                    for i=1:length(dLdW_out),
                        dLdW_dev{i} = zeros(size(dLdW_out{i}));
                    end
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Compute network-wide weight regularization loss/gradient %
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                dLdW_reg = self.bprop_weight_reg(l_weights);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Combine the gradients from all losses into a joint gradient %
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                dLdW = cell(1,length(dLdW_out));
                for l=1:length(dLdW_out),
                    dLdW{l} = dLdW_out{l} + dLdW_dev{l} + dLdW_reg{l};
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Compute updates for inter-layer weights using grads in dLdW %
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                layer_dW = cell(1,(self.depth-1));
                for l=1:(self.depth-1),
                    % Get relevant current weights and proposed update
                    l_weights = self.layer_weights{l};
                    l_dLdW = dLdW{l};
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
                    l2_bnd = self.layer_lams(l+1).l2_bnd;
                    wt_scales = l2_bnd ./ sqrt(sum(l_weights.^2,1));
                    l_weights = bsxfun(@times, l_weights, min(wt_scales,1));
                    % Store updated weights
                    self.layer_weights{l} = l_weights;
                end
                % Decay the learning rate after performing update
                rate = rate * params.decay_rate;
                % Occasionally recompute and display the loss and accuracy
                if ((e == 1) || (mod(e, 100) == 0))
                    [L_out L_dev L_reg] = self.check_losses(X,Y,dev_reps);
                    if (do_validate == 1)
                        [Lv_out Lv_dev] = ...
                            self.check_losses(params.Xv,params.Yv,dev_reps);
                        fprintf('    %d: T-(O=%.4f, D=%.4f, R=%.4f),',...
                            e, L_out, L_dev, L_reg);
                        fprintf(' V-(O=%.4f, D=%.4f)\n', Lv_out, Lv_dev);
                        [Xs Ys] = SGDevNet.sample_points(X, Y, 1000);
                        acc_t = self.check_acc(Xs,Ys);
                        [Xs Ys] = ...
                            SGDevNet.sample_points(params.Xv, params.Yv, 1000);
                        acc_v = self.check_acc(Xs,Ys);
                        fprintf('      acc_t: %.4f, acc_v: %.4f\n',acc_t,acc_v);
                    else
                        fprintf('    %d: O=%.4f, D=%.4f, R=%.4f\n',...
                            e, L_out, L_dev, L_reg);
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
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % DROPOUT ENSEMBLE VARIANCE LOSS %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [L dLdF] = drop_loss(F, b_obs, b_reps, dev_type)
            % Compute feature activations from droppy observations, and
            % grab a function handle for backpropping through activation
            %
            if ~exist('dev_type','var')
                dev_type = 1;
            end
            switch dev_type
                case 1
                    [F bp_F] = SGDevNet.norm_transform(F);
                case 2
                    [F bp_F] = SGDevNet.tanh_transform(F);
                case 3
                    [F bp_F] = SGDevNet.dont_transform(F);
                otherwise
                    error('Improperly specified dev_type');
            end 
            N = size(F,2);
            Ft = zeros(b_obs, N, b_reps);
            for i=1:b_reps,
                b_start = ((i-1) * b_obs) + 1;
                b_end = b_start + (b_obs - 1);
                Ft(:,:,i) = F(b_start:b_end,:);
            end
            % Compute mean of each repeated observations activations
            n = b_reps;
            m = (b_obs * b_reps * N);
            Fm = sum(Ft,3) ./ n;
            % Compute differences between individual activations and means
            Fd = bsxfun(@minus, Ft, Fm);
            % Compute droppy variance loss
            L = sum(Fd(:).^2) / m;
            % Compute droppy variance gradient (magic numbers everywhere!)
            dLdFt = -(2/m) * ((((1/n) - 1) * Fd) + ...
                ((1/n) * bsxfun(@minus, sum(Fd,3), Fd)));
            dLdF = zeros(size(F));
            for i=1:b_reps,
                b_start = ((i-1) * b_obs) + 1;
                b_end = b_start + (b_obs - 1);
                dLdF(b_start:b_end,:) = squeeze(dLdFt(:,:,i));
            end
            % Backprop through the transform determined by dev_type
            dLdF = bp_F(dLdF);
            return
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % OUTPUT LAYER LOSS FUNCTIONS %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [ L dL ] = loss_mclr(Yh, Y)
            % Compute a multiclass logistic regression loss and its gradients,
            % w.r.t. the proposed outputs Yh, given the true values Y.
            %
            obs_count = size(Yh,1);
            cl_count = size(Y,2);
            [Y_max Y_idx] = max(Y,[],2);
            P = bsxfun(@rdivide, exp(Yh), sum(exp(Yh),2));
            % Compute classification loss (deviance)
            p_idx = sub2ind(size(P), (1:obs_count)', Y_idx);
            L = -sum(sum(log(P(p_idx)))) / obs_count;
            if (nargout > 1)
                % Make a binary class indicator matrix
                Yi = bsxfun(@eq, Y_idx, 1:cl_count);
                % Compute the gradient of classification loss
                dL = (P - Yi) ./ obs_count;
            end
            return
        end
        
        function [ L dL ] = loss_mcl2h(Yh, Y)
            % Compute a multiclass L2 hinge loss and its gradients, w.r.t. the
            % proposed outputs Yh, given the true values Y.
            %
            obs_count = size(Yh,1);
            cl_count = size(Y,2);
            [Y_max Y_idx] = max(Y,[],2);
            % Make a class indicator matrix using +1/-1
            Yc = bsxfun(@(y1,y2) (2*(y1==y2))-1, Y_idx, 1:cl_count);
            % Compute current L2 hinge loss given the predictions in Yh
            margin_lapse = max(0, 1 - (Yc .* Yh));
            L = (0.5 * margin_lapse.^2);
            L = sum(sum(L)) / obs_count;
            if (nargout > 1)
                % For L2 hinge loss, dL is equal to the margin intrusion
                dL = -(Yc .* margin_lapse) ./ obs_count;
            end
            return
        end       
        
        function [ L dL ] = loss_lsq(Yh, Y)
            % Compute a least-sqaures regression loss and its gradients, for
            % each of the predicted outputs in Yh with true values Y.
            %
            obs_count = size(Yh,1);
            R = Yh - Y;
            L = R.^2;
            L = sum(sum(L)) / obs_count;
            if (nargout > 1)
                % Compute the gradient of least-squares loss
                dL = (2 * R) ./ obs_count;
            end
            return
        end
        
        function [ L dL ] = loss_hsq(Yh, Y, delta)
            % Compute a "huberized" least-sqaures regression loss and its
            % gradients, for each of the predicted outputs in Yh with true
            % values Y. This loss simply transitions from L2 to L1 loss for
            % element residuals greater than 1. This helps avoid descent
            % breaking oversized gradients.
            %
            if ~exist('delta','var')
                delta = 0.5;
            end
            obs_count = size(Yh,1);
            R = Yh - Y;
            mask = (abs(R) < delta);
            L = zeros(size(R));
            L(mask) = R(mask).^2;
            L(~mask) = (2 * delta * abs(R(~mask))) - delta^2;
            L = sum(sum(L)) / obs_count;
            if (nargout > 1)
                % Compute the gradient of huberized least-squares loss
                dL = zeros(size(R));
                dL(mask) = 2 * R(mask);
                dL(~mask) = (2 * delta) .* sign(R(~mask));
                dL = dL ./ obs_count;
            end
            return
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % HAPPY FUN BONUS FUNCTIONS %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [ Xb ] = bias(X, bias_val)
            % Add a column of constant bias to the observations in X
            if ~exist('bias_val','var')
                bias_val = 1;
            end
            Xb = [X (bias_val * ones(size(X,1),1))];
            return
        end
        
        function [ F BP ] = norm_transform(X)
            % L2 normalize X by rows, and return both the row-normalized matrix
            % and a function handle for backpropagating through normalization.
            N = sqrt(sum(X.^2,2) + 1e-6);
            F = bsxfun(@rdivide,X,N);
            % Backpropagate through normalization for unit norm
            BP = @( D ) ...
                (bsxfun(@rdivide,D,N) - bsxfun(@times,F,(sum(D.*X,2)./(N.^2))));
            return
        end
        
        function [ F BP ] = tanh_transform(X)
            % Transform the elements of X by hypertangent, and create a function
            % handle for backpropping through the transform.
            F = tanh(X);
            BP = @( D ) (D .* (1 - F.^2));
            return
        end
        
        function [ F BP ] = dont_transform(X)
            % Leave the values in X unchanged.
            F = X;
            BP = @( D ) (D .* ones(size(D)));
            return
        end
        
        function [ Xs Ys ] = sample_points(X, Y, sample_count)
            % Sample a batch of training observations.
            obs_count = size(X,1);
            idx = randsample(1:obs_count, sample_count, true);
            Xs = X(idx,:);
            Ys = Y(idx,:);
            return
        end
        
        function [ Yc ] = class_cats(Yi)
            % Convert +1/-1 indicator class matrix to a vector of categoricals
            [vals Yc] = max(Yi,[],2);
            return
        end
        
        function [ Yi ] = class_inds(Y, class_count)
            % Convert categorical class values into +1/-1 indicator matrix
            class_labels = sort(unique(Y),'ascend');
            if ~exist('class_count','var')
                class_count = numel(class_labels);
            end
            Yi = -ones(size(Y,1),class_count);
            for i=1:numel(class_labels),
                c_idx = (Y == class_labels(i));
                Yi(c_idx,i) = 1;
            end
            return
        end
        
        function [ Yi ] = to_inds( Yc )
            % This wraps class_cats and class_inds.
            Yi = SGDevNet.class_inds(Yc);
            Yc = SGDevNet.class_cats(Yi);
            Yi = SGDevNet.class_inds(Yc);
            return
        end
        
        function [ Yc ] = to_cats( Yc )
            % This wraps class_cats and class_inds.
            Yi = SGDevNet.class_inds(Yc);
            Yc = SGDevNet.class_cats(Yi);
            return
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % PARAMETER SETTING AND DEFAULT CHECKING %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [ params ] = check_params(params)
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
            if ~isfield(params, 'batch_size')
                params.batch_size = 100;
            end
            if ~isfield(params, 'dev_reps')
                params.dev_reps = 4;
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

