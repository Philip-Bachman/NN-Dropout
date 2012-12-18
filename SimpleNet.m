classdef SimpleNet < handle
    % This is a class for managing a simple multi-layer neural-net.
    %
    
    properties
        % act_func is an ActFunc instance for computing feed-forward activation
        % levels in hidden layers and backpropagating gradients
        act_func
        % out_func is an ActFunc instance for computing feed-forward activation
        % levels at the output layer
        out_func
        % loss_func is a LossFunc instance for computing loss values/gradients
        loss_func
        % depth is the number of layers (including in/out) in this neural-net
        depth
        % layer_sizes gives the size of each layer in this neural-net
        %   note: these sizes do _not_ include the bias
        layer_sizes
        % layer_weights is a cell array such that layer_weights{l} contains a
        % matrix in which entry (i,j) contains the weights between node i in
        % layer l and node j in layer l+1. The number of weight matrices in
        % layer_weights (i.e. its length) is self.depth - 1.
        %   note: due to biases each matrix in layer_weights has an extra row
        layer_weights
        % input_dim is the dimension of inputs to this network
        input_dim
        % output_dim is the dimension of outputs from this network
        output_dim
        % drop_stride controls node drop groupings
        drop_stride
        % half_weights tells whether to use half-weights during feedforward
        half_weights
    end
    
    methods
        function [self] = SimpleNet(layer_sizes, act_func, out_func, loss_func)
            % Constructor for SimpleNet class
            self.act_func = act_func;
            self.out_func = out_func;
            self.loss_func = loss_func;
            self.depth = numel(layer_sizes);
            self.layer_sizes = layer_sizes;
            self.input_dim = self.layer_sizes(1);
            self.output_dim = self.layer_sizes(end);
            self.drop_stride = 1;
            self.half_weights = 0;
            self.init_weights();
            return
        end
        
        function [ result ] = init_weights(self, weight_scale)
            % Initialize the hidden layers and connection weights for this
            % neural net.
            if ~exist('weight_scale','var')
                weight_scale = 0.1;
            end
            self.layer_weights = {};
            for i=1:(self.depth-1),
                % Add one to each outgoing layer weight count, for biases.
                weights = randn(self.layer_sizes(i)+1,self.layer_sizes(i+1));
                self.layer_weights{i} = weights .* weight_scale;
            end
            result = 0;
            return
        end
        
        function [ obs_acts ] = feedforward(self, X)
            % do a simple (i.e. no dropout) feed-forward computation for the
            % inputs in X
            obs_acts = X;
            for i=1:(self.depth-1),
                % Select activation function for the current layer
                if (i == self.depth-1)
                    func = self.out_func;
                else
                    func = self.act_func;
                end
                % Get weights connecting current layer to previous layer
                W = self.layer_weights{i};
                if (self.half_weights == 1)
                    W = W ./ 2;
                end
                % Compute activations at the current layer via feedforward
                obs_acts = func.feedforward(SimpleNet.bias(obs_acts), W);
            end
            return
        end
        
        function [ result ] = backprop(self, X, Y, dr_node, dr_obs)
            % Do a backprop computation with dropout for the data in X/Y.
            if ~exist('dr_node','var')
                dr_node = 0.0;
            end
            if ~exist('dr_obs','var')
                dr_obs = 0.0;
            end
            if (abs(dr_node - 0.5) < 0.1)
                self.half_weights = 1;
            else
                self.half_weights = 0;
            end
            obs_count = size(X,1);
            drop_weights = self.layer_weights;
            % Effect random observation and node dropping by zeroing afferent
            % and efferent weights for randomly selected observations/nodes
            for i=1:(self.depth-1),
                post_weights = drop_weights{i};
                if (i == 1)
                    % Do dropout at observation/input level
                    for n=1:(size(post_weights,1)-1),
                        if (rand() < dr_obs)
                            post_weights(n,:) = 0;
                        end
                    end      
                else
                    % Do dropout at hidden node level
                    pre_weights = drop_weights{i-1};
                    s = self.drop_stride;
                    for n=s:s:(size(post_weights,1)-1),
                        n_group = (n-(s-1)):n;
                        if (rand() < dr_node)
                            post_weights(n_group,:) = 0;
                            pre_weights(:,n_group) = 0;
                        end
                    end
                    drop_weights{i-1} = pre_weights;
                end
                drop_weights{i} = post_weights;
            end
            % Compute per-layer activations for the full observation set
            layer_acts = cell(1,self.depth);
            layer_acts{1} = X;
            for i=2:self.depth,
                if (i == self.depth)
                    func = self.out_func;
                else
                    func = self.act_func;
                end
                W = drop_weights{i-1};
                A = layer_acts{i-1};
                layer_acts{i} = func.feedforward(SimpleNet.bias(A), W);
            end
            % Compute gradients at all nodes, starting with loss values and
            % gradients for each observation at output layer
            node_grads = cell(1,self.depth);
            weight_grads = cell(1,self.depth-1);
            [L dL] = self.loss_func.evaluate(layer_acts{self.depth}, Y);
            for i=1:(self.depth-1),
                l_num = self.depth - i;
                if (l_num == (self.depth - 1))
                    func = self.out_func;
                    post_weights = 1;
                    post_grads = dL;
                else
                    func = self.act_func;
                    post_weights = drop_weights{l_num+1};
                    post_weights = post_weights(1:end-1,:);
                    post_grads = node_grads{l_num+1};
                end
                pre_acts = SimpleNet.bias(layer_acts{l_num});
                pre_weights = drop_weights{l_num};
                cur_grads = func.backprop(...
                    post_grads, post_weights, pre_acts, pre_weights);
                weight_grads{l_num} = pre_acts' * cur_grads;
                node_grads{l_num} = cur_grads;
            end
            % Normalize parameter gradients for batch size
            for i=1:(self.depth-1),
                weight_grads{i} = weight_grads{i} ./ obs_count;
            end
            result = struct();
            result.layer_grads = weight_grads;
            return
        end
        
        function [ Yh ] =  complex_update(self, X, Y, params)
            % Do fully parameterized training for a SimpleNet
            if ~exist('params','var')
                params = struct();
            end
            if ~isfield(params, 'epochs')
                params.epochs = 100;
            end
            if ~isfield(params, 'start_rate')
                params.start_rate = 1.0;
            end
            if ~isfield(params, 'decay_rate')
                params.decay_rate = 0.995;
            end
            if ~isfield(params, 'momentum')
                params.momentum = 0.25;
            end
            if ~isfield(params, 'weight_bound')
                params.weight_bound = 10;
            end
            if ~isfield(params, 'batch_size')
                params.batch_size = 100;
            end
            if ~isfield(params, 'dr_obs')
                params.dr_obs = 0.0;
            end
            if ~isfield(params, 'dr_node')
                params.dr_node = 0.0;
            end
            if ~isfield(params, 'do_validate')
                params.do_validate = 0;
            end
            if (params.do_validate == 1)
                if (~isfield(params, 'X_v') || ~isfield(params, 'Y_v'))
                    error('Validation set required for doing validation.');
                end
            end
            params.momentum = min(1, max(0, params.momentum));
            obs_count = size(X,1);
            rate = params.start_rate;
            dW_pre = cell(1,self.depth-1);
            fprintf('Updating weights (%d epochs):\n', params.epochs);
            for e=1:params.epochs,
                idx = randsample(obs_count, params.batch_size, false);
                Xtr = X(idx,:);
                Ytr = Y(idx,:);
                for r=1:params.batch_rounds,
                    % Run backprop to compute gradients for this training batch
                    res = self.backprop(...
                        Xtr, Ytr, params.dr_node, params.dr_obs);
                    for i=1:(self.depth-1),
                        % Update the weights at this layer using a momentum
                        % weighted mixture of the current gradients and the
                        % previous update.
                        l_grads = res.layer_grads{i};
                        l_weights = self.layer_weights{i};
                        if (e == 1)
                            dW = rate * l_grads;
                        else
                            dW = (params.momentum * dW_pre{i}) + ...
                                ((1 - params.momentum) * (rate * l_grads));
                        end
                        dW_pre{i} = dW;
                        l_weights = l_weights - dW;
                        % Force the collection of weights incident on each node
                        % on the outgoing side of this layer's weights to have
                        % norm bounded by params.weight_bound.
                        l_norms = sqrt(sum(l_weights.^2,1));
                        l_scales = min(1, (params.weight_bound ./ l_norms));
                        self.layer_weights{i} = ...
                            bsxfun(@times, l_weights, l_scales);
                    end
                end
                % Decay the learning rate after performing update
                rate = rate * params.decay_rate;
                % Compute and display the loss following this epoch
                if (mod(e, 50) == 0)
                    Yh = self.feedforward(X);
                    [max_vals Y_idx] = max(Y,[],2);
                    [max_vals Yh_idx] = max(Yh,[],2);
                    L = self.loss_func.evaluate(Yh, Y);
                    acc = sum(Y_idx == Yh_idx) / numel(Y_idx);
                    if (params.do_validate)
                        Yh_v = self.feedforward(params.X_v);
                        [max_vals Y_v_idx] = max(params.Y_v,[],2);
                        [max_vals Yh_v_idx] = max(Yh_v,[],2);
                        L_v = self.loss_func.evaluate(Yh_v, params.Y_v);
                        acc_v = sum(Yh_v_idx == Y_v_idx) / numel(Y_v_idx);
                        fprintf('    %d: t=(%.4f, %.4f) v=(%.4f, %.4f)\n',...
                            e, mean(L(:)), acc, mean(L_v(:)), acc_v);
                    else
                        fprintf('    %d: %.4f, %.4f\n', e, mean(L(:)), acc);
                    end
                end
            end
            fprintf('\n');
            Yh = self.feedforward(X);
            return
        end
        
    end
    
    methods (Static = true)
        function [ Xb ] = bias(X)
            % Add a column of constant bias to the observations in X
            Xb = [X ones(size(X,1),1)];
            return
        end
    end 
    
end










%%%%%%%%%%%%%%
% EYE BUFFER %
%%%%%%%%%%%%%%

