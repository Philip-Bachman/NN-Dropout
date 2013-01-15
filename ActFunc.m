classdef ActFunc < handle
    % This is a class for managing activation functions to be used by the hidden
    % and output layers of a neural-net.
    %
    
    properties
        % func_type determines which activation function to use
        func_type
    end
    
    methods
        function [self] = ActFunc( func_type )
            % Constructor for ActFunc class
            if ~exist('func_type','var')
                func_type = 1;
            end
            self.func_type = func_type;
            return
        end
        
        function [ acts ] = feedforward(self, pre_values, pre_weights)
            % Compute feed-forward activations according to some function
            switch self.func_type
                case 1
                    acts = ActFunc.linear_ff(pre_values, pre_weights);
                case 2
                    acts = ActFunc.sigmoid_ff(pre_values, pre_weights);
                case 3
                    acts = ActFunc.tanh_ff(pre_values, pre_weights);
                case 4
                    acts = ActFunc.logexp_ff(pre_values, pre_weights);
                case 5
                    acts = ActFunc.relu_ff(pre_values, pre_weights);
                case 6
                    acts = ActFunc.softmax_ff(pre_values, pre_weights);
                otherwise
                    error('No valid activation function type selected.');
            end
            return
        end
        
        function [ node_grads ] = backprop(...
                self, post_grads, post_weights, pre_values, pre_weights)
            % Backpropagate gradients through some activation function
            switch self.func_type
                case 1
                    node_grads = ActFunc.linear_bp(...
                        post_grads, post_weights, pre_values, pre_weights);
                case 2
                    node_grads = ActFunc.sigmoid_bp(...
                        post_grads, post_weights, pre_values, pre_weights);
                case 3
                    node_grads = ActFunc.tanh_bp(...
                        post_grads, post_weights, pre_values, pre_weights);
                case 4
                    node_grads = ActFunc.logexp_bp(...
                        post_grads, post_weights, pre_values, pre_weights);
                case 5
                    node_grads = ActFunc.relu_bp(...
                        post_grads, post_weights, pre_values, pre_weights);
                case 6
                    node_grads = ActFunc.softmax_bp(...
                        post_grads, post_weights, pre_values, pre_weights);
                otherwise
                    error('No valid activation function type selected.');
            end
            return
        end
    end
    
    methods (Static = true)
        % The static methods for ActFunc are feed-forwards and backprops
        %
        
        function [ cur_acts ] = linear_ff(pre_acts, pre_weights)
            % Compute simple linear activation function.
            %
            % Parameters:
            %   pre_acts: previous layer activations (obs_count x pre_dim)
            %   pre_weights: weights from pre -> cur (pre_dim x cur_dim)
            % Outputs:
            %   cur_acts: activations at current layer (obs_count x cur_dim)
            %
            cur_acts = pre_acts * pre_weights;
            return
        end
        
        function [ cur_grads ] = ...
                linear_bp(post_grads, post_weights, pre_acts, pre_weights)
            % Compute the gradient for each of the incident weights in
            % pre_weights given the precipitating gradient in post_grad
            % 
            % Parameters:
            %   post_grads: grads at next layer (obs_count x post_dim)
            %   post_weights: weights from current to post (cur_dim x post_dim)
            %   pre_acts: activations at previous layer (obs_count x pre_dim)
            %   pre_weights: weights from prev to current (pre_dim x cur_dim)
            % Outputs:
            %   cur_grads: gradients at current layer (obs_count x cur_dim)
            %
            cur_grads = post_grads * post_weights';
            return
        end
        
        function [ cur_acts ] = sigmoid_ff(pre_acts, pre_weights)
            % Compute simple sigmoid activation function.
            %
            % Parameters:
            %   pre_acts: previous layer activations (obs_count x pre_dim)
            %   pre_weights: weights from pre -> cur (pre_dim x cur_dim)
            % Outputs:
            %   cur_acts: activations at current layer (obs_count x cur_dim)
            %
            cur_acts = 1 ./ (1 + exp(-(pre_acts * pre_weights)));
            return
        end
        
        function [ cur_grads ] = ...
                sigmoid_bp(post_grads, post_weights, pre_acts, pre_weights)
            % Compute the gradient for each of the incident weights in
            % pre_weights given the precipitating gradient in post_grad
            % 
            % Parameters:
            %   post_grads: grads at next layer (obs_count x post_dim)
            %   post_weights: weights from current to post (cur_dim x post_dim)
            %   pre_acts: activations at previous layer (obs_count x pre_dim)
            %   pre_weights: weights from prev to current (pre_dim x cur_dim)
            % Outputs:
            %   cur_grads: gradients at current layer (obs_count x cur_dim)
            %
            e_mx = exp(-pre_acts * pre_weights);
            sig_grads = e_mx ./ (1 + e_mx).^2;
            cur_grads = sig_grads .* (post_grads * post_weights');
            return
        end
        
        function [ cur_acts ] = tanh_ff(pre_acts, pre_weights)
            % Compute simple hyperbolic tangent activation function.
            %
            % Parameters:
            %   pre_acts: previous layer activations (obs_count x pre_dim)
            %   pre_weights: weights from pre -> cur (pre_dim x cur_dim)
            % Outputs:
            %   cur_acts: activations at current layer (obs_count x cur_dim)
            %
            cur_acts = tanh(pre_acts * pre_weights);
            return
        end
        
        function [ cur_grads ] = ...
                tanh_bp(post_grads, post_weights, pre_acts, pre_weights)
            % Compute the gradient for each node in the current layer given
            % the gradients in post_grads for nodes at the next layer.
            % 
            % Parameters:
            %   post_grads: grads at next layer (obs_dim x post_dim)
            %   post_weights: weights from current to post (cur_dim x post_dim)
            %   pre_acts: activations at previous layer (obs_dim x pre_dim)
            %   pre_weights: weights from prev to current (pre_dim x cur_dim)
            % Outputs:
            %   cur_grads: gradients at current layer (obs_dim x cur_dim)
            %
            tanh_grads = 1 - (tanh(pre_acts * pre_weights)).^2;
            cur_grads = tanh_grads .* (post_grads * post_weights');
            return
        end
        
        function [ cur_acts ] = logexp_ff(pre_acts, pre_weights)
            % Compute simple logexp activation function log(1 + exp(x)).
            %
            % Parameters:
            %   pre_acts: previous layer activations (obs_count x pre_dim)
            %   pre_weights: weights from pre -> cur (pre_dim x cur_dim)
            % Outputs:
            %   cur_acts: activations at current layer (obs_count x cur_dim)
            %
            cur_acts = log(1 + exp(pre_acts * pre_weights));
            return
        end
        
        function [ cur_grads ] = ...
                logexp_bp(post_grads, post_weights, pre_acts, pre_weights)
            % Compute the gradient for each node in the current layer given
            % the gradients in post_grads for nodes at the next layer.
            % 
            % Parameters:
            %   post_grads: grads at next layer (obs_dim x post_dim)
            %   post_weights: weights from current to post (cur_dim x post_dim)
            %   pre_acts: activations at previous layer (obs_dim x pre_dim)
            %   pre_weights: weights from prev to current (pre_dim x cur_dim)
            % Outputs:
            %   cur_grads: gradients at current layer (obs_dim x cur_dim)
            %
            exp_vals = exp(pre_acts * pre_weights);
            logexp_grads = exp_vals ./ (exp_vals + 1);
            cur_grads = logexp_grads .* (post_grads * post_weights');
            return
        end
        
        function [ cur_acts ] = relu_ff(pre_acts, pre_weights)
            % Compute simple rectified linear activation function.
            %
            % Parameters:
            %   pre_acts: previous layer activations (obs_count x pre_dim)
            %   pre_weights: weights from pre -> cur (pre_dim x cur_dim)
            % Outputs:
            %   cur_acts: activations at current layer (obs_count x cur_dim)
            %
            cur_acts = max(0, pre_acts * pre_weights);
            return
        end
        
        function [ cur_grads ] = ...
                relu_bp(post_grads, post_weights, pre_acts, pre_weights)
            % Compute the gradient for each node in the current layer given
            % the gradients in post_grads for nodes at the next layer.
            % 
            % Parameters:
            %   post_grads: grads at next layer (obs_dim x post_dim)
            %   post_weights: weights from current to post (cur_dim x post_dim)
            %   pre_acts: activations at previous layer (obs_dim x pre_dim)
            %   pre_weights: weights from prev to current (pre_dim x cur_dim)
            % Outputs:
            %   cur_grads: gradients at current layer (obs_dim x cur_dim)
            %
            nz_acts = (pre_acts * pre_weights) > 0;
            cur_grads = (post_grads * post_weights') .* nz_acts;
            return
        end
        
        function [ cur_acts ] = softmax_ff(pre_acts, pre_weights)
            % Compute simple softmax activation function where each row in the
            % matrix (pre_acts * pre_weights) is "softmaxed".
            %
            % Parameters:
            %   pre_acts: previous layer activations (obs_count x pre_dim)
            %   pre_weights: weights from pre -> cur (pre_dim x cur_dim)
            % Outputs:
            %   cur_acts: activations at current layer (obs_count x cur_dim)
            %
            exp_vals = exp(pre_acts * pre_weights);
            cur_acts = bsxfun(@rdivide, exp_vals, sum(exp_vals,2));
            return
        end
        
        function [ cur_grads ] = ...
                softmax_bp(post_grads, post_weights, pre_acts, pre_weights)
            % Compute the gradient for each node in the current layer given
            % the gradients in post_grads for nodes at the next layer.
            % 
            % Parameters:
            %   post_grads: grads at next layer (obs_dim x post_dim)
            %   post_weights: weights from current to post (cur_dim x post_dim)
            %   pre_acts: activations at previous layer (obs_dim x pre_dim)
            %   pre_weights: weights from prev to current (pre_dim x cur_dim)
            % Outputs:
            %   cur_grads: gradients at current layer (obs_dim x cur_dim)
            %
            exp_vals = exp(pre_acts * pre_weights);
            sm_vals = bsxfun(@rdivide, exp_vals, sum(exp_vals,2));
            sm_grads = sm_vals .* (1 - sm_vals);
            cur_grads = sm_grads .* (post_grads * post_weights');
            return
        end
        
    end
    
end





%%%%%%%%%%%%%%
% EYE BUFFER %
%%%%%%%%%%%%%%

