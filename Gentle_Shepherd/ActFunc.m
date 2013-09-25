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
        
        function [ node_grads ] = backprop(self, post_grads, post_weights,...
                pre_values, pre_weights, act_grads)
            % Backpropagate gradients through some activation function
            %
            % BP functions take the following arguments:
            %   post_grads: gradients on the pre-transform activation at the
            %               next layer's nodes.
            %   post_weights: weights from the current layer's nodes to the next
            %                 layer's nodes. size: (cur_dim x nxt_dim)
            %   pre_values: post-transform activations at the previous layer's
            %               nodes.
            %   pre_weights: weights from the previous layer's nodes to the
            %                current layer's nodes. size: (pre_dim x cur_dim)
            %   act_grads: gradients directly on the post-transform activations
            %              at the current layer's nodes.
            obs_count = size(post_grads,1);
            cur_dim = max(size(pre_weights,2),size(post_weights,1));
            if ~exist('act_grads','var')
                act_grads = zeros(obs_count,cur_dim);
            end
            switch self.func_type
                case 1
                    node_grads = ActFunc.linear_bp(post_grads, post_weights,...
                        pre_values, pre_weights, act_grads);
                case 2
                    node_grads = ActFunc.sigmoid_bp(post_grads, post_weights,...
                        pre_values, pre_weights, act_grads);
                case 3
                    node_grads = ActFunc.tanh_bp(post_grads, post_weights,...
                        pre_values, pre_weights, act_grads);
                case 4
                    node_grads = ActFunc.logexp_bp(post_grads, post_weights,...
                        pre_values, pre_weights, act_grads);
                case 5
                    node_grads = ActFunc.relu_bp(post_grads, post_weights,...
                        pre_values, pre_weights, act_grads);
                case 6
                    node_grads = ActFunc.softmax_bp(post_grads, post_weights,...
                        pre_values, pre_weights, act_grads);
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
        
        function [ dLdF ] = linear_bp(post_grads, post_weights, ...
                pre_acts, pre_weights, act_grads)
            % Compute the gradients w.r.t. the pre-transform activations at all
            % nodes in the current layer.
            % 
            % Parameters:
            %   post_grads: loss gradients on pre-transform activations at each
            %               node in the next layer (obs_count x post_dim)
            %   post_weights: weights from nodes in current layer to nodes in 
            %                 next layer (cur_dim x post_dim)
            %   pre_acts: post-transform activations for each node in the
            %             previous layer (obs_count x pre_dim)
            %   pre_weights: weights from nodes in previous layer to nodes in 
            %                current layer (pre_dim x cur_dim)
            %   act_grads: direct loss gradients on post-transform activations
            %              for each node in current layer (obs_count x cur_dim)
            % Outputs:
            %   dLdF: gradients w.r.t pre-transform node activations at current
            %         layer (obs_count x cur_dim)
            %
            dLdA = (post_grads * post_weights') + act_grads;
            dAdF = 1;
            dLdF = dLdA .* dAdF;
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
        
        function [ dLdF ] = sigmoid_bp(post_grads, post_weights, ...
                pre_acts, pre_weights, act_grads)
            % Compute the gradients w.r.t. the pre-transform activations at all
            % nodes in the current layer.
            % 
            % Parameters:
            %   post_grads: loss gradients on pre-transform activations at each
            %               node in the next layer (obs_count x post_dim)
            %   post_weights: weights from nodes in current layer to nodes in 
            %                 next layer (cur_dim x post_dim)
            %   pre_acts: post-transform activations for each node in the
            %             previous layer (obs_count x pre_dim)
            %   pre_weights: weights from nodes in previous layer to nodes in 
            %                current layer (pre_dim x cur_dim)
            %   act_grads: direct loss gradients on post-transform activations
            %              for each node in current layer (obs_count x cur_dim)
            % Outputs:
            %   dLdF: gradients w.r.t pre-transform node activations at current
            %         layer (obs_count x cur_dim)
            %
            e_mx = exp(-pre_acts * pre_weights);
            dLdA = (post_grads * post_weights') + act_grads;
            dAdF = e_mx ./ (1 + e_mx).^2;
            dLdF = dLdA .* dAdF;
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
        
        function [ dLdF ] = tanh_bp(post_grads, post_weights, ...
                pre_acts, pre_weights, act_grads)
            % Compute the gradients w.r.t. the pre-transform activations at all
            % nodes in the current layer.
            % 
            % Parameters:
            %   post_grads: loss gradients on pre-transform activations at each
            %               node in the next layer (obs_count x post_dim)
            %   post_weights: weights from nodes in current layer to nodes in 
            %                 next layer (cur_dim x post_dim)
            %   pre_acts: post-transform activations for each node in the
            %             previous layer (obs_count x pre_dim)
            %   pre_weights: weights from nodes in previous layer to nodes in 
            %                current layer (pre_dim x cur_dim)
            %   act_grads: direct loss gradients on post-transform activations
            %              for each node in current layer (obs_count x cur_dim)
            % Outputs:
            %   dLdF: gradients w.r.t pre-transform node activations at current
            %         layer (obs_count x cur_dim)
            %
            dAdF = 1 - (tanh(pre_acts * pre_weights)).^2;
            dLdA = (post_grads * post_weights') + act_grads;
            dLdF = dLdA .* dAdF;
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
        
        function [ dLdF ] = logexp_bp(post_grads, post_weights, ...
                pre_acts, pre_weights, act_grads)
            % Compute the gradients w.r.t. the pre-transform activations at all
            % nodes in the current layer.
            % 
            % Parameters:
            %   post_grads: loss gradients on pre-transform activations at each
            %               node in the next layer (obs_count x post_dim)
            %   post_weights: weights from nodes in current layer to nodes in 
            %                 next layer (cur_dim x post_dim)
            %   pre_acts: post-transform activations for each node in the
            %             previous layer (obs_count x pre_dim)
            %   pre_weights: weights from nodes in previous layer to nodes in 
            %                current layer (pre_dim x cur_dim)
            %   act_grads: direct loss gradients on post-transform activations
            %              for each node in current layer (obs_count x cur_dim)
            % Outputs:
            %   dLdF: gradients w.r.t pre-transform node activations at current
            %         layer (obs_count x cur_dim)
            %
            exp_vals = exp(pre_acts * pre_weights);
            dAdF = exp_vals ./ (exp_vals + 1);
            dLdA = (post_grads * post_weights') + act_grads;
            dLdF = dLdA .* dAdF;
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
            cur_acts = pre_acts * pre_weights;
            cur_acts = max(0, cur_acts);
            return
        end
        
        function [ dLdF ] = relu_bp(post_grads, post_weights, ...
                pre_acts, pre_weights, act_grads)
            % Compute the gradients w.r.t. the pre-transform activations at all
            % nodes in the current layer.
            % 
            % Parameters:
            %   post_grads: loss gradients on pre-transform activations at each
            %               node in the next layer (obs_count x post_dim)
            %   post_weights: weights from nodes in current layer to nodes in 
            %                 next layer (cur_dim x post_dim)
            %   pre_acts: post-transform activations for each node in the
            %             previous layer (obs_count x pre_dim)
            %   pre_weights: weights from nodes in previous layer to nodes in 
            %                current layer (pre_dim x cur_dim)
            %   act_grads: direct loss gradients on post-transform activations
            %              for each node in current layer (obs_count x cur_dim)
            % Outputs:
            %   dLdF: gradients w.r.t pre-transform node activations at current
            %         layer (obs_count x cur_dim)
            %
            dAdF = (pre_acts * pre_weights) > 0;
            dLdA = (post_grads * post_weights') + act_grads;
            dLdF = dLdA .* dAdF;
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
        
        function [ dLdF ] = softmax_bp(post_grads, post_weights, ...
                pre_acts, pre_weights, act_grads)
            % Compute the gradients w.r.t. the pre-transform activations at all
            % nodes in the current layer.
            % 
            % Parameters:
            %   post_grads: loss gradients on pre-transform activations at each
            %               node in the next layer (obs_count x post_dim)
            %   post_weights: weights from nodes in current layer to nodes in 
            %                 next layer (cur_dim x post_dim)
            %   pre_acts: post-transform activations for each node in the
            %             previous layer (obs_count x pre_dim)
            %   pre_weights: weights from nodes in previous layer to nodes in 
            %                current layer (pre_dim x cur_dim)
            %   act_grads: direct loss gradients on post-transform activations
            %              for each node in current layer (obs_count x cur_dim)
            % Outputs:
            %   dLdF: gradients w.r.t pre-transform node activations at current
            %         layer (obs_count x cur_dim)
            %
            exp_vals = exp(pre_acts * pre_weights);
            sm_vals = bsxfun(@rdivide, exp_vals, sum(exp_vals,2));
            dAdF = sm_vals .* (1 - sm_vals);
            dLdA = (post_grads * post_weights') + act_grads;
            dLdF = dLdA .* dAdF;
            return
        end
        
    end
    
end





%%%%%%%%%%%%%%
% EYE BUFFER %
%%%%%%%%%%%%%%

