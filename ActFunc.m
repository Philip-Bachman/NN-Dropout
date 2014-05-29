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
                    acts = ActFunc.rehu_ff(pre_values, pre_weights);
                case 7
                    acts = ActFunc.norm_rehu_ff(pre_values, pre_weights);
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
                    node_grads = ActFunc.rehu_bp(post_grads, post_weights,...
                        pre_values, pre_weights, act_grads);
                case 7
                    node_grads = ActFunc.norm_rehu_bp(post_grads,...
                        post_weights, pre_values, pre_weights, act_grads);
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
            cur_acts = (1 ./ (1 + exp(-(pre_acts * pre_weights))));
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
        
        function [ cur_acts ] = rehu_ff(pre_acts, pre_weights)
            % Compute simple rectified Huber activation function.
            %
            % Parameters:
            %   pre_acts: previous layer activations (obs_count x pre_dim)
            %   pre_weights: weights from pre -> cur (pre_dim x cur_dim)
            % Outputs:
            %   cur_acts: activations at current layer (obs_count x cur_dim)
            %
            cur_acts = pre_acts * pre_weights;
            cur_acts = bsxfun(@max, cur_acts, 0);
            quad_mask = bsxfun(@lt, cur_acts, 0.5);
            line_mask = bsxfun(@ge, cur_acts, 0.5);
            cur_acts = (quad_mask .* cur_acts.^2) + ...
                (line_mask .* (cur_acts - 0.25));
            return
        end
        
        function [ dLdF ] = rehu_bp(post_grads, post_weights, ...
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
            dAdF = max(0, (pre_acts * pre_weights));
            dAdF = 2 * dAdF;
            dAdF(dAdF > 1) = 1;
            dLdA = (post_grads * post_weights') + act_grads;
            dLdF = dLdA .* dAdF;
            return
        end
        
        function [ cur_acts ] = norm_rehu_ff(pre_acts, pre_weights)
            % Compute simple normalized rectified Huber activation function.
            %
            % Parameters:
            %   pre_acts: previous layer activations (obs_count x pre_dim)
            %   pre_weights: weights from pre -> cur (pre_dim x cur_dim)
            % Outputs:
            %   cur_acts: activations at current layer (obs_count x cur_dim)
            %
            EPS = 1e-3;
            cur_acts = pre_acts * pre_weights;
            cur_acts = bsxfun(@max, cur_acts, 0);
            quad_mask = bsxfun(@lt, cur_acts, 0.5);
            line_mask = bsxfun(@ge, cur_acts, 0.5);
            cur_acts = (quad_mask .* cur_acts.^2) + ...
                (line_mask .* (cur_acts - 0.25));
            act_norms = sqrt(sum(cur_acts.^2,2) + EPS);
            cur_acts = bsxfun(@rdivide, cur_acts, act_norms);
            return
        end
        
        function [ dLdF ] = norm_rehu_bp(post_grads, post_weights, ...
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
            EPS = 1e-3;
            F = pre_acts * pre_weights;
            F = bsxfun(@max, F, 0);
            quad_mask = bsxfun(@lt, F, 0.5);
            line_mask = bsxfun(@ge, F, 0.5);
            A1 = (quad_mask .* F.^2) + ...
                (line_mask .* (F - 0.25));
            A1N = sqrt(sum(A1.^2,2) + EPS);
            A2 = bsxfun(@rdivide, A1, A1N);
            % Compute 
            dA1dF = 2*(quad_mask .* F) + line_mask;
            dLdA2 = (post_grads * post_weights') + act_grads;
            V = dLdA2 .* A1;
            V = sum(V, 2);
            dLdA1 = bsxfun(@rdivide, dLdA2, A1N) - ...
                bsxfun(@times, A2, (V ./ (A1N.^2.0)));
            dLdF = dLdA1 .* dA1dF;
            return
        end

        
    end
    
end





%%%%%%%%%%%%%%
% EYE BUFFER %
%%%%%%%%%%%%%%

