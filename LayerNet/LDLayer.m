classdef LDLayer < handle

    properties
        % act_trans gives the transform to apply after computing each of the
        % bilinear functions in this layer. Currently available are linear,
        % rectified linear, rectified huber, sigmoid, and hypertangent.
        % 
        act_trans
        % W stores the weights for this layer. W(i,j) gives the weight of the
        % connection from the jth node in the previous layer to the ith node in
        % this layer. W(:,end) gives bias weights.
        W
        % dim_input gives the dimension of the input to this layer. In the
        % linear case, dim_input is the integer dimension of the output of the
        % previous network layer (or the input if this is first layer).
        dim_input
        % dim_output gives the number of outputs produced by this layer. This
        % effectively is equivalent to the number of nodes in the layer.
        dim_output
        % ff_evals counts the total # of examples passed through feedforward
        ff_evals
        % bp_evals counts the total # of examples passed through backprop
        bp_evals
    end % END PROPERTIES
    
    methods
        function [ self ] = LDLayer(in_dim, out_dim, afun)
            self.act_trans = afun;
            self.dim_input = in_dim;
            self.dim_output = out_dim;
            self.init_weights(0.1,0.01);
            % Zero counters for timing stuff
            self.ff_evals = 0;
            self.bp_evals = 0;
            return
        end
        
        function [ N ] = weight_count(self)
            % Get the total number of weights in this layer.
            %
            N = self.dim_input * self.dim_output;
            return
        end
        
        function [ Wm ] = init_weights(self, wt_scale, b_scale, do_kill)
            % Initialize the weight struct for this layer.
            %
            if ~exist('do_kill','var')
                do_kill = 0;
            end
            Wm = wt_scale * randn(self.dim_output, self.dim_input);
            Wm(:,end) = b_scale;
            if (do_kill == 1)
                for i=1:size(Wm,1),
                    keep_count = 50;
                    if (keep_count < size(Wm,2))
                        keep_idx = randperm((size(Wm,2)-1));
                        kill_idx = keep_idx(51:end);
                        Wm(i,kill_idx) = 0.1 * Wm(i,kill_idx);
                    end
                end
            end
            self.W = Wm;
            return
        end
        
        function [ Wm ] = set_weights(self, Wm)
            % Set weights using the values in matrix/vector Wm.
            %
            assert((numel(Wm) == self.weight_count()),'Bad Wm.');
            if ((size(Wm,1) == 1) || (size(Wm,2) == 1))
                Wm = reshape(Wm,self.dim_output, self.dim_input);
            end
            self.W = Wm;
            return
        end
        
        function [ Wv ] = vector_weights(self, Wm)
            % Return a vectorized representation of the weight matrix Wm,
            %
            if ~exist('Wm','var')
                Wm = self.W;
            end
            assert((numel(Wm) == self.weight_count()),'Bad Wm.');
            Wv = Wm(:);
            return
        end
        
        function [ Wm ] = matrix_weights(self, Wv)
            % Return the vectorized weights Wv, reshaped to a matrix.
            %
            if ~exist('Wv','var')
                Wm = self.W;
                return
            end
            assert((numel(Wv) == self.weight_count()),'Invalid Wv');
            Wm = reshape(Wv, self.dim_output, self.dim_input);
            return
        end
        
        function [ Wm ] = bound_weights(self, Wm, wt_bnd)
            % Bound the incoming weights to each node in this layer to reside
            % within a ball of fixed radius.
            %
            if ~exist('Wm','var')
                Wm = self.W;
            end
            w_norms = sqrt(sum(Wm.^2,2) + 1e-8);
            w_scales = min(1, (wt_bnd ./ w_norms));
            Wm = bsxfun(@times, Wm, w_scales);
            return
        end
        
        function [ A_post A_pre ] = feedforward(self, X, Wm)
            % Compute feedforward activations for the inputs in X. Return both
            % the pre and post transform values.
            A_pre = X * Wm';
            % Pass linear function outputs through self.act_trans.
            A_post = self.act_trans(A_pre, 'ff');
            % Update timing info
            self.ff_evals = self.ff_evals + size(X,1);
            return
        end
        
        function [ dLdW dLdX ] = ...
                backprop(self, dLdA_post, dLdA_pre, A_post, X, Wm)
            % Backprop through the linear functions and post-linear transforms
            % for this layer.
            %
            dAdF = self.act_trans(A_post, 'bp');
            dLdF = (dLdA_post .* dAdF) + dLdA_pre;
            % Compute gradients with respect to linear function parameters
            dLdW = dLdF' * X;
            % Compute gradients with respect to input matrix X
            dLdX = dLdF * Wm;
            % Update timing info
            self.bp_evals = self.bp_evals + size(X,1);
            return
        end
        
    end % END INSTANCE METHODS
    
    methods (Static = true)
        
        function [ F ] = tanh_trans(X, comp_type)
            % Transform the elements of X by hypertangent.
            assert((strcmp(comp_type,'ff')||strcmp(comp_type,'bp')),'ff/bp?');
            if (strcmp(comp_type,'ff'))
                % Do feedforward
                F = tanh(X);
            else
                % Do backprop
                F = 1 - X.^2;
            end
            return
        end
        
        function [ F ] = relu_trans(X, comp_type)
            % Leave the values in X unchanged. Or, backprop through the
            % non-transform.
            assert((strcmp(comp_type,'ff')||strcmp(comp_type,'bp')),'ff/bp?');
            if (strcmp(comp_type,'ff'))
                % Do feedforward
                F = max(X, 0);
            else
                % Do backprop
                F = double(X > 0);
            end
            return
        end
        
        function [ F ] = rehu_trans(X, comp_type)
            % Leave the values in X unchanged. Or, backprop through the
            % non-transform.
            assert((strcmp(comp_type,'ff')||strcmp(comp_type,'bp')),'ff/bp?');
            if (strcmp(comp_type,'ff'))
                % Do feedforward
                F = max((X-0.25), 0);
                mask = (X > 0) & (X < 0.5);
                F(mask) = X(mask).^2;
            else
                % Do backprop
                mask = (X < 0.25) & (X > 1e-10);
                F = double(X > 0);
                F(mask) = 2*sqrt(X(mask));
            end
            return
        end
        
        function [ F ] = line_trans(X, comp_type)
            % Leave the values in X unchanged. Or, backprop through the
            % non-transform.
            assert((strcmp(comp_type,'ff')||strcmp(comp_type,'bp')),'ff/bp?');
            if (strcmp(comp_type,'ff'))
                % Do feedforward
                F = X;
            else
                % Do backprop
                F = ones(size(X));
            end
            return
        end
        
    end % END STATIC METHODS
    
        
end





%%%%%%%%%%%%%%
% EYE BUFFER %
%%%%%%%%%%%%%%
