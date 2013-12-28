classdef LineLayer < handle

    properties
        % act_trans gives the transform to apply after computing each of the
        % bilinear functions in this layer. Currently available are linear,
        % rectified linear, rectified huber, sigmoid, and hypertangent.
        % 
        act_trans
        weights
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
        function [ self ] = LineLayer(in_dim, out_dim, afun)
            self.act_trans = afun;
            self.dim_input = in_dim;
            self.dim_output = out_dim;
            self.init_weights(0.1);
            % Zero counters for timing stuff
            self.ff_evals = 0;
            self.bp_evals = 0;
            return
        end
        
        function [ N ] = weight_count(self)
            % Get the total number of weights in this layer.
            %
            N = (self.dim_input + 1) * self.dim_output;
            return
        end
        
        function [ Ws ] = init_weights(self, wt_scale, b_scale)
            % Initialize the weight struct for this layer.
            %
            if ~exist('b_scale','var')
                b_scale = wt_scale;
            end
            Ws = struct();
            Ws.w = wt_scale * randn(self.dim_output, self.dim_input);
            Ws.b = b_scale * randn(self.dim_output, 1);
            self.weights = Ws;
            return
        end
        
        function [ Ws ] = set_weights(self, W)
            % Set weights using the values in struct/vector W.
            %
            if isstruct(W)
                self.weights = W;
            else
                self.weights = self.struct_weights(W);
            end
            Ws = self.weights;
            return
        end
        
        function [ Wv ] = vector_weights(self, Ws)
            % Return a vectorized representation of the weight structure Ws,
            % which is assumed to come from this LineLayer instance.
            %
            % When no argument is given, return vectorized self.weights.
            %
            if ~exist('Ws','var')
                Ws = self.weights;
            end
            assert(((numel(Ws.w) == (self.dim_input*self.dim_output)) && ...
                (numel(Ws.b) == self.dim_output)), 'Invalid Ws');
            Wv = [Ws.w(:); Ws.b];
            return
        end
        
        function [ Ws ] = struct_weights(self, Wv)
            % Return a struct representation of the vectorized weights Wv,
            % which are assumed to follow the form of this LL instance.
            %
            % When no argument is given, return self.weights.
            %
            if ~exist('Wv','var')
                Ws = self.weights;
                return
            end
            assert((numel(Wv) == self.weight_count()),'Invalid Wv');
            i_dim = self.dim_input;
            o_dim = self.dim_output;
            Ws = struct();
            Ws.w = reshape(Wv(1:(i_dim*o_dim)),o_dim,i_dim);
            Ws.b = reshape(Wv(((i_dim*o_dim)+1):end),o_dim,1);
            return
        end
        
        function [ Ws ] = bound_weights(self, Ws, wt_bnd)
            % Bound the incoming weights to each node in this layer to reside
            % within a ball of fixed radius. Biases are not bounded.
            %
            return_struct = 0;
            if ~exist('Ws','var')
                Ws = self.weights;
            else
                if isstruct(Ws)
                    return_struct = 1;
                else
                    Ws = self.struct_weights(Ws);
                end
            end
            wt_bnd = wt_bnd^2;
            w_norms = sum(Ws.w.^2,2) + 1e-8;
            w_scales = min(1, (wt_bnd ./ w_norms));
            Ws.w = bsxfun(@times, Ws.w, w_scales);
            if (return_struct ~= 1)
                Ws = self.vector_weights(Ws);
            end
            return
        end
        
        function [ A_post A_pre ] = feedforward(self, X, Ws)
            % Compute feedforward activations for the inputs in X. Return both
            % the pre and post transform values.
            if ~exist('Ws','var')
                Ws = self.weights;
            else
                if ~isstruct(Ws)
                    Ws = self.struct_weights(Ws);
                end
            end
            A_pre = transpose(bsxfun(@plus, (Ws.w * X'), Ws.b));
            % Pass linear function outputs through self.act_trans.
            A_post = self.act_trans(A_pre, 'ff');
            % Update timing info
            self.ff_evals = self.ff_evals + size(X,1);
            return
        end
        
        function [ dLdW dLdX ] = backprop(self, dLdA, A, X, Ws)
            % Backprop through the linear functions and post-linear transforms
            % for this layer. Gradients on activations are given in dLdA, from
            % the activations A, which we assume were produced by feedforward
            % through this layer, with inputs X.
            %
            return_struct = 0;
            if ~exist('Ws','var')
                Ws = self.weights;
            else
                if isstruct(Ws)
                    return_struct = 1;
                else
                    Ws = self.struct_weights(Ws);
                end
            end
            dAdF = self.act_trans(A,'bp');
            dLdF = (dLdA .* dAdF);
            % Compute gradients with respect to linear function parameters
            dLdW = struct();
            dLdW.w = dLdF' * X;
            dLdW.b = transpose(sum(dLdF,1));
            % Compute gradients with respect to input matrix X
            dLdX = dLdF * Ws.w;
            if (return_struct ~= 1)
                % Vectorize the weight gradient structure
                dLdW = self.vector_weights(dLdW);
            end
            % Update timing info
            self.bp_evals = self.bp_evals + size(X,1);
            return
        end
        
        function [ dLdW dLdX ] = backprop_both(self, ...
                dLdA_post, dLdA_pre, A_post, X, Ws)
            % Backprop through the linear functions and post-linear transforms
            % for this layer. Gradients on activations are given in dLdA, from
            % the activations A, which we assume were produced by feedforward
            % through this layer, with inputs X.
            %
            return_struct = 0;
            if ~exist('Ws','var')
                Ws = self.weights;
            else
                if isstruct(Ws)
                    return_struct = 1;
                else
                    Ws = self.struct_weights(Ws);
                end
            end
            dAdF = self.act_trans(A_post,'bp');
            dLdF = (dLdA_post .* dAdF) + dLdA_pre;
            % Compute gradients with respect to linear function parameters
            dLdW = struct();
            dLdW.w = dLdF' * X;
            dLdW.b = transpose(sum(dLdF,1));
            % Compute gradients with respect to input matrix X
            dLdX = dLdF * Ws.w;
            if (return_struct ~= 1)
                % Vectorize the weight gradient structure
                dLdW = self.vector_weights(dLdW);
            end
            % Update timing info
            self.bp_evals = self.bp_evals + size(X,1);
            return
        end
        
        function [ result ] = check_grad(self, i_dim, o_dim, grad_checks)
            % Check backprop computation for this LineLayer
            %
            mf_opts = struct();
            mf_opts.Display = 'iter';
            mf_opts.Method = 'lbfgs';
            mf_opts.Corr = 10;
            mf_opts.Damped = 1;
            mf_opts.LS_type = 0;
            mf_opts.LS_init = 3;
            mf_opts.use_mex = 0;
            % Use minFunc's directional derivative gradient checking
            order = 1;
            type = 2;
            for i=1:grad_checks,
                fprintf('=============================================\n');
                fprintf('GRAD CHECK %d\n',i);
                % Generate a fake problem setting
                self.dim_input = i_dim;
                self.dim_output = o_dim;
                X = randn(500, i_dim);
                C = cov(randn(1000, o_dim));
                self.init_weights(0.1);
                w = self.vector_weights(self.weights);
                % Do check with respect to w
                fprintf('Checking wrt w\n');
                mf_func = @( W ) self.fake_loss_w(W, X(:), C);
                fastDerivativeCheck(mf_func,w(:),order,type);
                % Do check with respect to X
                fprintf('Checking wrt X\n');
                mf_func = @( x ) self.fake_loss_X(w(:), x,  C);
                fastDerivativeCheck(mf_func,X(:),order,type);
            end
            result = 1;
            return
        end
        
        function [ L dLdw ] = fake_loss_w(self, w, X, C)
            % Fake loss wrapper for gradient testing.
            i_dim = self.dim_input;
            obs_count = numel(X) / i_dim;
            X = reshape(X, obs_count, i_dim);
            A = self.feedforward(X, w);
            AC = A*C;
            L = sum(sum((A .* AC), 2)) / size(C,1);
            dLdA = 2*AC;
            dLdw = self.backprop(dLdA, A, X, w);
            dLdw = dLdw(:) ./ size(C,1);
            return
        end
        
        function [ L dLdX ] = fake_loss_X(self, w, X, C)
            % Fake loss wrapper for gradient testing.
            i_dim = self.dim_input;
            obs_count = numel(X) / i_dim;
            X = reshape(X, obs_count, i_dim);
            A = self.feedforward(X, w);
            AC = A*C;
            L = sum(sum((A .* AC), 2)) / size(C,1);
            dLdA = 2*AC;
            [dLdW dLdX] = self.backprop(dLdA, A, X, w);
            dLdX = dLdX(:) ./ size(C,1);
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
