classdef SyBiLayer < handle
    % Symmetric bililinear layer, ie feedforward(X1,X2) and feedforward(X2,X1)
    % are constrained to be the same. This roughly halves the number of weights
    % (total number of biases stays the same), and may be useful for learning
    % "non-linear" dot products for embeddings (e.g. for recommender systems).
    %

    properties
        % act_trans gives the transform to apply after computing each of the
        % bilinear functions in this layer. Currently available are linear,
        % rectified linear, rectified huber, sigmoid, and hypertangent.
        % 
        act_trans
        weights
        % dim_input gives the dimension of the input to this layer. In the
        % bilinear case, dim_input contains two integers, with dim_input(1)
        % giving the dimension of the "left" input "a1" and dim_input(2) giving
        % the dimension of the "right" input "a2", and each node in this layer
        % computes ((a1' * W * a2) + (w' * [a1; a2]) + b) for some W, w, b.
        dim_input
        % dim_output gives the number of outputs produced by this layer. This
        % effectively is equivalent to the number of nodes in the layer.
        dim_output
    end % END PROPERTIES
    
    methods
        function [ self ] = SyBiLayer(in_dims, out_dim, afun)
            self.act_trans = afun;
            assert((in_dims(1)==in_dims(2)),'Unbalanced SyBiLayer input dims.');
            self.dim_input = in_dims;
            self.dim_output = out_dim;
            self.init_weights(0.1);
            return
        end
        
        function [ N ] = weight_count(self)
            % Get the total number of weights in this layer.
            %
            l_dim = self.dim_input(1);
            r_dim = self.dim_input(2);
            N = (l_dim * r_dim) + (l_dim + r_dim) + 1;
            N = N * self.dim_output;
            return
        end
        
        function [ Ws ] = init_weights(self, wt_scale, b_scale)
            % Initialize the weight struct for this layer.
            %
            if ~exist('b_scale','var')
                b_scale = wt_scale;
            end
            Ws = struct();
            l_dim = self.dim_input(1);
            r_dim = self.dim_input(2);
            for i=1:self.dim_output,
                Wi = wt_scale * randn(l_dim,r_dim);
                wi = wt_scale * randn(l_dim,1);
                Ws(i).W = (Wi + Wi') ./ 2;
                Ws(i).w = [wi; wi];
                Ws(i).b = b_scale * randn();
            end
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
            % which is assumed to come from this SyBiLayer instance.
            %
            % If no argumnet is given, return vectorized self.weights.
            %
            if ~exist('Ws','var')
                Ws = self.weights;
            end
            assert((length(Ws) == self.dim_output), 'Invalid Ws');
            Wv = [];
            for i=1:length(Ws),
                Wv = [Wv; Ws(i).W(:)];
                Wv = [Wv; Ws(i).w(:)];
                Wv = [Wv; Ws(i).b];
            end
            return
        end
        
        function [ Ws ] = struct_weights(self, Wv)
            % Return a struct representation of the vectorized weights Wv,
            % which are assumed to follow the form of this BL instance.
            %
            % If no argument is given, return self.weights.
            %
            if ~exist('Wv','var')
                Ws = self.weights;
                return
            end
            assert((numel(Wv) == self.weight_count()),'Invalid Wv');
            o_dim = self.dim_output;
            l_dim = self.dim_input(1);
            r_dim = self.dim_input(2);
            % Compute total number of parameters per node/output.
            node_wts = (l_dim * r_dim) + (l_dim + r_dim) + 1;
            % Structurize the weight vector Wv
            Ws = struct();
            for i=1:o_dim,
                n_start = ((i-1) * node_wts) + 1;
                n_end = n_start + (node_wts - 1);
                % Grab the weights for this node
                Wn_all = Wv(n_start:n_end);
                % Put them in the structure
                Wn = Wn_all(1:(l_dim*r_dim));
                Ws(i).W = reshape(Wn,l_dim,r_dim);
                wn = Wn_all(((l_dim*r_dim)+1):((l_dim*r_dim)+(l_dim+r_dim)));
                Ws(i).w = reshape(wn,(l_dim+r_dim),1);
                Ws(i).b = Wn_all(end);
            end
            return
        end
        
        function [ A_post A_pre ] = feedforward(self, X1, X2, Wv)
            % Compute feedforward activations for the inputs in X1/X2 where
            % each row of X1 gives a "left" input and the corresponding row of
            % X2 gives it's partner "right" input.
            %
            if ~exist('Wv','var')
                Ws = self.weights;
            else
                Ws = self.struct_weights(Wv);
            end
            l_dim = self.dim_input(1);
            r_dim = self.dim_input(2);
            assert((size(X1,1) == size(X2,1)),'Mismatched X1/X2 size.');
            assert(((size(X1,2) == l_dim) && (size(X2,2) == r_dim)), ...
                'Invalid X1/X2 dim.');
            in_count = size(X1,1);
            o_dim = self.dim_output;
            A_pre = zeros(in_count, o_dim);
            for i=1:o_dim,
                Wi = Ws(i).W;
                wi = Ws(i).w;
                bi = Ws(i).b;
                A_pre(:,i) = sum((X1 .* (Wi * X2')'),2) + ([X1 X2] * wi) + bi;
            end
            % Pass bilinear function outputs through self.act_trans.
            A_post = self.act_trans(A_pre, 'ff');
            return
        end
        
        function [ dLdW dLdX1 dLdX2 ] = backprop(self, dLdA, A, X1, X2, Wv)
            % Backprop through the bilinear functions and post-bilinear
            % transforms for this layer. Gradients on activations are given in
            % dLdA, from the activations A, which we assume were produced by
            % feedforward through this layer, with inputs X1/X2.
            %
            if ~exist('Wv','var')
                Ws = self.weights;
            else
                Ws = self.struct_weights(Wv);
            end
            o_dim = self.dim_output;
            l_dim = self.dim_input(1);
            r_dim = self.dim_input(2);
            assert(((o_dim==size(dLdA,2)) && (o_dim==size(A,2)) && ...
                (size(dLdA,1)==size(A,1)) && (size(A,1)==size(X1,1)) && ...
                (size(X1,1)==size(X2,1)) && (size(X1,2)==l_dim) && ...
                (size(X2,2)==r_dim)), 'Wrong arg sizes for BL backprop.');
            % First, backprop dLdA through self.act_trans, assuming the
            % activations in A came from X1/X2 and induced dLdA.
            dAdF = self.act_trans(A,'bp');
            dLdF = dLdA .* dAdF;
            % Backprop onto weights through this layer's bilinear functions
            dLdW = struct();
            for i=1:o_dim,
                dLdW(i).W = zeros(size(Ws(i).W));
                dLdW(i).w = zeros(size(Ws(i).w));
                dLdW(i).b = 0;
            end
            for i=1:size(X1,1),
                dFdWi = X1(i,:)' * X2(i,:);
                dFdWi = (dFdWi + dFdWi') ./ 2;
                dFdwi = (X1(i,:) + X2(i,:)) ./ 2;
                dFdwi = [dFdwi'; dFdwi'];
                dLdFi = dLdF(i,:);
                % Do loopy updates of this layer's weight gradients
                for o=1:o_dim,
                    dLdW(o).W = dLdW(o).W + (dLdFi(o) * dFdWi);
                    dLdW(o).w = dLdW(o).w + (dLdFi(o) * dFdwi);
                    dLdW(o).b = dLdW(o).b + dLdFi(o);
                end
            end
%             % Symmetrize gradients in each node.
%             for i=1:o_dim,
%                 dWi = dLdW(i).W;
%                 dLdW(i).W = (dWi + dWi') ./ 2;
%                 dwi = dLdW(i).w;
%                 dwi = (dwi(1:l_dim)+dwi((l_dim+1):end)) ./ 2;
%                 dLdW(i).w = [dwi; dwi];
%             end
            % Backprop onto inputs through this layer's bilinear functions
            dLdX1 = zeros(size(X1));
            dLdX2 = zeros(size(X2));
            for i=1:o_dim,
                Wi = Ws(i).W;
                wi = Ws(i).w;
                dFidX1 = bsxfun(@plus, (Wi*X2')', wi(1:l_dim)');
                dLdX1 = dLdX1 + bsxfun(@times, dLdF(:,i), dFidX1);
                dFidX2 = bsxfun(@plus, (X1*Wi), wi((l_dim+1):end)');
                dLdX2 = dLdX2 + bsxfun(@times, dLdF(:,i), dFidX2);
            end
            % Vectorize the weight gradient structure
            dLdW = self.vector_weights(dLdW);
            return
        end
        
        function [ result ] = check_grad(self, l_dim, r_dim, o_dim, grad_checks)
            % Check backprop computations for this SyBiLayer.
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
                self.dim_output = o_dim;
                self.dim_input(1) = l_dim;
                self.dim_input(2) = r_dim;
                X1 = randn(500, l_dim);
                X2 = randn(500, r_dim);
                C = cov(randn(100, o_dim));
                self.init_weights(0.1);
                W = self.vector_weights(self.weights);
                % Do check with respect to W
                fprintf('Checking wrt W\n');
                mf_func = @( w ) self.fake_loss_W(w, X1, X2, C);
                fastDerivativeCheck(mf_func,W(:),order,type);
                % Do check with respect to X1
                fprintf('Checking wrt X1\n');
                mf_func = @( x1 ) self.fake_loss_X1(W, x1, X2, C);
                fastDerivativeCheck(mf_func,X1(:),order,type);
                % Do check with respect to X2
                fprintf('Checking wrt X2\n');
                mf_func = @( x2 ) self.fake_loss_X2(W, X1, x2, C);
                fastDerivativeCheck(mf_func,X2(:),order,type);
            end
            result = 1;
            return
        end
        
        function [ L dLdW ] = fake_loss_W(self, W, X1, X2, C)
            % Fake loss wrapper for gradient testing.
            A = self.feedforward(X1, X2, W);
            L = sum(sum((A .* (A * C')), 2));
            dLdA = A*C' + A*C;
            dLdW = self.backprop(dLdA, A, X1, X2, W);
            return
        end
        
        function [ L dLdX1 ] = fake_loss_X1(self, W, X1, X2, C)
            % Fake loss wrapper for gradient testing.
            X1 = reshape(X1, size(X2,1), self.dim_input(1));
            A = self.feedforward(X1, X2, W);
            L = sum(sum((A .* (A * C')), 2));
            dLdA = A*C' + A*C;
            [dLdW dLdX1] = self.backprop(dLdA, A, X1, X2, W);
            dLdX1 = dLdX1(:);
            return
        end
        
        function [ L dLdX2 ] = fake_loss_X2(self, W, X1, X2, C)
            % Fake loss wrapper for gradient testing.
            X2 = reshape(X2, size(X1,1), self.dim_input(2));
            A = self.feedforward(X1, X2, W);
            L = sum(sum((A .* (A * C')), 2));
            dLdA = A*C' + A*C;
            [dLdW dLdX1 dLdX2] = self.backprop(dLdA, A, X1, X2, W);
            dLdX2 = dLdX2(:);
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
                F = max(X, 0);
                mask = F > 0.5;
                F(~mask) = F(~mask).^2;
                F(mask) = F(mask) - 0.25;
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
