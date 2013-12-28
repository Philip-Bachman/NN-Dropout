classdef BLNet < handle

    properties
        % layers is a cell array of handles/pointers for the layers from which
        % this BLNet is composed.
        layers
        % layer_count gives the number of layers in this BLNet (not including
        % the input layer).
        layer_count
        % out_loss gives the loss function to apply at output layer
        out_loss
        % lam_l2 gives an L2 regularization penalty applied to all weights in
        % this BLNet.
        lam_l2
        % lam_l1 gives an L1 regularization penalty applied to all weights in
        % this BLNet.
        lam_l1
        % lam_fd_ords gives weight of FD penalty of various orders
        lam_fd_ords
    end % END PROPERTIES
    
    methods
        function [ self ] = BLNet( layer_handles, loss_func )
            self.layer_count = length(layer_handles);
            self.layers = cell(1,self.layer_count);
            for i=1:self.layer_count,
                self.layers{i} = layer_handles{i};
            end
            if ~exist('loss_func','var')
                self.out_loss = @BLNet.loss_lsq;
            else
                self.out_loss = loss_func;
            end
            self.lam_l2 = 0;
            self.lam_l1 = 0;
            self.lam_fd_ords = [0 0];
            return
        end
        
        function [ N ] = weight_count(self)
            % Get the total number of weights in this layer.
            %
            N = 0;
            for i=1:self.layer_count,
                N = N + self.layers{i}.weight_count();
            end
            return
        end
        
        function [ Wc ] = init_weights(self, wt_scale, b_scale)
            % Initialize the weights for each layer in this BLNet. Return a
            % cell array containing the weight structs for each layer.
            if ~exist('b_scale','var')
                b_scale = wt_scale;
            end
            Wv = [];
            for i=1:self.layer_count,
                lay_i = self.layers{i};
                lay_i.init_weights(wt_scale, b_scale);
                Wv = [Wv; lay_i.vector_weights(lay_i.weights)];
            end
            Wc = self.cell_weights(Wv);
            return
        end
        
        function [ W ] = set_weights(self, W)
            % Set weights for this BLNet, using the struct/vector W.
            %
            if (length(W) ~= length(self.layers))
                W = self.cell_weights(W);
            end
            for i=1:length(self.layers),
                self.layers{i}.set_weights(W{i});
            end
            return
        end
        
        function [ Wv ] = vector_weights(self, Wc)
            % Return a vectorized representation of the cell array of weight
            % structures in Wc. Assume each weight structure Wc{i} can be used
            % by the layer object at self.layers{i}.
            %
            % If no argument given, use current weights for each layer.
            %
            Wv = [];
            for i=1:self.layer_count,
                lay_i = self.layers{i};
                if exist('Wc','var')
                    Wv = [Wv; lay_i.vector_weights(Wc{i})];
                else
                    Wv = [Wv; lay_i.vector_weights()];
                end
            end
            return
        end
        
        function [ Wc ] = cell_weights(self, Wv)
            % Return a cell array containing weight structures for each layer
            % in this BLNet, based on the joint weight vector Wv.
            %
            % If no argument given, use current weights for each layer.
            %
            Wc = cell(1,self.layer_count);
            idx_start = 1;
            for i=1:self.layer_count,
                lay_i = self.layers{i};
                idx_end = idx_start + (lay_i.weight_count() - 1);
                if exist('Wv','var')
                    Wc{i} = lay_i.struct_weights(Wv(idx_start:idx_end));
                else
                    Wc{i} = lay_i.struct_weights();
                end
                idx_start = idx_end + 1;
            end
            return
        end
        
        function [ A_post A_pre ] = feedforward(self, X1, X2, Wv)
            % Compute feedforward activations for the inputs in X1/X2 where
            % each row of X1 gives a "left" input and the corresponding row of
            % X2 gives it's partner "right" input. Return a cell array A, in
            % which A{i} gives the activations for self.layers{i}.
            %
            % If no Wv is given, then using current weights for each layer.
            %
            if ~exist('Wv','var')
                Wv = self.vector_weights();
            end
            A_post = cell(1,self.layer_count);
            if (nargout > 1)
                A_pre = cell(1,self.layer_count);
            end
            idx_start = 1;
            for i=1:self.layer_count,
                lay_i = self.layers{i};
                idx_end = idx_start + (lay_i.weight_count() - 1);
                Wi = Wv(idx_start:idx_end);
                if (i == 1)
                    [post pre] = lay_i.feedforward(X1, X2, Wi);
                    A_post{i} = post;
                    if (nargout > 1)
                        A_pre{i} = pre;
                    end
                else
                    [post pre] = lay_i.feedforward(A_post{i-1}, Wi);
                    A_post{i} = post;
                    if (nargout > 1)
                        A_pre{i} = pre;
                    end
                end
                idx_start = idx_end + 1;
            end
            return
        end
        
        function [ dLdW dLdX1 dLdX2 ] = backprop(self, dLdA, A, X1, X2, Wv)
            % Backprop through the layers of this BLNet, assuming that A gives
            % the cell array from self.feedforward(X1,X2,Wv) and dLdA gives the
            % per-activation gradients that we want to backprop. The length of
            % dLdA and A should be the same, and the dimensions of the matrices
            % in each of their cells should match.
            %
            dLdW = [];
            idx_start = (numel(Wv) + 1) - self.layers{end}.weight_count();
            for i=self.layer_count:-1:1,
                lay_i = self.layers{i};
                idx_end = (idx_start + lay_i.weight_count()) - 1;
                Wi = Wv(idx_start:idx_end);
                if (i == 1)
                    [dLdWi dLdX1 dLdX2] = lay_i.backprop(dLdA{i},A{i},X1,X2,Wi);
                else
                    [dLdWi dLdAi] = lay_i.backprop(dLdA{i},A{i},A{i-1},Wi);
                    dLdA{i-1} = dLdA{i-1} + dLdAi;
                end
                dLdW = [dLdWi; dLdW];
                if (i > 1)
                    idx_start = idx_start - self.layers{i-1}.weight_count();
                end
            end
            return
        end
        
        function [ opts ] = train(self, X1, X2, Y, i_iters, o_iters, opts)
            % Train a multilayer feedforward network with an initial bilinear
            % layer followed by some number of linear/quadratic layers.
            %
            % Parameters:
            %   X1: left training observations
            %   X2: right training observations
            %   Y: target outputs for observation pairs in X1/X2
            %   i_iters: number of LBFGS iterations per minibatch
            %   o_iters: number of minibatches to sample and train with
            %   opts: struct containing training options
            %
            % Outputs:
            %   opts: the options used in training (options not present in the
            %         initial opts structure will be set to default values).
            %
            if ~exist('opts','var')
                opts = struct();
            end
            % Check and set method specific options to valid values
            opts = BLNet.check_opts_lbfgs(opts);
            if isfield(opts,'lam_l2')
                self.lam_l2 = opts.lam_l2;
            end
            if isfield(opts,'lam_l1')
                self.lam_l1 = opts.lam_l1;
            end
            % Set options for minFunc to reasonable values
            mf_opts = struct();
            mf_opts.Display = 'iter';
            mf_opts.Method = 'lbfgs';
            mf_opts.Corr = 10;
            mf_opts.Damped = 1;
            mf_opts.LS_type = 0;
            mf_opts.LS_init = 3;
            mf_opts.use_mex = 0;
            mf_opts.MaxIter = i_iters;
            % Loop over LBFGS updates for randomly sampled minibatches
            batch_size = opts.batch_size;
            for i=1:o_iters,
                % Grab a batch of training samples
                if (batch_size < size(X1,1))
                    idx = randsample(size(X1,1),batch_size,false);
                    X1b = X1(idx,:);
                    X2b = X2(idx,:);
                    Yb = Y(idx,:);
                else
                    X1b = X1;
                    X2b = X2;
                    Yb = Y;
                end
                [X1c c_lens] = BLNet.sample_fd_chains(X1b, batch_size, 2, 0.1, 0.1);
                % Package a function handle for use by minFunc
                mf_func = @( w ) self.fake_loss_W(w, X1b, X1c, X2b, Yb, c_lens, batch_size);
                % Do some learning using minFunc
                W = self.vector_weights();
                W = minFunc(mf_func, W, mf_opts);
                % Record updated weights
                self.set_weights(W);
                % Beep-Boop!
                self.fake_loss_W(W, X1b, X1c, X2b, Yb, c_lens, batch_size, 1);
            end
            return
        end
        
        function [ result ] = check_grad(self, l_dim, r_dim, o_dim, grad_checks)
            % Check backprop computations for this BiliLayer.
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
                o_count = 500;
                self.out_loss = @BLNet.loss_hsq;
                self.lam_fd_ords = [10.0 10.0];
                self.layer_count = 2;
                self.layers{1} = BiliLayer([l_dim r_dim], o_dim, @BiliLayer.tanh_trans);
                self.layers{2} = LineLayer(o_dim, 1, @LineLayer.tanh_trans);
                self.init_weights(0.1);
                X1 = randn(o_count, l_dim);
                X2 = randn(o_count, r_dim);
                Y = sum(X1 .* X2, 2);
                [X1c c_lens] = BLNet.sample_fd_chains(X1, o_count, 2, 0.1, 0.1);
                W = self.vector_weights();
                X1 = X1(:);
                X2 = X2(:);
                % Do check with respect to W
                fprintf('Checking wrt W\n');
                mf_func = @( w ) self.fake_loss_W(w, X1, X1c, X2, Y, c_lens, o_count);
                fastDerivativeCheck(mf_func,W,order,type);
                % Do check with respect to X1
                fprintf('Checking wrt X1\n');
                mf_func = @( x1 ) self.fake_loss_X1(W, x1, X2, o_count);
                fastDerivativeCheck(mf_func,X1,order,type);
                % Do check with respect to X2
                fprintf('Checking wrt X2\n');
                mf_func = @( x2 ) self.fake_loss_X2(W, X1, x2, o_count);
                fastDerivativeCheck(mf_func,X2,order,type);
            end
            result = 1;
            return
        end
        
        function [ L dLdW ] = lbfgs_loss_W(self, W, X1, X2, Y)
            % Loss wrapper BLNet training wrt layer parameters (i.e. not wrt
            % input pairs X1/X2).
            %
            A = self.feedforward(X1, X2, W);
            % Compute loss and gradient at output layer
            Yh = A{end};
            [L dL] = self.out_loss(Yh, Y);
            dLdA = cell(1,length(A));
            for i=1:length(A),
                dLdA{i} = zeros(size(A{i}));
            end
            dLdA{end} = dL;
            dLdW = self.backprop(dLdA, A, X1, X2, W);
            % Add loss and gradient for L2/L1 parameter regularization
            L = L + (self.lam_l2 * sum(W.^2));
            dLdW = dLdW + (2 * (self.lam_l2 * W));
            return
        end
        
        function [ L dLdW ] = fd_loss_W(self, W, X1c, X2, c_lens)
            % Compute fd-based curvature loss based on fd-chains for the "left"
            % inputs X1.
            %
            fd_pts = length(X1c);
            % Compute activations for the FD chains.
            As = cell(1,self.layer_count);
            for i=1:self.layer_count,
                As{i} = [];
            end
            Ac = cell(1,fd_pts);
            for i=1:fd_pts,
                Ai = self.feedforward(X1c{i},X2,W);
                Ac{i} = Ai{end};
                for j=1:self.layer_count,
                    As{j} = [As{j}; Ai{j}];
                end
            end
            % Compute losses and gradients for FD gradient penalty
            [L_fd dL_fd] = BLNet.loss_fd(Ac, c_lens, self.lam_fd_ords);
            dLdAs = cell(1,self.layer_count);
            for i=1:self.layer_count,
                dLdAs{i} = zeros(size(As{i}));
            end
            dLdAs{end} = [];
            % Stack chain points and gradients for backproppin' and lockin'
            X1s = [];
            X2s = [];
            for i=1:fd_pts,
                X1s = [X1s; X1c{i}];
                X2s = [X2s; X2];
                dLdAs{end} = [dLdAs{end}; dL_fd{i}];
            end
            % Backprop gradients
            dLdW = self.backprop(dLdAs, As, X1s, X2s, W);
            % Merge losses for the different orders
            L = sum(L_fd);
            return
        end
        
        function [ L dLdW ] = fake_loss_W(self, W, X1, X1c, X2, Y, c_lens, o_count, do_print)
            if ~exist('do_print','var')
                do_print = 0;
            end
            % Fake loss wrapper for gradient testing.
            X1 = reshape(X1, o_count, self.layers{1}.dim_input(1));
            X2 = reshape(X2, o_count, self.layers{1}.dim_input(2));
            [L1 dLdW1] = self.lbfgs_loss_W(W, X1, X2, Y);
            [L2 dLdW2] = self.fd_loss_W(W, X1c, X2, c_lens);
            if (do_print == 1)
                fprintf('pred loss: %.4f, grad loss: %.4f\n',L1,L2);
            end
            L = L1 + L2;
            dLdW = dLdW1 + dLdW2;
            return
        end
        
        function [ L dLdX1 ] = fake_loss_X1(self, W, X1, X2, o_count)
            % Fake loss wrapper for gradient testing.
            X1 = reshape(X1, o_count, self.layers{1}.dim_input(1));
            X2 = reshape(X2, o_count, self.layers{1}.dim_input(2));
            A = self.feedforward(X1, X2, W);
            L = sum(sum(A{2}.^2));
            dLdA = cell(1,length(A));
            for i=1:length(A),
                dLdA{i} = zeros(size(A{i}));
            end
            dLdA{2} = 2*A{2};
            [dLdW dLdX1] = self.backprop(dLdA, A, X1, X2, W);
            dLdX1 = dLdX1(:);
            return
        end
        
        function [ L dLdX2 ] = fake_loss_X2(self, W, X1, X2, o_count)
            % Fake loss wrapper for gradient testing.
            X1 = reshape(X1, o_count, self.layers{1}.dim_input(1));
            X2 = reshape(X2, o_count, self.layers{1}.dim_input(2));
            A = self.feedforward(X1, X2, W);
            L = sum(sum(A{2}.^2));
            dLdA = cell(1,length(A));
            for i=1:length(A),
                dLdA{i} = zeros(size(A{i}));
            end
            dLdA{2} = 2*A{2};
            [dLdW dLdX1 dLdX2] = self.backprop(dLdA, A, X1, X2, W);
            dLdX2 = dLdX2(:);
            return
        end
        
    end % END INSTANCE METHODS
    
    methods (Static = true)
        
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
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Functional norm accessory functions %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [ L_fd dL_fd ] = loss_fd(A_out,fd_lens,ord_lams)
            % This computes gradients for curvature loss of multiple orders,
            % given the fd chain points in A_in, and their associated output
            % values in A_out. 
            %
            % length(A_out) should equal numel(ord_lams)+1
            %
            % A_out{i} should be give sampled outputs for the ith link in some
            % fd chain.
            %
            % ord_lams gives the weightings for curvature order penalties
            %
            % fd_lens gives the step size of the underlying fd chains
            %
            % L_fd{i} holds the vector of losses for the fd subchain of order i
            %
            % dL_fd{i} holds the matrix of gradients on the function outputs in
            % A_out{i}.
            %
            obs_count = size(A_out{1},1);
            L_fd = zeros(1,numel(ord_lams));
            dL_fd = cell(1,length(A_out));
            for i=1:length(A_out),
                dL_fd{i} = zeros(size(A_out{i}));
            end
            % Compute loss and gradient for each fd chain, for each order
            for i=1:numel(ord_lams),
                olam = ord_lams(i);
                fd_coeffs = zeros(1,(i+1));
                for j=0:i,
                    fd_coeffs(j+1) = (-1)^j * nchoosek(i,j);
                end
                fd_diffs = zeros(size(A_out{1}));
                for j=1:(i+1),
                    fd_diffs = fd_diffs + (fd_coeffs(j) * A_out{j});
                end
                fd_ls = fd_lens.^i; %ones(size(fd_lens.^i));
                L_fd(i) = sum(sum(bsxfun(@rdivide,fd_diffs.^2,fd_ls.^2)));
                L_fd(i) = (olam / obs_count) * L_fd(i);
                for j=1:(i+1),
                    dL_j = bsxfun(@rdivide,(fd_coeffs(j)*fd_diffs),(fd_ls.^2));
                    dL_fd{j} = dL_fd{j} + ((2 * (olam / obs_count)) * dL_j);
                end
            end
            return
        end
        
        function [L_fd dL_fd] = loss_fd_huber(A_out,fd_lens,ord_lams)
            % This computes gradients for curvature loss of multiple orders,
            % given the fd chain points in A_in, and their associated output
            % values in A_out. 
            %
            % General idea is same as for SmoothNet.loss_fd, except a Huberized
            % loss is applied to finite differences, rather than a pure squared
            % loss. This helps mitigate the strong "outlier" effect that occurs
            % for FDs of higher-order curvature, due to tiny denominators.
            %
            if ~exist('hub_thresh','var')
                hub_thresh = 2.0;
            end 
            obs_count = size(A_out{1},1);
            L_fd = zeros(1,numel(ord_lams));
            dL_fd = cell(1,length(A_out));
            for i=1:length(A_out),
                dL_fd{i} = zeros(size(A_out{i}));
            end
            % Compute loss and gradient for each fd chain, for each order
            for i=1:numel(ord_lams),
                olam = ord_lams(i);
                fd_coeffs = zeros(1,(i+1));
                for j=0:i,
                    fd_coeffs(j+1) = (-1)^j * nchoosek(i,j);
                end
                fd_diffs = zeros(size(A_out{1}));
                for j=1:(i+1),
                    fd_diffs = fd_diffs + (fd_coeffs(j) * A_out{j});
                end
                fd_ls = fd_lens.^i;
                fd_vals = bsxfun(@rdivide,fd_diffs,fd_ls);
                % Get masks for FD values subject to quadratic/linear losses
                quad_mask = bsxfun(@lt, abs(fd_vals), hub_thresh);
                line_mask = bsxfun(@ge, abs(fd_vals), hub_thresh);
                % Compute quadratic and linear parts of FD loss
                quad_loss = fd_vals.^2;
                quad_loss = quad_loss .* quad_mask;
                line_loss = (((2*hub_thresh) * abs(fd_vals)) - hub_thresh^2);
                line_loss = line_loss .* line_mask;
                L_fd(i) = sum(quad_loss(:)) + sum(line_loss(:));
                L_fd(i) = (olam / obs_count) * L;
                for j=1:(i+1),
                    dL_quad = 2*bsxfun(@rdivide,(fd_coeffs(j)*fd_vals),fd_ls);
                    dL_line = 2*hub_thresh*fd_coeffs(j)*sign(fd_vals);
                    dL_j = (dL_quad .* quad_mask) + (dL_line .* line_mask);
                    dL_fd{j} = dL_fd{j} + (olam * dL_j);
                end
            end
            return
        end
        
        function [ X_fd fd_lens ] = sample_fd_chains(X, chain_count,...
                max_order, fuzz_len, fd_len, bias)
            % Sample chains of points for forward FD estimates of directional 
            % higher-order derivatives. The anchor point for each sampled chain
            % is sampled from a "fuzzed" version of the empirical distribution
            % described by the points in X.
            %
            % Sample chain directions from a uniform distribution over the
            % surface of a hypersphere of dimension size(X,2). Then, rescale
            % these directions based on fd_len. If given, apply the "biasing"
            % transform (a matrix) to the directions prior to setting lengths.
            %
            if ~exist('bias','var')
                bias = eye(size(X,2));
            end
            if ~exist('strict_len','var')
                strict_len = 1;
            end
            chain_len = max_order + 1;
            % Sample points from the empirical distribution described by X,
            % convolved with the isotropic distribution with a length
            % distribution given by a Gaussian with std dev fuzz_len.
            s_idx = randsample(size(X,1),chain_count,true);
            Xs = X(s_idx,:);
            Xd = randn(size(Xs));
            Xd = bsxfun(@rdivide, Xd, max(sqrt(sum(Xd.^2,2)),1e-8));
            Xd = bsxfun(@times, Xd, (fuzz_len * abs(randn(size(Xd,1),1))));
            Xs = Xs + Xd;
            % Sample (biased & scaled) displacement directions for each chain.
            Xd = randn(size(Xs));
            Xd = Xd * bias;
            Xd = bsxfun(@rdivide, Xd, max(sqrt(sum(Xd.^2,2)),1e-8));
            if (strict_len ~= 1)
                % Sample fd lengths from a scaled abs(normal) distribution
                fd_lens = (fd_len/2) * abs(randn(size(Xd,1),1));
                fd_lens = fd_lens + fd_len;
            else
                % Set fd lengths strictly (i.e. to a fixed value)
                fd_lens = fd_len * ones(size(Xd,1),1);
            end
            % Scale sampled displacement directions using sampled fd lengths
            Xd = bsxfun(@times, Xd, fd_lens);
            % Construct forward chains by stepwise displacement
            X_fd = cell(1,chain_len);
            for i=0:(chain_len-1),
                X_fd{i+1} = Xs + (i * Xd);
            end
            return
        end
        
        function [ nn_len ] = compute_nn_len(X, sample_count)
            % Compute a length scale for curvature regularization. 
            % Sample observations at random and compute the distance to their
            % nearest neighbor. Use these nearest neighbor distances to compute
            % nn_len.
            %
            obs_count = size(X,1);
            dists = zeros(sample_count,1);
            for i=1:sample_count,
                idx = randi(obs_count);
                x1 = X(idx,:);
                dx = sqrt(sum(bsxfun(@minus,X,x1).^2,2));
                dx(idx) = max(dx) + 1;
                dists(i) = min(dx);
            end
            nn_len = median(dists) / 2;
            return
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % CLASS MATRIX MANIPULATIONS %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
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
            Yi = BLNet.class_inds(Yc);
            Yc = BLNet.class_cats(Yi);
            Yi = BLNet.class_inds(Yc);
            return
        end
        
        function [ Yc ] = to_cats( Yc )
            % This wraps class_cats and class_inds.
            Yi = BLNet.class_inds(Yc);
            Yc = BLNet.class_cats(Yi);
            return
        end 
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % PARAMETER CHECKING AND DEFAULT SETTING %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [ opts ] = check_opts_lbfgs( opts )
            % Process parameters to use in training of some sort.
            if ~isfield(opts, 'rounds')
                opts.rounds = 10000;
            end
            if ~isfield(opts, 'start_rate')
                opts.start_rate = 0.1;
            end
            if ~isfield(opts, 'decay_rate')
                opts.decay_rate = 0.999;
            end
            if ~isfield(opts, 'momentum')
                opts.momentum = 0.8;
            end
            if ~isfield(opts, 'batch_size')
                opts.batch_size = 100;
            end
            if ~isfield(opts, 'do_validate')
                opts.do_validate = 0;
            end
            if (opts.do_validate == 1)
                if (~isfield(opts, 'Xv') || ~isfield(opts, 'Yv'))
                    error('Validation set required for doing validation.');
                end
            end
            % Clip momentum to be in range [0...1]
            opts.momentum = min(1, max(0, opts.momentum));
            return
        end
        
        function [ opts ] = check_opts_sgd( opts )
            % Process parameters to use in training of some sort.
            if ~isfield(opts, 'rounds')
                opts.rounds = 10000;
            end
            if ~isfield(opts, 'start_rate')
                opts.start_rate = 0.1;
            end
            if ~isfield(opts, 'decay_rate')
                opts.decay_rate = 0.999;
            end
            if ~isfield(opts, 'momentum')
                opts.momentum = 0.8;
            end
            if ~isfield(opts, 'batch_size')
                opts.batch_size = 100;
            end
            if ~isfield(opts, 'do_validate')
                opts.do_validate = 0;
            end
            if (opts.do_validate == 1)
                if (~isfield(opts, 'Xv') || ~isfield(opts, 'Yv'))
                    error('Validation set required for doing validation.');
                end
            end
            % Clip momentum to be in range [0...1]
            opts.momentum = min(1, max(0, opts.momentum));
            return
        end
        
    end % END STATIC METHODS
    
        
end





%%%%%%%%%%%%%%
% EYE BUFFER %
%%%%%%%%%%%%%%
