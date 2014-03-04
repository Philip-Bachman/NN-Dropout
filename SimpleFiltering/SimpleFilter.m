classdef SimpleFilter < handle

    properties
        % bias_val determines the magnitude of the bias column appended to
        % observations prior to training and/or evaluation.
        bias_val
        % filter_method tells what feature learning algorithm to use
        filter_method
        % filters stores the filters (in rows) for this instance
        filters
        % pre_filters stores the pre-learning filters (for diagnostics)
        pre_filters
        % act_func gives the activation function/transform to apply to linear
        % filter responses during learning and evaluation
        act_func
        % obj_func gives the objective function to optimize
        obj_func
        % feedforward gives the feedforward associated with self.obj_func
        feedforward
        % The values of act_func, obj_func, and feedforward are determined by
        % parameters passed to SimpleFilter() at the time of object
        % instantiation.
        %
        % Possible values of act_func include SimpleFilter.actfun_soft_relu,
        % SimpleFilter.actfun_soft_abs, and SimpleFilter.actfun_rehu.
        %
        % Several filter learning methods are currently implemented:
        %   1: Sparse, dispersed filtering (my new method)
        %   2: Sparse filtering, following Ngiam et. al, NIPS 2011
        %   3: Reconstruction ICA, following Le et. al, NIPS 2011
        %   4: Sparse autoencoder
        %   5: K-sparse filtering, another new method
        %
    end % END PROPERTIES
    
    methods
        
        function [ self ] = SimpleFilter(filter_method, afun, b_val)
            if ~exist('filter_method','var')
                filter_method = 1;
            end
            % Record desired filtering method
            self.filter_method = filter_method;
            if ~exist('afun','var')
                % Choose a reasonable activation function if none wass given
                switch self.filter_method
                    case 1
                        afun = @SimpleFilter.actfun_rehu;
                    case 2
                        afun = @SimpleFilter.actfun_soft_abs;
                    case 3
                        afun = @SimpleFilter.actfun_soft_relu;
                    case 4
                        afun = @SimpleFilter.actfun_soft_relu;
                    case 5
                        afun = @SimpleFilter.actfun_soft_abs;
                    otherwise
                        error('Invalid self.filter_method.');
                end
            end
            if ~exist('b_val','var')
                b_val = 1;
            end
            % Set our activation function
            self.act_func = afun;
            % Set self.obj_func and self.feedforward based on filter_method
            switch self.filter_method
                case 1
                    self.obj_func = @SimpleFilter.of_spar_disp;
                    self.feedforward = @SimpleFilter.ff_spar_disp;
                case 2
                    self.obj_func = @SimpleFilter.of_sparse_filter;
                    self.feedforward = @SimpleFilter.ff_sparse_filter;
                case 3
                    self.obj_func = @SimpleFilter.of_rica;
                    self.feedforward = @SimpleFilter.ff_rica;
                case 4
                    self.obj_func = @SimpleFilter.of_spae;
                    self.feedforward = @SimpleFilter.ff_spae;
                case 5
                    self.obj_func = @SimpleFilter.of_kspar_filter;
                    self.feedforward = @SimpleFilter.ff_kspar_filter;
                otherwise
                    error('Invalid self.filter_method.');
            end
            % No usable filters just yet, be patient.
            self.bias_val = b_val;
            self.filters = [];
            self.pre_filters = [];
            return
        end
        
        function [ W ] = init_filters(self, X, N, do_kmeans, T_en)
            % Initialize filters. If do_kmeans == 1, then use (dot-product)
            % kmeans cluster centers learned from X as initial filters.
            % Otherwise, use random filters drawn from a zmuv Gaussian.
            if ~exist('do_kmeans','var')
                do_kmeans = 0;
            end
            if (~exist('T_en','var') || (numel(T_en) == 1))
                T_en = eye(size(X,2),size(X,2));
            end
            % Select initial filters, at random or via kmeans
            if (do_kmeans == 0)
                if (size(X,1) > 5000)
                    X = X(randsample(size(X,1),5000),:);
                end
                C = pinv(T_en);
                C = (C + C') ./ 2;
                W = mvnrnd(zeros(1,size(X,2)),C,N);
                if (self.bias_val > 1e-2)
                    W(:,end) = -0.01 * (1 / self.bias_val);
                else
                    W(:,end) = -0.01;
                end
                % Select +/- each initial filter to maximize its variance
                Fp = self.feedforward(X, W, self.act_func);
                Fn = self.feedforward(X, -W, self.act_func);
                for i=1:N,
                    if (var(Fp(:,i)) < var(Fn(:,i)))
                        W(i,:) = -W(i,:);
                    end
                end
            else
                W = kkmeans(X, N, 5, 10, 1);
                W = W + ((0.1*mean(std(X))) * randn(size(W)));
            end
            % Normalize filters to unit length
            W = SimpleFilter.norm_rows(W);
            return
        end
        
        function [ W ] = init_rand_filters(self, X, N, sparse_val, avg_act)
            % Initialize filters.
            X = SimpleFilter.add_bias(X, self.bias_val);
            W = randn(N, size(X,2));
            W(:,end) = 0.0;
            % Select +/- each initial filter to maximize its variance
            Fp = self.feedforward(X, W, self.act_func);
            Fn = self.feedforward(X, -W, self.act_func);
            for i=1:N,
                if (var(Fp(:,i)) < var(Fn(:,i)))
                    W(i,:) = -W(i,:);
                end
            end
            clear Fp Fn;
            % Normalize filters to unit norm
            W = SimpleFilter.norm_rows(W);
            % Scale filters to desired "variance"
            W = self.weight_scale(W, X, avg_act, 1);
            % Shift for sparsity
            F = max((X * W'), 0);
            F = sort(F,1,'descend');
            sparse_idx = ceil(sparse_val * size(F,1));
            if (sparse_idx < (size(F,1) - 1))
                sparse_shift = max(F(sparse_idx,:), 0);
                for i=1:N,
                    W(i,end) = W(i,end) - sparse_shift(i);
                end
            end
            self.filters = W;
            return
        end 
        
        function [ F ] = evaluate(self, X)
            % Why is this even here? Why is anything here?
            X = SimpleFilter.add_bias(X, self.bias_val);
            F = self.feedforward(X, self.filters, self.act_func);
            return
        end
        
        function [ Wp ] = weight_scale(self, W, X, avg_act, be_thorough)
            % Rescale the weights in W to have a given expected squared output
            if ~exist('be_thorough','var')
                be_thorough = 0;
            end
            if (be_thorough == 1)
                scale_space = logspace(-2,1,22);
            else
                scale_space = logspace(-1,1,7);
            end
            if (size(X,1) > 5000)
                X = X(randsample(size(X,1),5000),:);
            end
            sq_act = avg_act^2;
            opt_sqact = 1e10;
            opt_scale = 0;
            for s_num=1:numel(scale_space),
                s = scale_space(s_num);
                Ws = s * W;
                F = self.act_func(X, Ws);
                sqact = quantile(mean(F.^2,1),0.4);
                if (abs(sq_act - sqact) < opt_sqact)
                    opt_sqact = abs(sq_act - sqact);
                    opt_scale = s;
                end
            end
            Wp = opt_scale * W;
        end
        
        function [ W opts drop_opts ] = train(self, ...
                X, N, i_iters, o_iters, reset_filters, opts, drop_opts)
            % Train a collection of filters using one of the four simple
            % algorithms: Sparse Dispersed Filtering, Sparse Filtering,
            % Reconstruction ICA, and The Informatron.
            %
            % Parameters:
            %   X: data from which to learn the filters
            %   N: number of filters to learn
            %   i_iters: number of minFunc iterations for each batch
            %   o_iters: number of batches to train on
            %   reset_filters: whether or not to start from existing filters
            %   opts: struct containing method-specific options
            %   drop_opts: struct containing droppy ensemble var-reg options
            %
            % Outputs:
            %   W: the learned filters (also stored in self.filters)
            %   opts: opts used in training
            %   drop_opts: drop_opts used in training
            %
            if ~exist('opts','var')
                opts = struct();
            end
            if ~exist('drop_opts','var')
                drop_opts = struct();
            end
            % Add a bias to X
            X = SimpleFilter.add_bias(X, self.bias_val);
            % Check and set method specific options to valid values
            opts = SimpleFilter.check_opts(opts, self.filter_method);
            drop_opts = SimpleFilter.check_drop_opts(drop_opts);
            % Initialize filters as desired (or if previously uninitialized)
            if ((reset_filters == 1) || (numel(self.filters) == 0))
                W = self.init_filters(X, N, opts.init_kmeans);
                if ((self.filter_method == 1) || (self.filter_method == 5))
                    % For methods with dispersion objectives, rescale W
                    W = self.weight_scale(W, X, opts.avg_act, 1);
                else
                    % For methods without dispersion objectives, normalize W
                    W = SimpleFilter.norm_rows(W);
                end
                self.filters = W;
                self.pre_filters = W;
            else
                W = self.filters;
            end
            % Check that dimensions of self.filters are copasetic
            if ((size(self.filters,1) ~= N) || ...
                    (size(self.filters,2) ~= size(X,2)) || ...
                    (sum(abs(size(self.filters)-size(self.pre_filters))) > 0))
                error('filter sizes out of whack.');
            end
            % Set options for minFunc to reasonable values
            mf_opts = struct();
            mf_opts.Display = 'iter';
            mf_opts.Method = 'sd';
            mf_opts.optTol = 1e-8;
            mf_opts.Corr = 10;
            mf_opts.Damped = 1;
            mf_opts.LS_type = 0;
            mf_opts.LS_init = 3;
            mf_opts.use_mex = 0;
            mf_opts.MaxIter = i_iters;
            % Get some per-batch sizing options from drop_opts
            batch_obs = drop_opts.batch_obs;
            batch_reps = drop_opts.batch_reps;
            pause on;
            for i=1:o_iters,
                % Grab a batch of training samples
                if (opts.batch_size < size(X,1))
                    Xo = X(randsample(size(X,1),opts.batch_size,false),:);
                else
                    Xo = X;
                end
                Xs = X(randsample(size(X,1),batch_obs,true),:);
                Xs = repmat(Xs,batch_reps,1);
                % Generate a "drop mask" for this batch's observations
                drop_mask = (rand(size(Xs)) > drop_opts.drop_rate);
                % Keep bias, never drop it.
                drop_mask(:,end) = 1;
                % Setup the objective function for this batch
                mf_func = @( w ) self.drop_loss_wrapper(w,drop_opts.lam_drop,...
                    Xo, Xs, N, drop_mask, batch_obs, batch_reps, opts);
                % Do some learning using minFunc
                W = minFunc(mf_func, W(:), mf_opts);
                % Record the result of partial optimization
                self.filters = reshape(W, N, size(X,2));
                %C = ZMUV(self.filters(:,1:(end-1))')';
                %show_centroids(C, 6, 6);
                %pause(2.0);
            end
            % Record optimized weights, for returnage
            W = reshape(W, N, size(X,2));
            return
        end
        
        function [ diffs ] = check_grad(self, ...
                X, N, pre_iter, grad_iter, opts, drop_opts)
            % Train a collection of filters using one of the four simple
            % algorithms: Sparse Dispersed Filtering, Sparse Filtering,
            % Reconstruction ICA, and The Informatron.
            %
            % Parameters:
            %   X: data from which to learn the filters
            %   N: number of filters to learn
            %   pre_iter: number of minFunc iterations to perform to "warm up"
            %             the initial filters.
            %   grad_iter: number of "fast grad checks" to perform.
            %   opts: struct containing method-specific options
            %   drop_opts: struct containing droppy ensemble var-reg options
            %
            % Outputs:
            %   W: the learned filters
            %
            if ~exist('opts','var')
                opts = struct();
            end
            if ~exist('drop_opts','var')
                drop_opts = struct();
            end
            % Add a bias to X
            X = SimpleFilter.add_bias(X, self.bias_val);
            % Check and set method specific options to valid values
            opts = SimpleFilter.check_opts(opts, self.filter_method);
            drop_opts = SimpleFilter.check_drop_opts(drop_opts);
            % Initialize filters
            W = self.init_filters(X, N, 0);
            W = self.weight_scale(W, X, opts.avg_act, 1);
            % Set options for minFunc to reasonable values
            mf_opts = struct();
            mf_opts.MaxIter = pre_iter;
            mf_opts.Display = 'iter';
            mf_opts.Method = 'lbfgs';
            mf_opts.Corr = 5;
            mf_opts.Damped = 1;
            mf_opts.LS = 0;
            mf_opts.LS_type = 0;
            mf_opts.LS_init = 3;
            mf_opts.use_mex = 0;
            % Grab a batch of training samples
            if (opts.batch_size < size(X,1))
                Xo = X(randsample(size(X,1),opts.batch_size,false),:);
            else
                Xo = X;
            end
            Xs = X(randsample(size(X,1),drop_opts.batch_obs,true),:);
            Xs = repmat(Xs,drop_opts.batch_reps,1);
            % Generate a "drop mask" for this batch's observations.
            drop_mask = (rand(size(Xs)) > drop_opts.drop_rate);
            % Keep bias, never drop it.
            drop_mask(:,end) = 1;
            % Setup the objective function for this batch
            batch_obs = drop_opts.batch_obs;
            batch_reps = drop_opts.batch_reps;
            mf_func = @( w ) self.drop_loss_wrapper(w, drop_opts.lam_drop, ...
                Xo, Xs, N, drop_mask, batch_obs, batch_reps, opts);
            %mf_func = @( w ) self.obj_func(w, X, N, self.act_func, opts);
            % Do some learning using minFunc
            W = minFunc(mf_func, W(:), mf_opts);
            % Check gradients for current W
            order = 1;
            type = 2;
            diffs = zeros(1,grad_iter);
            for i=1:grad_iter,
                fprintf('=============================================\n');
                fprintf('GRAD CHECK %d\n',i);
                fastDerivativeCheck(mf_func,W,order,type);
            end
            return
        end
        
        function [ L dLdW ] = drop_loss_wrapper(self, W, lam_drop, ...
                Xo, Xd, N, drop_mask, batch_obs, batch_reps, opts)
            % Reshape W into matrix form
            W = reshape(W,N,size(Xo,2));
            % Compute main objective part of loss, and its gradient
            [Lo dLodW] = ...
                self.obj_func(W(:), Xo, N, self.act_func, opts);
            clear('Xo');
            if (batch_reps > 1)
                % Compute feature activations from droppy observations, and
                % grab a function handle for backpropping through activation
                Xd = (Xd .* drop_mask);
                clear('drop_mask');
                [F bp_dFdW] = self.act_func(Xd, W);
                [F bp_dFdF] = SimpleFilter.norm_rows(F);
                [Ld dLddF] = SimpleFilter.drop_loss(F, batch_obs, batch_reps);
                dLddW = bp_dFdW(bp_dFdF(dLddF), Xd);
            else
                Ld = 0;
                dLddW = zeros(size(W));
            end
            % Combine losses and gradients
            fprintf('      Lo: %.6f, Ld: %.6f\n', Lo, (lam_drop * Ld));
            L = 10 * (Lo + (lam_drop * Ld));
            dLdW = 10 * (dLodW + (lam_drop * dLddW(:)));
            return
        end
        

    end % END INSTANCE METHODS
    
    methods (Static = true)
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % SPARSE, DISPERSED FILTERING FUNCTIONS %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [ L dW ] = of_spar_disp(W, X, N, afun, opts)
            % Reshape W into matrix form
            W = reshape(W, N, size(X,2));
            if (nargout == 1)
                L = SimpleFilter.bp_spar_disp(X, W, afun, opts);
            else
                [L dW] = SimpleFilter.bp_spar_disp(X, W, afun, opts);
                dW = dW(:);
            end
            return
        end

        function [ F bp_dFdW ] = ff_spar_disp(X, W, afun)
            [F bp_dFdW] = afun(X, W);
            return
        end
        
        function [ L dLdW ] = bp_spar_disp(X, W, afun, opts)
            o_count = size(X,1);
            f_count = size(W,1);
            lam_Sr = opts.lam_S / o_count;
            lam_Sc = opts.lam_S / (10*f_count);
            lam_Dr = opts.lam_D / o_count;
            lam_Dc = opts.lam_D / f_count;
            lam_E = opts.en_opts.lam;
            lam_B = 1e-4;
            avg_act = opts.avg_act;
            %%%%%%%%%%%%%%%%%%%%%
            % Loss computations %
            %%%%%%%%%%%%%%%%%%%%%
            % Feedforward
            [F1 bp_dF1dW] = SimpleFilter.ff_spar_disp(X, W, afun);
            % Compute row-wise sparsity
            [Fr bp_Fr] = SimpleFilter.norm_rows(F1);
            Sr = sum(sum(Fr));
            % Compute column-wise sparsity
            [Fc bp_Fc] = SimpleFilter.norm_cols(F1);
            Sc = sum(sum(Fc));
            % Compute row-wise dispersion penalty
            Dr = (sum(F1.^2,2) / f_count) - avg_act^2;
            LDr = sum(Dr.^2);
            % Compute column-wise dispersion penalty
            Dc = (sum(F1.^2,1) / o_count) - avg_act^2;
            LDc = sum(Dc.^2);
            % Compute Tikhonovy Elastic Net loss/gradient on weights
            [Le dEdW] = SimpleFilter.reg_elnet(W,opts.en_opts);
            % Compute babel function loss/gradient on weights
            [Lb dBdW] = SimpleFilter.babel_loss(W,1);
            % Combine losses, like communists
            L = lam_Sr*Sr + lam_Sc*Sc + lam_Dr*LDr + lam_Dc*LDc + ...
                lam_E*Le + lam_B*Lb;
            %%%%%%%%%%%%%%%%%%%%%%%%%
            % Gradient computations %
            %%%%%%%%%%%%%%%%%%%%%%%%%
            % Compute gradient due to sparsity penalty
            dSdF1 = bp_Fr(lam_Sr*ones(size(Fr))) + bp_Fc(lam_Sc*ones(size(Fc)));
            % Compute gradient due to dispersion penalty
            dDdF1 = (lam_Dr * ((4/f_count) * bsxfun(@times,F1,Dr))) + ...
                (lam_Dc * ((4/o_count) * bsxfun(@times,F1,Dc)));
            % Combine gradients
            dLdW = bp_dF1dW((dSdF1 + dDdF1), X) + lam_E*dEdW + lam_B*dBdW;
            %%%%%%%%%%%%%%%
            % Diagnostics %
            %%%%%%%%%%%%%%%
            % Textual
            fprintf('          Sr: %.6f, Sc: %.6f, Dr: %.6f, Dc: %.6f, Le: %.6f, Lb: %.6f\n',...
                lam_Sr*Sr, lam_Sc*Sc, lam_Dr*LDr, lam_Dc*LDc, lam_E*Le, lam_B*Lb);
            % Visual
            subplot(6,4,[1 2 3 4 5 6 7 8]);
            cla;
            hold on;
            plot(sum(F1.^2,1)./o_count,'o');
            plot(1:f_count,(avg_act^2 * ones(1,f_count)),'r--');
            ylim([0 2*(avg_act^2)]);
            xlim([0 f_count+1]);
            Wi = W(:,1:(end-1));
            dim = sqrt(size(Wi,2));
            if (abs(round(dim) - dim) > 1e-10)
                sq_val = (floor(dim) + 1)^2;
                Wi = [Wi zeros(size(Wi,1),(sq_val-size(Wi,2)))];
            end
            dim = round(sqrt(size(Wi,2)));
            for j=1:16,
                subplot(6,4,8+j);
                cla;
                imagesc(reshape(Wi(j,:),dim,dim)');
                set(gca,'xtick',[],'ytick',[]);
                axis square;
                colormap('gray');
            end
            drawnow();
            return
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % K-SPARSE FILTERING FUNCTIONS %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [ L dW ] = of_kspar_filter(W, X, N, afun, opts)
            % Reshape W into matrix form
            W = reshape(W, N, size(X,2));
            if (nargout == 1)
                L = SimpleFilter.bp_kspar_filter(X, W, afun, opts);
            else
                [L dW] = SimpleFilter.bp_kspar_filter(X, W, afun, opts);
                dW = dW(:);
            end
            return
        end
        
        function [ F bp_dFdW ] = ff_kspar_filter(X, W, afun)
            [F bp_dFdW] = afun(X, W);
            return
        end
        
        function [ L dLdW ] = bp_kspar_filter(X, W, afun, opts)
            o_count = size(X,1);
            f_count = size(W,1);
            lam_Sr = opts.lam_S / o_count;
            lam_Sc = opts.lam_S / (10*f_count);
            lam_D = opts.lam_D / o_count;
            k_spar = round(f_count / 10);
            lam_E = opts.en_opts.lam;
            lam_B = 1e-4;
            avg_act = opts.avg_act;
            %%%%%%%%%%%%%%%%%%%%%
            % Loss computations %
            %%%%%%%%%%%%%%%%%%%%%
            % Feedforward
            [F1 bp_dF1dW] = SimpleFilter.ff_kspar_filter(X, W, afun);
            % Compute row-wise sparsity
            [Fr bp_Fr] = SimpleFilter.norm_rows(F1);
            Sr = sum(sum(Fr));
            % Compute column-wise sparsity
            [Fc bp_Fc] = SimpleFilter.norm_cols(F1);
            Sc = sum(sum(Fc));
            % Compute row-wise k-sparsity penalty
            Fs = sort(F1,2,'descend');
            Fs_mask = double(bsxfun(@ge, F1, Fs(:,k_spar)));
            Fs = sort(F1,1,'descend');
            Fs_mask = Fs_mask + ...
                double(bsxfun(@ge, F1, Fs(10,:)));
            D = -sum(sum((F1 .* Fs_mask)));
            % Compute Tikhonovy Elastic Net loss/gradient on weights
            [Le dEdW] = SimpleFilter.reg_elnet(W, opts.en_opts);
            % Compute babel function loss/gradient on weights
            [Lb dBdW] = SimpleFilter.babel_loss(W, 1);
            % Combine losses, like communists
            L = lam_Sr*Sr + lam_Sc*Sc + lam_D*D + lam_E*Le + lam_B*Lb;
            %%%%%%%%%%%%%%%%%%%%%%%%%
            % Gradient computations %
            %%%%%%%%%%%%%%%%%%%%%%%%%
            % Compute gradient due to sparsity penalty
            dSdF1 = bp_Fr(lam_Sr*ones(size(Fr))) + bp_Fc(lam_Sc*ones(size(Fc)));
            % Compute gradient due to row-wise k-sparsity penalty
            dDdF1 = -(lam_D * Fs_mask);
            % Combine gradients
            dLdW = bp_dF1dW((dSdF1 + dDdF1), X) + lam_E*dEdW + lam_B*dBdW;
            %%%%%%%%%%%%%%%
            % Diagnostics %
            %%%%%%%%%%%%%%%
            % Textual
            fprintf('          Sr: %.6f, Sc: %.6f, D: %.6f, Le: %.6f, Lb: %.6f\n',...
                lam_Sr*Sr, lam_Sc*Sc, lam_D*D, lam_E*Le, lam_B*Lb);
            % Visual
            subplot(6,4,[1 2 3 4 5 6 7 8]);
            cla;
            hold on;
            plot(sum(F1.^2,1)./o_count,'o');
            plot(1:f_count,(avg_act^2 * ones(1,f_count)),'r--');
            ylim([0 2*(avg_act^2)]);
            xlim([0 f_count+1]);
            Wi = W(:,1:(end-1));
            dim = sqrt(size(Wi,2));
            if (abs(round(dim) - dim) > 1e-10)
                sq_val = (floor(dim) + 1)^2;
                Wi = [Wi zeros(size(Wi,1),(sq_val-size(Wi,2)))];
            end
            dim = round(sqrt(size(Wi,2)));
            for j=1:16,
                subplot(6,4,8+j);
                cla;
                imagesc(reshape(Wi(j,:),dim,dim)');
                set(gca,'xtick',[],'ytick',[]);
                axis square;
                colormap('gray');
            end
            drawnow();
            return
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % SPARSE FILTERING FUNCTIONS %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [ L dLdW ] = of_sparse_filter(W, X, N, afun, opts)
            % Reshape W into matrix form
            W = reshape(W, N, size(X,2));
            % Compute objective function and gradient
            [L dLdW] = SimpleFilter.bp_sparse_filter(X, W, afun, opts);
            dLdW = dLdW(:);
            return
        end

        function [ F ] = ff_sparse_filter(X, W, afun)
            % Feedforward
            F = afun(X, W);
            % Normalize columns of F
            F = SimpleFilter.norm_cols(F);
            % Normalize rows of F
            F = SimpleFilter.norm_rows(F);
            return
        end

        function [ L dLdW ] = bp_sparse_filter(X, W, afun, opts)
            lam_E = opts.en_opts.lam;
            % Feedforward
            [F bp_dFdW] = afun(X, W);
            % Normalize columns of F
            [F BPc] = SimpleFilter.norm_cols(F);
            % Normalize rows of F
            [F BPr] = SimpleFilter.norm_rows(F);
            % Compute sparsity part of loss
            Ls = sum(sum(sqrt(F.^2 + 1e-6)));
            % Compute Tikhonovy Elastic Net loss/gradient on weights
            [Le dEdW] = SimpleFilter.reg_elnet(W,opts.en_opts);
            % Combine losses
            L = Ls + (lam_E * Le);
            % Compute gradient of sparsity penalty
            dSdF3 = F ./ sqrt(F.^2 + 1e-6);
            % Backpropagate sparsity gradient through normalization
            dLdW = bp_dFdW(BPc(BPr(dSdF3)), X) + (lam_E * dEdW);
            return
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % RECONSTRUCTION ICA FUNCTIONS %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [ L dLdW ] = of_rica(W, X, N, afun, opts)
            % Reshape W into matrix form
            W = reshape(W, N, size(X,2));
            % Compute objective function and gradient
            [L dLdW] = SimpleFilter.bp_rica(X, W, afun, opts);
            dLdW = dLdW(:);
            return
        end

        function [ F ] = ff_rica(X, W, afun)
            % Feedforward
            W = SimpleFilter.norm_rows(W);
            F = X * W';
            return
        end

        function [ L dLdW ] = bp_rica(X, W, afun, opts)
            lam_S = opts.lam_S / size(X,1);
            lam_E = opts.en_opts.lam;
            obs_count = size(X,1);
            % Feedforward
            [W bp_W] = SimpleFilter.norm_rows(W);
            Xw = W*X';
            % Compute reconstruction residual and loss
            WtWx_x = W'*Xw - X';
            % WtWx_x = (W' * (W * X')) - X'
            Lr = sum(sum(WtWx_x.^2)) / obs_count;
            % Compute sparsity loss
            [Xn bp_Xn] = SimpleFilter.norm_cols(Xw);
            Ls = sum(sum(sqrt(Xn.^2 + 1e-6)));
            % Compute L1/L2 penalty/gradient on filter weights
            [Le dEdW] = SimpleFilter.reg_elnet(W,opts.en_opts);
            % Compute joint reconstruction and sparsity loss
            L = Lr + (lam_S * Ls) + (lam_E * Le);
            % Compute gradients
            fprintf('          Lr: %.6f, Ls: %.6f, Le: %.6f\n', ...
                Lr, lam_S*Ls, lam_E*Le);
            % Compute reconstruction part of gradient
            dRdW = (2 * W * (WtWx_x * X)) + (2 * W * X' * WtWx_x');
            % Compute regularization part of gradient
            dSdXn = Xn ./ sqrt(Xn.^2 + 1e-6);
            dSdW = bp_Xn(dSdXn) * X;
            % Normalize gradient for number of observations
            dLdW = ((dRdW ./ obs_count) + (lam_S * dSdW)) + (lam_E * dEdW);
            dLdW = bp_W(dLdW);
            return
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % SPARSE AUTOENCODER FUNCTIONS %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [ L dLdW ] = of_spae(W, X, N, afun, opts)
            % Reshape W into matrix form
            W = reshape(W, N, size(X,2));
            % Compute objective function and gradient
            [L dLdW] = SimpleFilter.bp_spae(X, W, afun, opts);
            dLdW = dLdW(:);
            return
        end

        function [ F bp_dFdW ] = ff_spae(X, W, afun)
            % Feedforward
            [F bp_dFdW] = afun(X, W);
            return
        end

        function [ L dLdW ] = bp_spae(X, W, afun, opts)
            o_count = size(X,1);
            lam_S = opts.lam_S / o_count;
            lam_E = opts.en_opts.lam;
            % Feedforward
            [F bp_dFdW] = SimpleFilter.ff_spae(X, W, afun);
            % Compute reconstruction residual and loss
            R = (F * W) - X;
            Lr = sum(sum(R.^2)) / o_count;
            % Compute sparsity loss
            [Fn bp_Fn] = SimpleFilter.norm_rows(F);
            Ls = sum(sum(sqrt(Fn.^2 + 1e-8)));
            % Compute L1/L2 penalty/gradient on filter weights
            [Le dEdW] = SimpleFilter.reg_elnet(W,opts.en_opts);
            % Compute joint loss
            L = Lr + lam_S*Ls + lam_E*Le;
            % Compute reconstruction part of gradient
            dRdW = (2 * F' * R) ./ o_count;
            dRdF = (2 * R * W') ./ o_count;
            % Compute sparsity of gradient
            dSdF = lam_S * Fn ./ sqrt(Fn.^2 + 1e-8);
            % Normalize gradient for number of observations
            dLdW = bp_dFdW((bp_Fn(dSdF) + dRdF), X) + dRdW + lam_E*dEdW;
            %%%%%%%%%%%%%%%
            % DIAGNOSTICS %
            %%%%%%%%%%%%%%%
            fprintf('          Lr: %.6f, Ls: %.6f, Le: %.6f\n', ...
                Lr, lam_S*Ls, lam_E*Le);
            Wi = W(:,1:(end-1));
            dim = sqrt(size(Wi,2));
            if (abs(round(dim) - dim) > 1e-10)
                sq_val = (floor(dim) + 1)^2;
                Wi = [Wi zeros(size(Wi,1),(sq_val-size(Wi,2)))];
            end
            dim = round(sqrt(size(Wi,2)));
            for j=1:25,
                subplot(5,5,j);
                cla;
                imagesc(reshape(Wi(j,:),dim,dim)');
                set(gca,'xtick',[],'ytick',[]);
                axis square;
                colormap('gray');
            end
            drawnow();
            return
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % DROPOUT ENSEMBLE VARIANCE LOSS/GRAD %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [L dLdF] = drop_loss(F, b_obs, b_reps, dev_type)
            % Compute feature activations from droppy observations, and
            % grab a function handle for backpropping through activation
            %
            if ~exist('dev_type','var')
                dev_type = 1;
            end
            switch dev_type
                case 1
                    [F bp_F] = SimpleFilter.norm_rows(F);
                case 2
                    [F bp_F] = SimpleFilter.tanh_transform(F);
                case 3
                    [F bp_F] = SimpleFilter.dont_transform(F);
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
        
        %%%%%%%%%%%%%%%%%%%%
        % HELPER FUNCTIONS %
        %%%%%%%%%%%%%%%%%%%%
        
        function [ L dLdW ] = reg_elnet( W, en_opts )
            % Soft-absolute value penalty and gradient for weights matrix W.
            %
            alpha = en_opts.alpha;
            T = en_opts.T;
            if (size(T,1) == (size(W,2)-1))
                T = [T zeros(size(T,1),1)];
                T = [T; zeros(1,size(T,2))];
                T(end,end) = mean(diag(T));
            end
            % Compute Tikhonov part of loss
            Lt = sum(sum(bsxfun(@times,W,(T * W')')));
            % Compute Lasso part of loss
            Ll = sum(sum(sqrt(W.^2 + 1e-6)));
            % Combine losses
            L = (alpha * Lt) + ((1 - alpha) * Ll);
            % Compute Tikhonov part of gradient
            dLtdW = 2 * (W * T);
            % Compute Lasso part of gradient
            dLldW = W ./ sqrt(W.^2 + 1e-6);
            % Combine gradients
            dLdW = (alpha * dLtdW) + ((1 - alpha) * dLldW);
            return
        end
        
        function [ F BP ] = norm_rows(X)
            % L2 normalize X by rows, and return both the row-normalized matrix
            % and a function handle for backpropagating through normalization.
            N = sqrt(sum(X.^2,2) + 1e-6);
            F = bsxfun(@rdivide,X,N);
            % Backpropagate through normalization for unit norm
            BP = @( D ) ...
                (bsxfun(@rdivide,D,N) - bsxfun(@times,F,(sum(D.*X,2)./(N.^2))));
            return
        end
        
        function [ F BP ] = norm_cols(X)
            % L2 normalize X by columns, and return the row-normalized matrix
            % and a function handle for backpropagating through normalization.
            N = sqrt(sum(X.^2,1) + 1e-6);
            F = bsxfun(@rdivide,X,N);
            % Backpropagate through normalization for unit norm
            BP = @( D ) ...
                (bsxfun(@rdivide,D,N) - bsxfun(@times,F,(sum(D.*X,1)./(N.^2))));
            return
        end
        
        function [ F BP ] = prob_rows(X)
            % Normalize rows of X. which we assume to be non-negative, to
            % probability distribution form. Return the normalized observations
            % and a function for backpropagating through the normalization.
            N = sum(X,2);
            F = bsxfun(@rdivide, X, N);
            BP = @( D ) bsxfun(@minus,bsxfun(@rdivide,D,N),(sum(D.*X,2)./(N.^2)));
            return
        end
        
        function [ F BP ] = prob_cols(X)
            % Normalize rows of X. which we assume to be non-negative, to
            % probability distribution form. Return the normalized observations
            % and a function for backpropagating through the normalization.
            N = sum(X,1);
            F = bsxfun(@rdivide, X, N);
            BP = @( D ) bsxfun(@minus,bsxfun(@rdivide,D,N),(sum(D.*X,1)./(N.^2)));
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
        
        function [ T ] = cool_regmat(X, lam)
            % Generate a normalized covariance matrix for X
            %
            C = (X'*X) ./ size(X,1);
            C = C + ((lam * mean(diag(C))) * eye(size(C)));
            T = pinv(C);
            [ evals ] = eig(T);
            T = T ./ mean(evals);
            T = (T + T') ./ 2;
            return
        end
        
        function [ fl ] = flog(X)
            % Fake log, for use in entropy, defines log(~0) to be 0. Will freak
            % out for negative inputs, as is appropriate.
            fl = log(X);
            fl(X < 1e-10) = 0;
            return
        end
        
        function [ L dLdW ] = reg_max_norm(W, max_norm, lam)
            % Penalize squared L2 norm weights for being larger than max_norm.
            % Penalty strength is determined by lam.
            margin = max(0, (sum(W.^2,2) - max_norm^2));
            margin_viol = margin > 0;
            L = lam * sum(margin);
            if (nargout > 1)
                dLdW = 2*lam * bsxfun(@times, W, margin_viol);
            end
            return
        end
        
        function [ Xb ] = add_bias(X, bias_val)
            % Add a bias column to X.
            if ~exist('bias_val','var')
                bias_val = 1;
            end
            Xb = [X (bias_val * ones(size(X,1),1))];
            return
        end
        
        function [ L dLdW ] = babel_loss( W, has_bias )
            % Compute a loss to disourage redundancy in the rows of W, by
            % penalizing the cosine similarity between rows with positive
            % pair-wise dot-products.
            %
            if ~exist('has_bias','var')
                has_bias = 0;
            end
            if (has_bias == 1)
                W = W(:,1:(end-1));
            end
            [Wn bp_dLdWn] = SimpleFilter.norm_rows(W);
            % Compute pair-wise dot-products between normalized rows
            C = Wn * Wn';
            % Keep only the non-negative, off-diagonal pw-dps
            mask = (C > 0);
            for i=1:size(Wn,1),
                mask(i,i) = 0;
            end
            C = C .* mask;
            % Compute the loss
            L = sum(C(:)) / size(Wn,1);
            % Compute gradient for loss
            dLdWn = zeros(size(Wn));
            for i=1:size(Wn,1),
                dLdWn(i,:) = sum(Wn(mask(i,:),:),1);
            end
            dLdW = bp_dLdWn(dLdWn) ./ size(Wn,1);
            if (has_bias == 1)
                dLdW = [dLdW zeros(size(W,1),1)];
            end
            return
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%
        % ACTIVATION FUNCTIONS %
        %%%%%%%%%%%%%%%%%%%%%%%%
        
        function [ A BP_W BP_X ] = actfun_tanh(X, W)
            % Hyperbolic tangent activation function
            F = X*W';
            A = tanh(F);
            if (nargout > 1)
                dAdF = 1 - (A.^2);
                BP_W = @( dLdA, X ) ((dLdA .* dAdF)' * X);
                BP_X = @( dLdA, W ) ((dLdA .* dAdF) * W);
            end
            return
        end
        
        function [ A BP_W BP_X ] = actfun_sigmoid(X, W)
            % Hyperbolic tangent activation function
            F = X*W';
            A = 1 ./ (1 + exp(-F));
            if (nargout > 1)
                dAdF = exp(-F) ./ ((1 + exp(-F)).^2);
                BP_W = @( dLdA, X ) ((dLdA .* dAdF)' * X);
                BP_X = @( dLdA, W ) ((dLdA .* dAdF) * W);
            end
            return
        end
        
        function [ A BP_W BP_X ] = actfun_soft_relu(X, W)
            % Rectified soft-absolute activation function
            EPS = 1e-3;
            SR_EPS = sqrt(EPS);
            F = X*W';
            pos_mask = F > 0;
            A = (sqrt(F.^2 + EPS) - SR_EPS) .* pos_mask;
            if (nargout > 1)
                dAdF = (F ./ sqrt(F.^2 + EPS)) .* pos_mask;
                BP_W = @( dLdA, X ) ((dLdA .* dAdF)' * X);
                BP_X = @( dLdA, W ) ((dLdA .* dAdF) * W);
            end
            return
        end

        function [ A BP_W BP_X ] = actfun_rehu(X, W)
            % Huberized rectified linear activation function
            delta = 0.5;
            A_pre = X * W';
            A_pre = bsxfun(@max, A_pre, 0);
            mask = bsxfun(@lt, A_pre, delta);
            A = zeros(size(A_pre));
            A(mask) = A_pre(mask).^2;
            A(~mask) = (2 * delta * A_pre(~mask)) - delta^2;
            if (nargout > 1)
                dAdF = zeros(size(A_pre));
                dAdF(mask) = 2 * A_pre(mask);
                dAdF(~mask) = 2 * delta;
                BP_W = @( dLdA, X ) ((dLdA .* dAdF)' * X);
                BP_X = @( dLdA, W ) ((dLdA .* dAdF) * W);
            end
            return
        end
        
        function [ A BP_W BP_X ] = actfun_rechu(X, W)
            % Chuberized rectified linear activation function
            A_pre = (X * W');
            A_pre = bsxfun(@max, A_pre, 0);
            cub_mask = (A_pre < sqrt(0.5)) & (A_pre > 0);
            lin_mask = (A_pre >= sqrt(0.5));
            A = zeros(size(A_pre));
            A(cub_mask) = (2/3) * A_pre(cub_mask).^3;
            A(lin_mask) = A_pre(lin_mask) - (sqrt(0.5) - ((2/3)*sqrt(0.5)^3));
            if (nargout > 1)
                dAdF = zeros(size(A_pre));
                dAdF(cub_mask) = 2 * A_pre(cub_mask).^2;
                dAdF(lin_mask) = 1;
                BP_W = @( dLdA, X ) ((dLdA .* dAdF)' * X);
                BP_X = @( dLdA, W ) ((dLdA .* dAdF) * W);
            end
            return
        end
        
        function [ A BP_W BP_X ] = actfun_soft_abs(X, W)
            % Rectified soft-absolute activation function
            EPS = 1e-8;
            F = X*W';
            A = sqrt(F.^2 + EPS);
            if (nargout > 1)
                dAdF = F ./ A;
                BP_W = @( dLdA, X ) ((dLdA .* dAdF)' * X);
                BP_X = @( dLdA, W ) ((dLdA .* dAdF) * W);
            end
            return
        end
        
        function [ A BP_W BP_X ] = actfun_norm_pool(X, W)
            % Square root of weighted sum of squares.
            EPS = 1e-8;
            X_sqr = X.^2;
            W_abs = sqrt(W.^2 + EPS);
            F = (X_sqr*(W_abs')) + EPS;
            A = sqrt(F);
            if (nargout > 1)
                dAdF = 1 ./ (2 * sqrt(F));
                dWdW = W ./ W_abs;
                dXdX = 2*X;
                BP_W = @( dLdA, X ) (((dLdA .* dAdF)' * X.^2) .* dWdW);
                BP_X = @( dLdA, W ) (((dLdA .* dAdF) * W_abs) .* dXdX);
            end
            return
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % DEFAULT PARAMETER SETTING AND CHECKING %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [ opts ] = check_opts( opts, filter_method )
            % Ensure that valid parameters are available for the given method.
            if ~isfield(opts,'init_kmeans')
                % This option determines whether or not to initialize filters
                % using kmeans on the training data.
                opts.init_kmeans = 0;
            else
                if ((opts.init_kmeans ~= 0) && (opts.init_kmeans ~= 1))
                    error('Invalid option value: opts.init_kmeans.');
                end
            end
            if ~isfield(opts,'batch_size')
                opts.batch_size = 5000;
            end
            if ~isfield(opts,'en_opts')
                % These are options for Tikhonov Elastic Net regularization of
                % the filter weights.
                opts.en_opts = struct();
                opts.en_opts.T = 1;
                opts.en_opts.lam = 5e-6;
                opts.en_opts.alpha = 0.9;
            else
                if ~isfield(opts.en_opts,'T')
                    % en_opts.T gives the "Tikhonov" regularization matrix
                    opts.en_opts.T = 1;
                end
                if ~isfield(opts.en_opts,'lam')
                    % en_opts.lam gives the strength of T.E.N. regularization
                    opts.en_opts.lam = 5e-6;
                end
                if ~isfield(opts.en_opts,'alpha')
                    % en_opts.alpha gives the relative weighting of Tikhonov
                    % regularization versus (soft) L1 regularization.
                    opts.en_opts.alpha = 0.9;
                end
            end
            switch filter_method
                case 1
                    % Check options for Sparse Dispersed Filtering
                    if ~isfield(opts,'lam_S')
                        opts.lam_S = 5e-3;
                    end
                    if ~isfield(opts,'lam_D')
                        opts.lam_D = 5e3;
                    end
                    if ~isfield(opts,'avg_act')
                        opts.avg_act = 0.1;
                    end
                case 2
                    % Check options for Sparse Filtering (there are none)
                    opts.sparse_filtering = 'yes';
                    if ~isfield(opts,'avg_act')
                        opts.avg_act = 0.1;
                    end
                case 3
                    % Check options for Reconstruction ICA
                    if ~isfield(opts,'lam_S')
                        opts.lam_S = 0.1;
                    end
                    if ~isfield(opts,'avg_act')
                        opts.avg_act = 0.1;
                    end
                case 4
                    % Check options for Sparse Autoencoder
                    if ~isfield(opts,'lam_S')
                        opts.lam_S = 0.1;
                    end
                    if ~isfield(opts,'avg_act')
                        opts.avg_act = 0.1;
                    end
                case 5
                    % Check options for k-sparse filtering
                    if ~isfield(opts,'lam_S')
                        opts.lam_S = 1e-2;
                    end
                    if ~isfield(opts,'lam_D')
                        opts.lam_D = 1e-2;
                    end
                    if ~isfield(opts,'avg_act')
                        opts.avg_act = 0.1;
                    end
                otherwise
                    error('Invalid filter_method.');
            end
            return
        end
        
        function [ drop_opts ] = check_drop_opts( drop_opts )
            % Ensure that valid parameters are available for droppy training.
            if ~isfield(drop_opts,'lam_drop')
                drop_opts.lam_drop = 0.0;
            end
            if ~isfield(drop_opts,'drop_rate')
                drop_opts.drop_rate = 0.0;
            end
            if ~isfield(drop_opts,'batch_obs')
                drop_opts.batch_obs = 1000;
            end
            if ~isfield(drop_opts,'batch_reps')
                drop_opts.batch_reps = 5;
            end
            return
        end
        
    end % END STATIC METHODS
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Entropy-based sparsity computations %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Compute row-wise entropy
%[Fp_r BP_r] = SimpleFilter.prob_rows(F1+1e-5);
%Sr = -sum(sum((Fp_r .* SimpleFilter.flog(Fp_r))));
%% Compute column-wise entropy
%[Fp_c BP_c] = SimpleFilter.prob_cols(F1+1e-5);
%Sc = -sum(sum((Fp_c .* SimpleFilter.flog(Fp_c))));
% Row-wise entropy gradient
%dSrdFr = -(SimpleFilter.flog(Fp_r) + 1);
%% Column-wise entropy gradient
%dScdFc = -(SimpleFilter.flog(Fp_c) + 1);
%% Combine entropy gradients
%dSdF1 = (lam_Sr * BP_r(dSrdFr)) + (lam_Sc * BP_c(dScdFc));


%%%%%%%%%%%%%%
% EYE BUFFER %
%%%%%%%%%%%%%%
