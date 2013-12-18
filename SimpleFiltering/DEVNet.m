classdef DEVNet < handle

    properties
        % layer_weights stores the weights for each layer of this DEVNet
        layer_weights
        % layer_sizes stores the size of each inter-layer weight matrix
        layer_sizes
        % layer_afuns stores the function handle for effecting the activation
        % function for each layer of this DEVNet
        layer_afuns
        % layer_dev_types stores the DEV type for each layer. The DEV type
        % determines the transform to apply to a layer's droppy ensemble
        % activations before computing variance.
        layer_dev_types
        % bias_val gives the magnitude of the bias in this network
        bias_val
        % class_loss determines the loss function to use for classification
        class_loss
    end % END PROPETIES
    
    methods
        
        function [ self ] = DEVNet(l_weights, l_afuns, b_val, out_type)
            if ~exist('out_type','var')
                out_type = 1;
            end
            if (out_type == 1)
                l_afuns{end} = @DEVNet.actfun_linear;
                self.class_loss = @DEVNet.mcl2h_loss;
            else
                l_afuns{end} = @DEVNet.actfun_softmax;
                self.class_loss = @DEVNet.crent_loss;
            end 
            self.layer_sizes = zeros(length(l_weights),2);
            self.layer_dev_types = zeros(length(l_weights),2);
            for i=1:size(self.layer_sizes,1),
                self.layer_weights{i} = l_weights{i};
                self.layer_afuns{i} = l_afuns{i};
                self.layer_sizes(i,1) = size(l_weights{i},1);
                self.layer_sizes(i,2) = size(l_weights{i},2);
                self.layer_dev_types(i) = 1;
            end
            self.bias_val = b_val;
            return
        end
        
        function [ Xb ] = bias(self, Xub)
            % Shortcut function, to save keystrokes
            Xb = DEVNet.add_bias(Xub, self.bias_val);
            return
        end
        
        function [ Xub ] = unbias(self, Xb)
            % Shortcut function, to save keystrokes
            Xub = Xb(:,1:(end-1));
            return
        end
        
        function [ acc ] = check_acc(self, X, Y)
            % Check accuracy of this net, when considered as a classifier.
            %
            F = self.feedforward(X, 1);
            Yh = DEVNet.class_cats(F);
            acc = sum(Yh == DEVNet.to_cats(Y)) / numel(Yh);
            return
        end
        
        function [ drop_masks ] = ...
                get_drop_masks(self, obs_count, drop_rate, drop_input)
            % Get masks for droppy variance training
            l_count = size(self.layer_sizes,1);
            drop_masks = cell(1,l_count);
            for i=1:l_count,
                obs_dim = size(self.layer_weights{i},2);
                % Make a drop mask, with an extra undropped column, for bias
                if (i == 1)
                    mask = (rand(obs_count,obs_dim) > drop_input);
                else
                    mask = (rand(obs_count,obs_dim) > drop_rate);
                end
                mask(:,end) = 1;
                % Record/store the generated mask for future use
                drop_masks{i} = mask;
            end
            return
        end
        
        function [ F ] = feedforward(self, X, only_output)
            % Do feedforward activation for the observations in X. If
            % only_output is 1, only return activations for the finaly network
            % layer.
            %
            if ~exist('only_output','var')
                only_output = 1;
            end
            l_count = size(self.layer_sizes,1);
            if (only_output == 1)
                F = X;
                for i=1:l_count,
                    W = self.layer_weights{i};
                    afun = self.layer_afuns{i};
                    F = afun(self.bias(F), W);
                end
            else
                F = cell(1,l_count);
                for i=1:l_count,
                    W = self.layer_weights{i};
                    afun = self.layer_afuns{i};
                    if (i == 1)
                        F{i} = afun(self.bias(X), W);
                    else
                        F{i} = afun(self.bias(F{i-1}), W);
                    end
                end
            end
            return
        end
                
        
        function [ W opts drop_opts ] = train_dev(self, ...
                X, Y, i_iters, o_iters, opts, drop_opts)
            % Train a multilayer feedforward network combining classification
            % loss, SPDF loss, and dropout ensemble variance loss.
            %
            % Parameters:
            %   X: training observations
            %   Y: class labels for observations in X
            %   i_iters: number of LBFGS iterations per minibatch
            %   o_iters: number of minibatches to sample and train with
            %   opts: struct containing method-specific options
            %   drop_opts: struct containing dropout related options
            %
            % Outputs:
            %   W: the learned filters (also stored in self.filters)
            %
            if ~exist('opts','var')
                opts = struct();
            end
            if ~exist('drop_opts','var')
                drop_opts = struct();
            end
            % Check and set method specific options to valid values
            opts = DEVNet.check_opts(opts);
            drop_opts = DEVNet.check_drop_opts(drop_opts);
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
            batch_obs = drop_opts.batch_obs;
            batch_reps = drop_opts.batch_reps;
            drop_rate = drop_opts.drop_rate;
            drop_input = drop_opts.drop_input;
            for i=1:o_iters,
                % Grab a batch of training samples
                if (batch_size < size(X,1))
                    idx = randsample(size(X,1),batch_size,false);
                    Xo = X(idx,:);
                    Yo = Y(idx,:);
                else
                    Xo = X;
                    Yo = Y;
                end
                Xs = X(randsample(size(X,1),batch_obs,true),:);
                Xs = repmat(Xs,batch_reps,1);
                % Generate a "drop mask" for each layer in this DEVNet, for the
                % activations for each observation in Xs.
                drop_masks = ...
                    self.get_drop_masks(size(Xs,1), drop_rate, drop_input);
                % Package a function handle for use by minFunc
                mf_func = @( w ) self.dev_loss(w, ...
                    Xo, Yo, Xs, drop_masks, opts, drop_opts);
                % Do some learning using minFunc
                W = DEVNet.vector_weights(self.layer_weights,self.layer_sizes);
                W = minFunc(mf_func, W(:), mf_opts);
                % Record the result of partial optimization
                self.layer_weights = DEVNet.matrix_weights(W,self.layer_sizes);
                % Check new accuracy on training data
                fprintf('    train_acc: %.4f\n',self.check_acc(X,Y));
                if (isfield(opts,'Xv') && isfield(opts,'Yv'))
                    fprintf('    valid_acc: %.4f\n',...
                        self.check_acc(opts.Xv,opts.Yv));
                end
            end
            % Record optimized weights, for returnage
            W = DEVNet.matrix_weights(W,self.layer_sizes);
            return
        end
        
        function [ W opts drop_opts ] = train_cde(self, ...
                X, Y, i_iters, o_iters, opts, drop_opts)
            % Train a multilayer feedforward network combining classification
            % loss, SPDF loss, and dropout ensemble variance loss.
            %
            % Parameters:
            %   X: training observations
            %   Y: class labels for observations in X
            %   i_iters: number of LBFGS iterations per minibatch
            %   o_iters: number of minibatches to sample and train with
            %   opts: struct containing method-specific options
            %   drop_opts: struct containing dropout related options
            %
            % Outputs:
            %   W: the learned filters (also stored in self.filters)
            %
            if ~exist('opts','var')
                opts = struct();
            end
            if ~exist('drop_opts','var')
                drop_opts = struct();
            end
            % Check and set method specific options to valid values
            opts = DEVNet.check_opts(opts);
            drop_opts = DEVNet.check_drop_opts(drop_opts);
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
            drop_rate = drop_opts.drop_rate;
            drop_input = drop_opts.drop_input;
            for i=1:o_iters,
                % Grab a batch of training samples
                idx = randsample(size(X,1),batch_size,true);
                Xb = X(idx,:);
                Yb = Y(idx,:);
                % Generate a "drop mask" for each layer in this DEVNet, for the
                % activations for each observation in Xb.
                drop_masks = ...
                    self.get_drop_masks(size(Xb,1), drop_rate, drop_input);
                % Package a function handle for use by minFunc
                mf_func = @( w ) self.cde_loss(w, Xb, Yb, drop_masks, opts);
                % Do some learning using minFunc
                W = DEVNet.vector_weights(self.layer_weights,self.layer_sizes);
                W = minFunc(mf_func, W(:), mf_opts);
                % Record the result of partial optimization
                self.layer_weights = DEVNet.matrix_weights(W,self.layer_sizes);
                % Check new accuracy on training data
                fprintf('    train_acc: %.4f\n',self.check_acc(X,Y));
                if (isfield(opts,'Xv') && isfield(opts,'Yv'))
                    fprintf('    valid_acc: %.4f\n',...
                        self.check_acc(opts.Xv,opts.Yv));
                end
            end
            % Record optimized weights, for returnage
            W = DEVNet.matrix_weights(W,self.layer_sizes);
            return
        end
        
        function [ W opts drop_opts ] = check_grad(self, ...
                X, Y, i_iters, o_iters, grad_checks, opts, drop_opts)
            % Train a multilayer feedforward network combining classification
            % loss, SPDF loss, and dropout ensemble variance loss.
            %
            % Parameters:
            %   X: training observations
            %   Y: class labels for observations in X
            %   i_iters: number of LBFGS iterations per minibatch
            %   o_iters: number of minibatches to sample and train with
            %   grad_checks: number of fast gradient checks to perform
            %   opts: struct containing method-specific options
            %   drop_opts: struct containing dropout related options
            %
            % Outputs:
            %   W: the learned filters (also stored in self.filters)
            %
            if ~exist('opts','var')
                opts = struct();
            end
            if ~exist('drop_opts','var')
                drop_opts = struct();
            end
            % Check and set method specific options to valid values
            opts = DEVNet.check_opts(opts);
            drop_opts = DEVNet.check_drop_opts(drop_opts);
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
            batch_obs = drop_opts.batch_obs;
            batch_reps = drop_opts.batch_reps;
            drop_rate = drop_opts.drop_rate;
            drop_input = drop_opts.drop_input;
            W = DEVNet.vector_weights(self.layer_weights,self.layer_sizes);
            for i=1:o_iters,
                % Grab a batch of training samples
                if (opts.batch_size < size(X,1))
                    idx = randsample(size(X,1),opts.batch_size,false);
                    Xo = X(idx,:);
                    Yo = Y(idx,:);
                else
                    Xo = X;
                    Yo = Y;
                end
                Xs = X(randsample(size(X,1),batch_obs,true),:);
                Xs = repmat(Xs,batch_reps,1);
                % Generate a "drop mask" for each layer in this DEVNet, for the
                % activations for each observation in Xs.
                drop_masks = ...
                    self.get_drop_masks(size(Xs,1), drop_rate, drop_input);
                %drop_masks = ...
                %    self.get_drop_masks(size(Xo,1), drop_rate, drop_input);
                % Package a function handle for use by minFunc
                mf_func = @( w ) self.dev_loss(w, ...
                    Xo, Yo, Xs, drop_masks, opts, drop_opts);
                %mf_func = @( w ) self.cde_loss(w, Xo, Yo, drop_masks, opts);
                % Do some learning using minFunc
                W = minFunc(mf_func, W(:), mf_opts);
            end
            order = 1;
            type = 2;
            for i=1:grad_checks,
                fprintf('=============================================\n');
                fprintf('GRAD CHECK %d\n',i);
                fastDerivativeCheck(mf_func,W,order,type);
            end
            return
        end
        
        function [ L dLdW ] = dev_loss(self, w,...
                Xo, Yo, Xs, drop_masks, opts, drop_opts)
            % Function for combining classification, SPDF, and droppy ensemble
            % variance losses, for use by minFunc.
            %
            l_count = size(self.layer_sizes,1);
            l_weights = DEVNet.matrix_weights(w, self.layer_sizes);
            l_afuns = self.layer_afuns;
            l_F = cell(1, l_count);
            l_dLdF = cell(1, l_count);
            l_bp_dFdW = cell(1, l_count);
            l_bp_dFdX = cell(1, l_count);
            % Perform a feedforward pass, collecting activations and function
            % handles for backproppin' and lockin' along the way.
            Lc = 0;
            for i=1:l_count,
                l_afun = l_afuns{i};
                if (i == 1)
                    Xi = self.bias(Xo);
                else
                    Xi = self.bias(l_F{i-1});
                end
                [F bp_dFdW bp_dFdX] = l_afun(Xi, l_weights{i});
                if (i < l_count)
                    dLdF = zeros(size(F));
                else
                    % Do classification loss at the output layer, instead of
                    % SPDF loss (if this net is training for classification).
                    [L dLdF] = self.class_loss(F, DEVNet.to_inds(Yo));
                    Lc = Lc + (opts.lam_class * L);
                    dLdF = opts.lam_class * dLdF;
                end
                l_F{i} = F;
                l_dLdF{i} = dLdF;
                l_bp_dFdW{i} = bp_dFdW;
                l_bp_dFdX{i} = bp_dFdX;
            end
            % Perform a backprop pass, using information about activations and
            % gradients collected during the feedforward pass.
            l_dLcdW = cell(1, l_count);
            for i=l_count:-1:1,
                dLdF = l_dLdF{i};
                bp_dFdW = l_bp_dFdW{i};
                bp_dFdX = l_bp_dFdX{i};
                if ((i == l_count) || (opts.class_only ~= 1))
                    if (i > 1)
                        l_dLcdW{i} = bp_dFdW(dLdF, self.bias(l_F{i-1}));
                        l_dLdF{i-1} = l_dLdF{i-1} + ...
                            self.unbias(bp_dFdX(dLdF, l_weights{i}));
                    else
                        l_dLcdW{i} = bp_dFdW(dLdF, self.bias(Xo));
                    end
                else
                    l_dLcdW{i} = zeros(size(l_weights{i}));
                end
            end
            dLcdW = DEVNet.vector_weights(l_dLcdW, self.layer_sizes);
            clear('l_dLcdW');
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % DROPOUT ENSEMBLE VARIANCE LOSS AND GRADIENT %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            l_F = cell(1, l_count);
            l_dLdF = cell(1, l_count);
            l_bp_dFdW = cell(1, l_count);
            l_bp_dFdX = cell(1, l_count);
            % Perform a feedforward pass, collecting activations and function
            % handles for backproppin' and lockin' along the way.
            Ld = 0;
            batch_obs = drop_opts.batch_obs;
            batch_reps = drop_opts.batch_reps;
            for i=1:l_count,
                l_afun = l_afuns{i};
                if (i == 1)
                    Xl = self.bias(Xs);
                else
                    Xl = self.bias(l_F{i-1});
                end
                Xl = Xl .* drop_masks{i};
                [F bp_dFdW bp_dFdX] = l_afun(Xl, l_weights{i});
                [L dLdF] = DEVNet.drop_loss(F, batch_obs, batch_reps, ...
                    self.layer_dev_types(i));
                if ((i == l_count) || (opts.class_only ~= 1))
                    Ld = Ld + (drop_opts.lam_drop * L);
                end
                l_F{i} = F;
                l_dLdF{i} = drop_opts.lam_drop * dLdF;
                l_bp_dFdW{i} = bp_dFdW;
                l_bp_dFdX{i} = bp_dFdX;
            end
            % Perform a backprop pass, using information about activations and
            % gradients collected during the feedforward pass.
            l_dLddW = cell(1, l_count);
            for i=l_count:-1:1,
                dLdF = l_dLdF{i};
                bp_dFdW = l_bp_dFdW{i};
                bp_dFdX = l_bp_dFdX{i};
                if ((i == l_count) || (opts.class_only ~= 1))
                    if (i > 1)
                        l_dLddW{i} = bp_dFdW(dLdF, ...
                            (drop_masks{i} .* self.bias(l_F{i-1})));
                        l_dLdF{i-1} = l_dLdF{i-1} + self.unbias(...
                            (drop_masks{i} .* bp_dFdX(dLdF, l_weights{i})));
                    else
                        l_dLddW{i} = bp_dFdW(dLdF, ...
                            (drop_masks{i} .* self.bias(Xs)));
                    end
                else
                    l_dLddW{i} = zeros(size(l_weights{i}));
                end
            end
            dLddW = DEVNet.vector_weights(l_dLddW, self.layer_sizes);
            clear('l_dLddW');
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % ELASTIC NET LOSS AND GRADIENT %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            l_dLedW = cell(1, l_count);
            Le = 0;
            for i=1:l_count,
                [L dLdW] = DEVNet.elnet_loss(l_weights{i},opts.en_opts);
                if ((i == l_count) || (opts.class_only ~= 1))
                    Le = Le + (opts.en_opts.lam * L);
                    l_dLedW{i} = opts.en_opts.lam * dLdW;
                else
                    l_dLedW{i} = zeros(size(dLdW));
                end
            end
            dLedW = DEVNet.vector_weights(l_dLedW, self.layer_sizes);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % COMPUTE COMBINED LOSS AND GRADIENT %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            fprintf('          Lc: %.6f, Ld: %.6f, Le: %.6f\n', Lc, Ld, Le);
            L = Lc + Ld + Le;
            dLdW = dLcdW + dLddW + dLedW;
            return
        end
        
        function [ L dLdW ] = cde_loss(self, w, X, Y, drop_masks, opts)
            % Function for copmuting class dropout ensemble loss.
            %
            l_count = size(self.layer_sizes,1);
            l_weights = DEVNet.matrix_weights(w, self.layer_sizes);
            l_afuns = self.layer_afuns;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % DROPOUT ENSEMBLE VARIANCE LOSS AND GRADIENT %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            l_F = cell(1, l_count);
            l_dLdF = cell(1, l_count);
            l_bp_dFdW = cell(1, l_count);
            l_bp_dFdX = cell(1, l_count);
            % Perform a feedforward pass, collecting activations and function
            % handles for backproppin' and lockin' along the way.
            Lc = 0;
            for i=1:l_count,
                l_afun = l_afuns{i};
                if (i == 1)
                    Xi = self.bias(X);
                else
                    Xi = self.bias(l_F{i-1});
                end
                Xi = Xi .* drop_masks{i};
                [F bp_dFdW bp_dFdX] = l_afun(Xi, l_weights{i});
                if (i == l_count)
                    % Do classification loss at the output layer, instead of
                    % SPDF loss (if this net is training for classification).
                    [L dLdF] = self.class_loss(F, DEVNet.to_inds(Y));
                    Lc = Lc + (opts.lam_class * L);
                    dLdF = opts.lam_class * dLdF;
                else
                    dLdF = zeros(size(F));
                end
                l_F{i} = F;
                l_dLdF{i} = dLdF;
                l_bp_dFdW{i} = bp_dFdW;
                l_bp_dFdX{i} = bp_dFdX;
            end
            % Perform a backprop pass, using information about activations and
            % gradients collected during the feedforward pass.
            l_dLcdW = cell(1, l_count);
            for i=l_count:-1:1,
                dLdF = l_dLdF{i};
                bp_dFdW = l_bp_dFdW{i};
                bp_dFdX = l_bp_dFdX{i};
                if ((i == l_count) || (opts.class_only ~= 1))
                    if (i > 1)
                        l_dLcdW{i} = bp_dFdW(dLdF, ...
                            (drop_masks{i} .* self.bias(l_F{i-1})));
                        l_dLdF{i-1} = l_dLdF{i-1} + self.unbias(...
                            (drop_masks{i} .* bp_dFdX(dLdF, l_weights{i})));
                    else
                        l_dLcdW{i} = bp_dFdW(dLdF, ...
                            (drop_masks{i} .* self.bias(X)));
                    end
                else
                    l_dLcdW{i} = zeros(size(l_weights{i}));
                end
            end
            dLcdW = DEVNet.vector_weights(l_dLcdW, self.layer_sizes);
            clear('l_dLcdW');
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % ELASTIC NET LOSS AND GRADIENT %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            l_dLedW = cell(1, l_count);
            Le = 0;
            for i=1:l_count,
                [L dLdW] = DEVNet.elnet_loss(l_weights{i},opts.en_opts);
                if ((i == l_count) || (opts.class_only ~= 1))
                    Le = Le + (opts.en_opts.lam * L);
                    l_dLedW{i} = opts.en_opts.lam * dLdW;
                else
                    l_dLedW{i} = zeros(size(dLdW));
                end
            end
            dLedW = DEVNet.vector_weights(l_dLedW, self.layer_sizes);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % COMPUTE COMBINED LOSS AND GRADIENT %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            fprintf('          Lc: %.6f, Le: %.6f\n', Lc, Le);
            L = Lc + Le;
            dLdW = dLcdW + dLedW;
            return
        end

    end % END INSTANCE METHODS
    
    methods (Static = true)
        
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
                    [F bp_F] = DEVNet.norm_rows(F);
                case 2
                    [F bp_F] = DEVNet.tanh_transform(F);
                case 3
                    [F bp_F] = DEVNet.dont_transform(F);
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
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % CLASSIFICATION LOSS FUNCTIONS %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [ L dLdYh ] = mcl2h_loss(Yh, Y)
            % Compute a multiclass L2 hinge loss and its gradients, w.r.t. the
            % proposed outputs Yh, given the true values Y.
            %
            cl_count = size(Y,2);
            [Y_max Y_idx] = max(Y,[],2);
            % Make a class indicator matrix using +1/-1
            Yc = bsxfun(@(y1,y2) (2*(y1==y2))-1, Y_idx, 1:cl_count);
            % Compute current L2 hinge loss given the predictions in Yh
            margin_lapse = max(0, 1 - (Yc .* Yh));
            L = 0.5 * margin_lapse.^2;
            L = sum(L(:)) / size(Yc,1);
            if (nargout > 1)
                % For L2 hinge loss, dL is equal to the margin intrusion
                dLdYh = -Yc .* margin_lapse;
                dLdYh = dLdYh ./ size(Yc,1);
            end
            return
        end
        
        function [ L dLdYh ] = crent_loss(Yh, Y)
            % Compute cross-entropy loss, given the +1/-1 indicator matrix Y
            % containing the true target classes, and the matrix Yh, in which
            % each row describes a distribution over its columns. Presumably,
            % the values in Yh were derived from DEVNet.actfun_softmax...
            %
            class_mask = (Y > 0);
            log_probs = log(Yh);
            L = -sum(sum(log_probs(class_mask))) / size(Y,1);
            dLdYh = -(class_mask ./ Yh) ./ size(Y,1);
            return
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % ELASTIC NET(ish) WEIGHT LOSS %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [ L dLdW ] = elnet_loss( W, en_opts )
            % Soft-absolute value penalty and gradient for weights matrix W.
            %
            T = eye(size(W,2));
            alpha = en_opts.alpha;
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
        
        %%%%%%%%%%%%%%%%%%%%%%%%
        % ACTIVATION FUNCTIONS %
        %%%%%%%%%%%%%%%%%%%%%%%%
        
        function [ A BP_W BP_X ] = actfun_linear(X, W)
            % Linear activation function
            A = X*W';
            if (nargout > 1)
                BP_W = @( dLdA, X ) (dLdA' * X);
                BP_X = @( dLdA, W ) (dLdA * W);
            end
            return
        end
        
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
        
        function [ A BP_W BP_X ] = actfun_softmax(X, W)
            % Hyperbolic tangent activation function
            F = X*W';
            Fe = exp(F);
            A = bsxfun(@rdivide, Fe, sum(Fe,2));
            if (nargout > 1)
                BP_W = @( dLdA, X ) (((dLdA .* A) - ...
                    bsxfun(@times, A, sum((dLdA .* A),2)))' * X);
                BP_X = @( dLdA, W ) (((dLdA .* A) - ...
                    bsxfun(@times, A, sum((dLdA .* A),2))) * W);
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
        
        %%%%%%%%%%%%%%%%%%%%
        % HELPER FUNCTIONS %
        %%%%%%%%%%%%%%%%%%%%
        
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
            Yi = DEVNet.class_inds(Yc);
            Yc = DEVNet.class_cats(Yi);
            Yi = DEVNet.class_inds(Yc);
            return
        end
        
        function [ Yc ] = to_cats( Yc )
            % This wraps class_cats and class_inds.
            Yi = DEVNet.class_inds(Yc);
            Yc = DEVNet.class_cats(Yi);
            return
        end
        
        function [ Wv ] = vector_weights(Wm, Wm_sz)
            % Vectorize the cell array of weight matrices in Wm.
            %
            Wm_count = size(Wm_sz,1);
            tel_count = 0;
            for i=1:Wm_count,
                tel_count = tel_count + (Wm_sz(i,1) * Wm_sz(i,2));
                if ((Wm_sz(i,1) * Wm_sz(i,2)) ~= numel(Wm{i}))
                    error('Incorrect weight matrix sizes.');
                end
            end        
            Wv = zeros(tel_count,1);
            end_idx = 0;
            for i=1:length(Wm),
                Wi = Wm{i};
                start_idx = end_idx + 1;
                end_idx = start_idx + (numel(Wi) - 1);
                Wv(start_idx:end_idx) = reshape(Wi,numel(Wi),1);
            end
            return
        end
        
        function [ Wm ] = matrix_weights(Wv, Wm_sz)
            % Convert the vector of weights Wv into a cell array of inter-layer
            % weight matrices, based on the sizes of the inter-layer weight
            % matrices in Wm_sz.
            %
            Wm_count = size(Wm_sz,1);
            tel_count = 0;
            for i=1:Wm_count,
                tel_count = tel_count + (Wm_sz(i,1) * Wm_sz(i,2));
            end
            if (tel_count ~= numel(Wv))
                error('Incorrect weight vector size.');
            end
            Wm = cell(1,Wm_count);
            end_idx = 0;
            for i=1:length(Wm),
                start_idx = end_idx + 1;
                end_idx = start_idx + ((Wm_sz(i,1) * Wm_sz(i,2)) - 1);
                Wm{i} = reshape(Wv(start_idx:end_idx), Wm_sz(i,1), Wm_sz(i,2));
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
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % DEFAULT PARAMETER SETTING AND CHECKING %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [ opts ] = check_opts( opts )
            % Ensure that valid parameters are available for the given method.
            if ~isfield(opts,'en_opts')
                % These are options for Tikhonov Elastic Net regularization of
                % the filter weights.
                opts.en_opts = struct();
                opts.en_opts.lam = 5e-6;
                opts.en_opts.alpha = 0.9;
            else
                if ~isfield(opts.en_opts,'lam')
                    % en_opts.lam gives the strength of T.E.N. regularization
                    opts.en_opts.lam = 5e-6;
                end
                if ~isfield(opts.en_opts,'alpha')
                    % en_opts.alpha gives the relative weighting of Tikhonov
                    % regularization versus (soft) L1 regularization.
                    opts.en_opts.alpha = 0.95;
                end
            end
            if ~isfield(opts,'batch_size')
                opts.batch_size = 2500;
            end
            % If opts.class_only == 1, then only class layer is trained
            if ~isfield(opts,'class_only')
                opts.class_only = 0;
            end
            % Set default classification and SPDF weights, if not given
            if ~isfield(opts,'lam_class')
                opts.lam_class = 1.0;
            end
            return
        end
        
        function [ drop_opts ] = check_drop_opts( drop_opts )
            % Ensure that valid parameters are available for droppy training.
            if ~isfield(drop_opts,'lam_drop')
                drop_opts.lam_drop = 0.0;
            end
            if ~isfield(drop_opts,'drop_input')
                drop_opts.drop_input = 0;
            end
            if ~isfield(drop_opts,'drop_rate')
                drop_opts.drop_rate = 0.0;
            end
            if ~isfield(drop_opts,'batch_obs')
                drop_opts.batch_obs = 500;
            end
            if ~isfield(drop_opts,'batch_reps')
                drop_opts.batch_reps = 5;
            end
            return
        end
        
    end % END STATIC METHODS
    
end


%%%%%%%%%%%%%%
% EYE BUFFER %
%%%%%%%%%%%%%%
