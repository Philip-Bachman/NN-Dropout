classdef LinMOE < handle
    % This class manages multiclass linear mixtures of self-assertive experts.
    %
    
    properties
        % moe_wts gives the weights for the experts for each class, and its
        % dimensions contain all size info for this LinMOE.
        %
        % size(moe_wts) = (class_count, moe_size, obs_dim)
        %
        moe_wts
        % lam_l2sqz moderates how experts in each class are squeezed together.
        lam_l2sqz
        % lam_l2nrm moderates the euclidean norm of each expert
        lam_l2nrm
        % class_loss is a handle to the desired classification loss function
        class_loss
        % use_max tells whether to actual max or exp-based softmax.
        %   note: exp softmax is more correct mathematically, but subject to
        %         issues with numeric overflow and inf/nan with big values.
        use_max
    end
    
    methods
        
        function [ self ] = LinMOE(X, Y, moe_size, kill_bias)
            % Constructor for LinMOE class
            if ~exist('kill_bias','var')
                kill_bias = 0;
            end
            self.lam_l2sqz = 1e-3;
            self.lam_l2nrm = 1.0;
            self.class_loss = @(Yh, Y) LinMOE.loss_hinge(Yh, Y);
            self.use_max = 1;
            L = self.init_weights(X, Y, moe_size, kill_bias);
            fprintf('Initial loss: %.4f\n',mean(L(:)));

            return
        end
        
        function [ L ] = init_weights(self, X, Y, moe_size, kill_bias)
            % Initialize weights for this LinMOE, using the observations and
            % classes in X/Y.
            %
            if ~exist('kill_bias','var')
                kill_bias = 0;
            end
            opts = statset('Display','off');
            obs_dim = size(X,2);
            class_count = size(Y,2);
            self.moe_wts = zeros(class_count,moe_size,obs_dim);
            for c=1:class_count,
                Xc = X((Y(:,c) > 0),:);
                [idx ctrs] = kmeans(Xc,moe_size,'Distance','sqEuclidean',...
                    'emptyaction','singleton','Options',opts);
                ctrs = bsxfun(@rdivide, ctrs, max(1e-10,sqrt(sum(ctrs.^2,2))));
                if (kill_bias == 1)
                    ctrs(:,end) = 1e-3 * randn(size(ctrs(:,end)));
                end
                self.moe_wts(c,:,:) = ctrs;
            end
            if (self.use_max > 0)
                L = LinMOE.loss_linmoe_max(X,Y,self.moe_wts,self.class_loss);
            else
                L = LinMOE.loss_linmoe_exp(X,Y,self.moe_wts,self.class_loss);
            end
            return
        end
        
        function [ L ] = init_weights_randn(self, X, Y, moe_size)
            % Initialize weights for this LinMOE, using the observations and
            % classes in X/Y.
            %
            obs_dim = size(X,2);
            class_count = size(Y,2);
            self.moe_wts = zeros(class_count,moe_size,obs_dim);
            for c=1:class_count,
                Wc = randn(moe_size,obs_dim);
                Wc = bsxfun(@rdivide, Wc, max(sqrt(sum(Wc.^2,2)),1e-10));
                self.moe_wts(c,:,:) = 0.1 * Wc;
            end
            if (self.use_max > 0)
                L = LinMOE.loss_linmoe_max(X,Y,self.moe_wts,self.class_loss);
            else
                L = LinMOE.loss_linmoe_exp(X,Y,self.moe_wts,self.class_loss);
            end
            return
        end
        
        function [ F ] = evaluate(self, X)
            % Do either exp or maxxy outputs
            %
            if (self.use_max > 0)
                F = self.outputs_max(X);
            else
                F = self.outputs_exp(X);
            end
            return
        end
        
        function [ F ] = outputs_max(self, X)
            % Evaluate this LinMOE using the current weights, for pts in X
            %
            obs_count = size(X,1);
            class_count = size(self.moe_wts,1);
            F = zeros(obs_count,class_count);
            for c=1:class_count,
                wc = squeeze(self.moe_wts(c,:,:));
                if (size(self.moe_wts,2) == 1)
                    wc = wc';
                end
                Fc = X * wc';
                F(:,c) = max(Fc,[],2);
            end
            return
        end
        
        function [ F ] = outputs_exp(self, X)
            % Compute outputs for this LinMOE using the current weights.
            %
            obs_count = size(X,1);
            class_count = size(self.moe_wts,1);
            F = zeros(obs_count,class_count);
            for c=1:class_count,
                wc = squeeze(self.moe_wts(c,:,:));
                if (size(self.moe_wts,2) == 1)
                    wc = wc';
                end
                Fc = X * wc';
                Fc_exp = exp(Fc);
                Fc_sum = bsxfun(@rdivide, Fc_exp, sum(Fc_exp,2));
                F(:,c) = sum((Fc .* Fc_sum), 2);
            end
            return
        end           
            
    
        function [ result ] = train(self, X, Y, opts)
            % Use minFunc to update the weight in this LinMOE.
            %
            % Parameters:
            %   X: training input observations
            %   Y: training class matrix (in +1/-1 indicator form)
            %   opts: optional options struct for minFunc
            % Outputs:
            %   result: training result info
            %  
            if ~exist('opts','var')
                % Setup options structure for minFunc
                opts = struct();
                opts.Display = 'iter';
                opts.Method = 'lbfgs';
                opts.Corr = 10;
                opts.LS = 0;
                opts.LS_init = 0;
                opts.MaxIter = 200;
                opts.MaxFunEvals = 500;
                opts.TolX = 1e-10;
            end
            % Get relevant dimension info from self.moe_wts
            class_count = size(self.moe_wts,1);
            moe_size = size(self.moe_wts,2);
            obs_dim = size(self.moe_wts,3);
            % Setup a loss function for use by minFunc
            loss_func = @( w ) ...
                self.loss_wrapper(w, X, Y, class_count, moe_size, obs_dim);
            % Run minFunc to compute optimal SVM parameters
            Wi = reshape(self.moe_wts, class_count*moe_size*obs_dim, 1);
            Wf = minFunc(loss_func, Wi, opts);
            self.moe_wts = reshape(Wf, class_count, moe_size, obs_dim);
            % Clip weights to fixed norm ball
            self.clip_weights(sqrt(obs_dim));
            % Record result info
            result = struct();
            result.W_pre = reshape(Wi, class_count, moe_size, obs_dim);
            result.W_post = reshape(Wf, class_count, moe_size, obs_dim);
            return
        end
        
        function [ result ] = clip_weights(self, max_norm)
            % Clip weights to an L2 ball of radius max_norm.
            %
            W = self.moe_wts;
            class_count = size(W,1);
            for c=1:class_count,
                Wc = squeeze(W(c,:,:));
                if (size(W,2) == 1)
                    Wc = Wc';
                end
                Wc_norms = sqrt(sum(Wc.^2,2));
                Wc_noclip = Wc_norms > 1e10; %max_norm;
                Wc_scales = max_norm ./ Wc_norms;
                Wc_scales(Wc_noclip) = 1;
                Wc = bsxfun(@times, Wc, Wc_scales);
                W(c,:,:) = Wc;
            end
            self.moe_wts = W;
            result = 1;
            return
        end
            
            
        
        function [L dLdW] = loss_wrapper(self, w, X, Y, c_count, m_size, o_dim)
            % Wrapper of LinMOE classification loss for use with minFunc. This
            % also adds some, hopefully, helpful L2 regularization.
            %
            o_count = size(X,1);
            W = reshape(w, c_count, m_size ,o_dim);
            if (nargout == 1)
                if (self.use_max > 0)
                    L_cls = LinMOE.loss_linmoe_max(X, Y, W, self.class_loss);
                else
                    L_cls = LinMOE.loss_linmoe_exp(X, Y, W, self.class_loss);
                end
                L_l2nrm = 0.5 * self.lam_l2nrm * (W.^2);
                L_l2sqz = zeros(size(W));
                for c=1:c_count,
                    wc = squeeze(W(c,:,:));
                    if (size(self.moe_wts,2) == 1)
                        wc = wc';
                    end
                    c_mean = mean(wc,1);
                    wc_sqz = bsxfun(@minus, wc, c_mean);
                    L_l2sqz(c,:,:) = 0.5 * self.lam_l2sqz * (wc_sqz.^2);
                end
                L = (sum(L_cls(:)) / o_count) + ...
                    ((sum(L_l2nrm(:)) + sum(L_l2sqz(:))) / m_size);
            else
                if (self.use_max > 0)
                    [L_cls dLdWc] = ...
                        LinMOE.loss_linmoe_max(X, Y, W, self.class_loss); 
                else
                    [L_cls dLdWc] = ...
                        LinMOE.loss_linmoe_exp(X, Y, W, self.class_loss);
                end
                L_l2nrm = 0.5 * self.lam_l2nrm * (W.^2);
                L_l2sqz = zeros(size(W));
                Ws = zeros(size(W));
                for c=1:c_count,
                    wc = squeeze(W(c,:,:));
                    if (size(self.moe_wts,2) == 1)
                        wc = wc';
                    end
                    c_mean = mean(wc,1);
                    wc_sqz = bsxfun(@minus, wc, c_mean);
                    L_l2sqz(c,:,:) = 0.5 * self.lam_l2sqz * (wc_sqz.^2);
                    Ws(c,:,:) = wc_sqz;
                end
                L = (sum(L_cls(:)) / o_count) + ...
                    ((sum(L_l2nrm(:)) + sum(L_l2sqz(:))) / m_size);
                dLdW = dLdWc + ...
                    (((self.lam_l2nrm * W) + (self.lam_l2sqz * Ws)) ./ m_size);
                dLdW = reshape(dLdW, c_count*m_size*o_dim, 1);
            end
            return
        end
        
        function [ L ] = obs_loss(self, X, Y)
            % Do either maxxy or exppy multiclass MOE loss for X/Y.
            %
            if (self.use_max > 0)
                L = self.obs_loss_max(X,Y);
            else
                L = self.obs_loss_exp(X,Y);
            end
            return
        end 
        
        function [ L ] = obs_loss_max(self, X, Y)
            % Compute multiclass loss for linear mixture of self-asserive
            % experts for the observations and classes in X/Y, given the MOE
            % weights in W.
            %
            W = self.moe_wts;
            class_count = size(W,1);
            % Fc contains the per-MOE output for each observation
            Fc = zeros(size(Y));
            for c=1:class_count,
                wc = squeeze(W(c,:,:));
                if (size(W,2) == 1)
                    wc = wc';
                end
                fc = X * wc';
                Fc(:,c) = max(fc,[],2);
            end
            L = self.class_loss(Fc, Y);
            return
        end
        
        function [ L ] = obs_loss_exp(self, X, Y)
            % Compute multiclass loss for linear mixture of self-asserive
            % experts for the observations and classes in X/Y, given the MOE
            % weights in W.
            %
            W = self.moe_wts;
            class_count = size(W,1);
            % Fc contains the per-MOE output for each observation
            Fc = zeros(size(Y));
            for c=1:class_count,
                wc = squeeze(W(c,:,:));
                if (size(W,2) == 1)
                    wc = wc';
                end
                fc = X * wc';
                fc_exp = exp(fc);
                fc_sum = bsxfun(@rdivide, fc_exp, sum(fc_exp,2));
                Fc(:,c) = sum((fc .* fc_sum), 2);
            end
            L = self.class_loss(Fc, Y);
            return
        end
        
        function [ dLdX ] = obs_grads(self, X, Y)
            % Compute maxxy or exppy grads of the loss w.r.t. the inputs X/Y.
            %
            if (self.use_max > 0)
                dLdX = self.obs_grads_max(X, Y);
            else
                dLdX = self.obs_grads_exp(X, Y);
            end
            return
        end
        
        function [ dLdX ] = obs_grads_max(self, X, Y)
            % Compute multiclass loss and gradients of linear mixture of
            % self-assertive experts for the observations and classes in X/Y,
            % given the MOE weights in W. Compute gradients w.r.t. X.
            %
            W = self.moe_wts;
            class_count = size(W,1);
            % Fc contains the per-MOE output for each observation
            Fc = zeros(size(Y));
            for c=1:class_count,
                wc = squeeze(W(c,:,:));
                if (size(W,2) == 1)
                    wc = wc';
                end
                fc = X * wc';
                Fc(:,c) = max(fc,[],2);
            end
            [L dL] = self.class_loss(Fc, Y);
            dLdX = zeros(size(X));
            for c=1:class_count,
                wc = squeeze(W(c,:,:));
                if (size(W,2) == 1)
                    wc = wc';
                end
                fc = X * wc';
                fc_max = max(fc,[],2);
                dFdE = bsxfun(@ge, fc, fc_max);
                dLdF = dL(:,c);
                dLdE = bsxfun(@times, dFdE, dLdF);
                dLdX = dLdX + (dLdE * wc);
            end
            return
        end

        function [ dLdX ] = obs_grads_exp(self, X, Y)
            % Compute multiclass loss and gradients of linear mixture of
            % self-assertive experts for the observations and classes in X/Y,
            % given the MOE weights in W. Compute gradients w.r.t. X.
            %
            W = self.moe_wts;
            class_count = size(W,1);
            % Fc contains the per-MOE output for each observation
            Fc = zeros(size(Y));
            for c=1:class_count,
                wc = squeeze(W(c,:,:));
                if (size(W,2) == 1)
                    wc = wc';
                end
                fc = X * wc';
                fc_exp = exp(fc);
                fc_sum = bsxfun(@rdivide, fc_exp, sum(fc_exp,2));
                Fc(:,c) = sum((fc .* fc_sum), 2);
            end
            [L dL] = self.class_loss(Fc, Y);
            dLdX = zeros(size(X));
            for c=1:class_count,
                wc = squeeze(W(c,:,:));
                if (size(W,2) == 1)
                    wc = wc';
                end
                fc_raw = X * wc';
                fc_exp = exp(fc_raw);
                fc_sum = sum(fc_exp,2);
                %
                % Big gradient computation coming up, so watch out! (how?)
                %
                % syms x y x;
                % f = (x * exp(x)) / (exp(x) + exp(y) + exp(z));
                % pretty(simple(diff(f,x)));
                %
                % Let numerator of ^^^ be exp(2x) + MAGIC ...
                %
                MAGIC = fc_exp.*((fc_raw+1).*bsxfun(@minus,fc_sum,fc_exp));
                dFdE = bsxfun(@rdivide, (fc_exp.^2 + MAGIC), fc_sum.^2);
                dLdF = dL(:,c);
                dLdE = bsxfun(@times, dFdE, dLdF);
                dLdX = dLdX + (dLdE * wc);
            end
            return
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%
    % STATIC METHODS %
    %%%%%%%%%%%%%%%%%%
    
    methods (Static = true)
        
        function [ Xb ] = bias(X)
            % Add a column of constant bias to the observations in X
            Xb = [X ones(size(X,1),1)];
            return
        end
        
        function [ L dLdW ] = loss_linmoe_max(X, Y, W, loss_func)
            % Compute multiclass loss and gradients of linear mixture of
            % self-assertive experts for the observations and classes in X/Y,
            % given the MOE weights in W.
            %
            class_count = size(W,1);
            % Fc contains the per-MOE output for each observation
            Fc = zeros(size(Y));
            for c=1:class_count,
                wc = squeeze(W(c,:,:));
                if (size(W,2) == 1)
                    wc = wc';
                end
                fc = X * wc';
                Fc(:,c) = max(fc,[],2);
            end
            [L dL] = loss_func(Fc, Y);
            if (nargout > 1)
                dLdW = zeros(size(W));
                for c=1:class_count,
                    wc = squeeze(W(c,:,:));
                    if (size(W,2) == 1)
                        wc = wc';
                    end
                    fc = X * wc';
                    fc_max = max(fc,[],2);
                    dFdE = bsxfun(@ge, fc, fc_max);
                    dLdF = dL(:,c);
                    dLdE = bsxfun(@times, dFdE, dLdF);
                    dLdW(c,:,:) = (dLdE' * X) ./ size(X,1);
                end 
            end
            return
        end
        
        function [ L dLdW ] = loss_linmoe_exp(X, Y, W, loss_func)
            % Compute multiclass loss and gradients of linear mixture of
            % self-assertive experts for the observations and classes in X/Y,
            % given the MOE weights in W.
            %
            class_count = size(W,1);
            % Fc contains the per-MOE output for each observation
            Fc = zeros(size(Y));
            for c=1:class_count,
                wc = squeeze(W(c,:,:));
                if (size(W,2) == 1)
                    wc = wc';
                end
                fc = X * wc';
                fc_exp = exp(fc);
                fc_sm = bsxfun(@rdivide, fc_exp, sum(fc_exp,2));
                Fc(:,c) = sum((fc .* fc_sm), 2);
            end
            [L dL] = loss_func(Fc, Y);
            if (nargout > 1)
                dLdW = zeros(size(W));
                for c=1:class_count,
                    wc = squeeze(W(c,:,:));
                    if (size(W,2) == 1)
                        wc = wc';
                    end
                    fc_raw = X * wc';
                    fc_exp = exp(fc_raw);
                    fc_sum = sum(fc_exp,2);
                    %
                    % Big gradient computation coming up, so watch out! (how?)
                    %
                    % syms x y x;
                    % f = (x * exp(x)) / (exp(x) + exp(y) + exp(z));
                    % pretty(simple(diff(f,x)));
                    %
                    % Let numerator of ^^^ be exp(2x) + MAGIC ...
                    %
                    MAGIC = fc_exp.*((fc_raw+1).*bsxfun(@minus,fc_sum,fc_exp));
                    dFdE = bsxfun(@rdivide, (fc_exp.^2 + MAGIC), fc_sum.^2);
                    dLdF = dL(:,c);
                    dLdE = bsxfun(@times, dFdE, dLdF);
                    dLdW(c,:,:) = (dLdE' * X) ./ size(X,1);
                end 
            end
            return
        end
        
        function [ L dL ] = loss_mclr(Yh, Y)
            % Compute a multiclass logistic regression loss and its gradients,
            % w.r.t. the proposed outputs Yh, given the true values Y.
            %
            cl_count = size(Y,2);
            obs_count = size(Yh,1);
            [Y_max Y_idx] = max(Y,[],2);
            P = bsxfun(@rdivide, exp(Yh), sum(exp(Yh),2));
            % Compute classification loss (deviance)
            p_idx = sub2ind(size(P), (1:obs_count)', Y_idx);
            L = -log(P(p_idx));
            if (nargout > 1)
                % Make a binary class indicator matrix
                Yi = bsxfun(@eq, Y_idx, 1:cl_count);
                % Compute the gradient of classification loss
                dL = P - Yi;
            end
            return
        end

        function [ L dL ] = loss_mcl1h(Yh, Y)
            % Compute a multiclass L2 hinge loss and its gradients, w.r.t. the
            % proposed outputs Yh, given the true values Y.
            %
            cl_count = size(Y,2);
            [Y_max Y_idx] = max(Y,[],2);
            % Make a class indicator matrix using +1/-1
            Yc = bsxfun(@(y1,y2) (2*(y1==y2))-1, Y_idx, 1:cl_count);
            % Compute current L2 hinge loss given the predictions in Yh
            margin_lapse = max(0, 1 - (Yc .* Yh));
            L = margin_lapse;
            if (nargout > 1)
                % For L2 hinge loss, dL is equal to the margin intrusion
                dL = -Yc .* (margin_lapse > 0);
            end
            return
        end
        
        function [ L dL ] = loss_mcl2h(Yh, Y)
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
            if (nargout > 1)
                % For L2 hinge loss, dL is equal to the margin intrusion
                dL = -Yc .* margin_lapse;
            end
            return
        end
        
        function [ L dL ] = loss_hinge(Yh, Y)
            % Compute a multiclass L2 hinge loss and its gradients, w.r.t. the
            % proposed outputs Yh, given the true values Y.
            %
            cl_count = size(Y,2);
            [Y_max Y_idx] = max(Y,[],2);
            % Make a mask based on class membership
            c_mask = zeros(size(Y));
            for c=1:cl_count,
                c_mask(Y_idx==c,c) = 1;
            end
            Fc = Yh;
            Fp = sum(Yh .* c_mask, 2);
            margin_trans = max(bsxfun(@minus, Fc, Fp) + 1, 0);
            margin_trans = margin_trans .* (1 - c_mask);
            L = sum(margin_trans,2);
            if (nargout > 1)
                dL = double(margin_trans > 0);
                for c=1:cl_count,
                    not_c = setdiff(1:cl_count,c);
                    dL(Y_idx==c,c) = -sum(dL(Y_idx==c,not_c),2);
                end
            end
            return
        end
        
        function [ L dL ] = loss_lsq(Yh, Y)
            % Compute a least-sqaures regression loss and its gradients, for
            % each of the predicted outputs in Yh with true values Y.
            %
            R = Yh - Y;
            L = 0.5 * R.^2;
            if (nargout > 1)
                % Compute the gradient of least-squares loss
                dL = R;
            end
            return
        end
        
    end 
    
end










%%%%%%%%%%%%%%
% EYE BUFFER %
%%%%%%%%%%%%%%

