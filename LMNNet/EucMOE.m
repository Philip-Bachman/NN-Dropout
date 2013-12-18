classdef EucMOE < handle
    % This class manages mixtures of self-assertive (centroidy) experts.
    %
    
    properties
        % moe_pts gives the points for the experts for each class, and its
        % dimensions contain all size info for this EucMOE.
        %
        % size(moe_pts) = (class_count, moe_size, obs_dim)
        %
        moe_pts
        % lam_l2sqz moderates how experts in each class are squeezed together.
        lam_l2sqz
        % lam_l2nrm moderates the euclidean norm of each expert.
        lam_l2nrm
        % class_loss is a handle to the desired classification loss function.
        class_loss
        % k_nn tells how many points/centroids fromn each class to consider for
        % computing distance to that class.
        k_nn
    end
    
    methods
        
        function [ self ] = EucMOE(X, Y, moe_size)
            % Constructor for EucMOE class
            self.lam_l2sqz = 1e-8;
            self.lam_l2nrm = 1e-8;
            self.class_loss = @(Yh, Y) EucMOE.loss_hinge(Yh, Y);
            self.k_nn = 1;
            L = self.init_weights(X, Y, moe_size);
            fprintf('Initial loss: %.4f\n', mean(L(:)));

            return
        end
        
        function [ L ] = init_weights(self, X, Y, moe_size)
            % Initialize weights for this EucMOE, using the observations and
            % classes in X/Y.
            %
            opts = statset('Display','off');
            obs_dim = size(X,2);
            class_count = size(Y,2);
            self.moe_pts = zeros(class_count,moe_size,obs_dim);
            for c=1:class_count,
                Xc = X((Y(:,c) > 0),:);
                [idx ctrs] = kmeans(Xc,moe_size,'Distance','sqEuclidean',...
                    'emptyaction','singleton','Options',opts);
                self.moe_pts(c,:,:) = ctrs;
            end
            F = self.evaluate(X);
            L = self.class_loss(F, Y);
            return
        end
        
        function [ F ] = evaluate(self, X)
            % Compute distance between points in X and the centroids in
            % self.moe_pts. Output for each class is minus the average of the
            % self.k_nn nearest point distances.
            %
            W = self.moe_pts;
            obs_count = size(X,1);
            class_count = size(W,1);
            moe_size = size(W,2);
            F = zeros(obs_count,class_count);
            for c=1:class_count,
                Dc = zeros(obs_count,moe_size);
                for m=1:moe_size,
                    Dc(:,m) = sum(bsxfun(@minus,X,squeeze(W(c,m,:))').^2,2);
                end
                F(:,c) = -min(Dc,[],2);
            end
            return
        end           
    
        function [ result ] = train(self, X, Y, opts)
            % Use minFunc to update the weight in this EucMOE.
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
            % Get relevant dimension info from self.moe_pts
            class_count = size(self.moe_pts,1);
            moe_size = size(self.moe_pts,2);
            obs_dim = size(self.moe_pts,3);
            % Setup a loss function for use by minFunc
            loss_func = @( w ) ...
                self.loss_wrapper(w, X, Y, class_count, moe_size, obs_dim);
            % Run minFunc to compute optimal SVM parameters
            Wi = reshape(self.moe_pts, class_count*moe_size*obs_dim, 1);
            Wf = minFunc(loss_func, Wi, opts);
            self.moe_pts = reshape(Wf, class_count, moe_size, obs_dim);
            % Record result info
            result = struct();
            result.W_pre = reshape(Wi, class_count, moe_size, obs_dim);
            result.W_post = reshape(Wf, class_count, moe_size, obs_dim);
            return
        end 
        
        function [L dLdW] = loss_wrapper(self, w, X, Y, c_count, m_size, o_dim)
            % Wrapper of EucMOE classification loss for use with minFunc. This
            % also adds some, hopefully, helpful L2 regularization.
            %
            o_count = size(X,1);
            W = reshape(w, c_count, m_size ,o_dim);
            if (nargout == 1)
                L_cls = EucMOE.loss_eucmoe(X, Y, W, self.class_loss);
                L_l2nrm = 0.5 * self.lam_l2nrm * (W.^2);
                L_l2sqz = zeros(size(W));
                for c=1:c_count,
                    wc = squeeze(W(c,:,:));
                    if (size(self.moe_pts,2) == 1)
                        wc = wc';
                    end
                    c_mean = mean(wc,1);
                    wc_sqz = bsxfun(@minus, wc, c_mean);
                    L_l2sqz(c,:,:) = 0.5 * self.lam_l2sqz * (wc_sqz.^2);
                end
                L = (sum(L_cls(:)) / o_count) + ...
                    ((sum(L_l2nrm(:)) + sum(L_l2sqz(:))) / m_size);
            else
                [L_cls dLdWc] = EucMOE.loss_eucmoe(X, Y, W, self.class_loss); 
                L_l2nrm = 0.5 * self.lam_l2nrm * (W.^2);
                L_l2sqz = zeros(size(W));
                Ws = zeros(size(W));
                for c=1:c_count,
                    wc = squeeze(W(c,:,:));
                    if (size(self.moe_pts,2) == 1)
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
            W = self.moe_pts;
            L = EucMOE.loss_eucmoe(X, Y, W, self.class_loss);
            return
        end

        function [ dLdX ] = obs_grads(self, X, Y)
            % Compute maxxy or exppy grads of the loss w.r.t. the inputs X/Y.
            %
            W = self.moe_pts;
            obs_count = size(X,1);
            class_count = size(W,1);
            moe_size = size(W,2);
            Fc = zeros(obs_count,class_count);
            for c=1:class_count,
                Dc = zeros(obs_count,moe_size);
                for e=1:moe_size,
                    Dc(:,e) = sum(bsxfun(@minus,X,squeeze(W(c,e,:))').^2,2);
                end
                Fc(:,c) = -min(Dc,[],2);
            end
            [L dL] = self.class_loss(Fc, Y);
                dLdX = zeros(size(X));
                for c=1:class_count,
                    wc = squeeze(W(c,:,:));
                    if (size(W,2) == 1)
                        wc = wc';
                    end
                    % Compute distance from each expert to each observation
                    Dc = zeros(obs_count, moe_size);
                    for e=1:moe_size,
                        Dc(:,e) = sum(bsxfun(@minus,X,wc(e,:)).^2,2);
                    end
                    % Find minimum distances and their associated experts
                    [Dc_min min_idx] = min(Dc,[],2);
                    % For each expert, compute gradients arising from the times
                    % when it is closest to an observation among the experts
                    % associated with its corresponding class
                    dLdF = dL(:,c);
                    for e=1:moe_size,
                        e_idx = (min_idx == e);
                        dFdE = bsxfun(@minus,X(e_idx,:),wc(e,:));
                        dLdE = bsxfun(@times,dFdE,dLdF(e_idx));
                        dLdX(e_idx,:) = dLdX(e_idx,:) - dLdE;
                    end
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
        
        function [ L dLdW ] = loss_eucmoe(X, Y, W, loss_func)
            % Compute multiclass loss and gradients of linear mixture of
            % self-assertive experts for the observations and classes in X/Y,
            % given the MOE weights in W.
            %
            obs_count = size(X,1);
            class_count = size(W,1);
            moe_size = size(W,2);
            Fc = zeros(obs_count,class_count);
            for c=1:class_count,
                Dc = zeros(obs_count,moe_size);
                for e=1:moe_size,
                    Dc(:,e) = sum(bsxfun(@minus,X,squeeze(W(c,e,:))').^2,2);
                end
                Fc(:,c) = -min(Dc,[],2);
            end
            [L dL] = loss_func(Fc, Y);
            if (nargout > 1)
                dLdW = zeros(size(W));
                for c=1:class_count,
                    wc = squeeze(W(c,:,:));
                    if (size(W,2) == 1)
                        wc = wc';
                    end
                    % Compute distance from each expert to each observation
                    Dc = zeros(obs_count, moe_size);
                    for e=1:moe_size,
                        Dc(:,e) = sum(bsxfun(@minus,X,wc(e,:)).^2,2);
                    end
                    % Find minimum distances and their associated experts
                    [Dc_min min_idx] = min(Dc,[],2);
                    % For each expert, compute gradients arising from the times
                    % when it is closest to an observation among the experts
                    % associated with its corresponding class
                    dLdF = dL(:,c);
                    for e=1:moe_size,
                        e_idx = (min_idx == e);
                        dFdE = bsxfun(@minus,X(e_idx,:),wc(e,:));
                        dLdE = bsxfun(@times,dFdE,dLdF(e_idx));
                        dLdW(c,e,:) = sum(dLdE,1);
                    end
                end
                %dLdW = dLdW ./ size(X,1);
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
        
    end 
    
end










%%%%%%%%%%%%%%
% EYE BUFFER %
%%%%%%%%%%%%%%



