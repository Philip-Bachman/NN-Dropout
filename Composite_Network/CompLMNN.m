classdef CompLMNN < handle
    % This class controls a compound neural-net comprising multiple BaseNets.
    %
    % For now, this will be constrained to a "Siamese Network" architecture,
    % comprising a feature encoding network which feeds into two networks: a
    % classification network and a decoder network.
    %
    
    properties
        % feat_net is a handle for the BaseNet instance which acts as a feature
        % encoder for this CompLMNN.
        feat_net
        % decode_net is a handle for the BaseNet instance which acts as a
        % nonlinear "decoder" for this CompLMNN
        decode_net
        % lam_lmnn gives relative weight of grads on class_net
        lam_lmnn
        % lam_decode gives relative weight of grads on decode_net
        lam_decode
        % lam_fgrad and lam_dgrad give gradient reg weights on feat/deco net
        lam_fgrad
        lam_dgrad
        % lam_fhess and lam_dhess give Hessian reg weights on feat/deco net
        lam_fhess
        lam_dhess
        
    end
    
    methods
        function [self] = CompLMNN(X, feat_net, decode_net)
            % Constructor for CompLMNN class
            self.feat_net = feat_net;
            self.decode_net = decode_net;
            % Check size compatability to make sure the BaseNets form a valid
            % dual-headed network architecture.
            if (feat_net.layer_nsizes(1) ~= size(X,2))
                error('Wrong size at feat_net input layer.');
            end
            if (feat_net.layer_nsizes(end) ~= decode_net.layer_nsizes(1))
                error('Mismatch between feat_net/decode_net out/in layers.');
            end
            if (decode_net.layer_nsizes(end) ~= size(X,2))
                error('Wrong size at decode_net output layer.');
            end
            % Set relative weights of obj gradients (note: feat_net fixed to 1)
            self.lam_lmnn = 1.0;
            self.lam_decode = 1.0;
            % Set weights for functional regularization
            self.lam_fgrad = 0.0;
            self.lam_dgrad = 0.0;
            self.lam_fhess = 0.0;
            self.lam_dhess = 0.0;
            return
        end
        
        function [ result ] = init_weights(self, weight_scale)
            % Initialize the connection weights for this CompLMNN.
            %
            if ~exist('weight_scale','var')
                weight_scale = 0.1;
            end
            self.feat_net.init_weights(weight_scale);
            self.decode_net.init_weights(weight_scale);
            result = weight_scale;
            return
        end
        
        function [ result ] = set_drop_rates(self, drop_rate)
            % Set drop rates in the underlying BaseNet networks.
            %
            self.feat_net.drop_rate = drop_rate;
            self.decode_net.drop_rate = drop_rate;
            self.decode_net.drop_input = drop_rate;
            result = drop_rate;
            return
        end
        
        function [ w_feat w_decode ] = get_drop_weights(self, do_noise)
            % Get droppy/fuzzy weights for the BaseNets in this CompLMNN.
            %
            if ~exist('do_noise','var')
                do_noise = 0;
            end
            w_feat = self.feat_net.get_drop_weights(do_noise);
            w_decode = self.decode_net.get_drop_weights(do_noise);
            return
        end
        
        function [ a_feat a_decode ] = evaluate(self, X)
            % Do a simple feed-forward computation for the inputs in X. This
            % computation performs neither dropout nor weight fuzzing. For some
            % drop rates, it approximates ensemble averaging using Hinton's
            % suggestion of weight-halving.
            %
            % Note: the "feedforward(X, weights)" function can be used to
            %       evaluate this network with droppy/fuzzy weights.
            a_feat = self.feat_net.evaluate(X);
            a_decode = self.decode_net.evaluate(a_feat);
            return
        end
        
        function [ a_feat a_decode ] = feedforward(self, X, w_feat, w_decode)
            % Get full per-layer activations for the points in X, given the
            % weights in w_feat and w_decode.
            %
            a_feat = self.feat_net.feedforward(X, w_feat);
            a_decode = self.decode_net.feedforward(a_feat{end}, w_decode);
            return
        end
        
        function [ dW_feat dN_feat ] = bp_lmnn(self, a_feat, w_feat, g_lmnn)
            % Get per-layer weight gradients, based on the given per-layer
            % activations, per-layer weights, and loss gradients at the output
            % layer (i.e., perform backprop). SUPER IMPORTANT FUNCTION!!
            %
            % Parameters:
            %   a_---: per-layer post-transform activations for ---
            %   w_---: inter-layer weights w.r.t. which to gradient for ---
            %   g_---: per layer grads on post-transform activations for ---
            %
            % Outputs:
            %   dW: grad on each inter-layer weight (size of l_weights)
            %   dN: grad on each pre-transform activation (size of l_grads)
            %       note: dN{1} will be grads on inputs to network
            %
            g_feat = cell(size(a_feat));
            for i=1:length(a_feat),
                g_feat{i} = zeros(size(a_feat{i}));
            end
            % Inject gradients coming from the LMNN objective and backprop
            % through the feature transform net.
            g_feat{end} = g_lmnn;
            [dW_feat dN_feat] = self.feat_net.backprop(a_feat, w_feat, g_feat);
            return
        end
        
        function [ dW dN ] = bp_decode(self, a_feat, a_decode, w_feat, ...
                w_decode, g_decode)
            % Get per-layer weight gradients, based on the given per-layer
            % activations, per-layer weights, and loss gradients at the output
            % layer (i.e., perform backprop). SUPER IMPORTANT FUNCTION!!
            %
            % Parameters:
            %   a_---: per-layer post-transform activations for ---
            %   w_---: inter-layer weights w.r.t. which to gradient for ---
            %   g_---: per layer grads on post-transform activations for ---
            %
            % Outputs:
            %   dW: grad on each inter-layer weight (size of l_weights)
            %   dN: grad on each pre-transform activation (size of l_grads)
            %       note: dN{1} will be grads on inputs to network
            %
            [dW_decode dN_decode] = ...
                self.decode_net.backprop(a_decode, w_decode, g_decode);
            % Create "dummy" gradient package for the feature net.
            g_feat = cell(size(a_feat));
            for i=1:length(a_feat),
                g_feat{i} = zeros(size(a_feat{i}));
            end
            % Inject gradients coming from the decoder and backprop through the
            % feature transform net.
            g_feat{end} = dN_decode{1};
            [dW_feat dN_feat] = ...
                self.feat_net.backprop(a_feat, w_feat, g_feat);
            % Pack structs with the results of BP
            dW = struct();
            dW.feat = dW_feat;
            dW.decode = dW_decode;
            dN = struct();
            dN.feat = dN_feat;
            dN.decode = dN_decode;
            return
        end
        
        function [ dW dN L ] = bp_grad_feat(self, Xl, Xr, al_feat, ar_feat,...
                w_feat, smpl_wts)
            % Compute and backprop gradients from grad-norm regularization on
            % the feature transform network.
            %
            [dW dN L] = self.feat_net.bprop_grad_jnt(...
                w_feat, al_feat, ar_feat, Xl, Xr, smpl_wts);
            return
        end
        
        function [ g_feat g_decode ] = int_grads(self, a_feat, a_decode)
            % Compute gradients intrinsic to the activations of each BaseNet
            % underlying this CompLMNN (i.e. _not_ classify/encode grads).
            %
            g_feat = cell(1,self.feat_net.depth);
            for i=1:self.feat_net.depth,
                g_feat{i} = 1e-5 * a_feat{i};
            end
            g_decode = cell(1,self.decode_net.depth);
            for i=1:self.decode_net.depth,
                g_decode{i} = zeros(size(a_decode{i}));
            end
            return
        end
        
        function [ g_decode ] = loss_grads_decode(self, X, a_decode, g_decode)
            % Compute the loss and gradients for the decoding objective
            [L dL] = CompLMNN.loss_lsq(a_decode{end}, X);
            g_decode{end} = g_decode{end} + dL;
            return
        end
        
        function [ mW ] = update_weights(self, dW, mW, m, s)
            % Update weights in each of the underlying networks using the given
            % cell arrays of updates to each network's inter-layer weights.
            %
            for i=1:(self.feat_net.depth-1),
                Wi = self.feat_net.layer_weights{i};
                mWi = mW.feat{i};
                dWi = dW.feat{i};
                dWi = (m * mWi) + ((1 - m) * dWi);
                mW.feat{i} = dWi;
                self.feat_net.layer_weights{i} = Wi - (s * dWi);
            end
            for i=1:(self.decode_net.depth-1),
                Wi = self.decode_net.layer_weights{i};
                mWi = mW.decode{i};
                dWi = dW.decode{i};
                dWi = (m * mWi) + ((1 - m) * dWi);
                mW.decode{i} = dWi;
                self.decode_net.layer_weights{i} = Wi - (s * dWi);
            end
            return
        end

        function [ result ] =  train_decode(self, X, Y, rounds, step)
            % Train this CompLMNN on the decoding objective.
            %
            obs_count = size(X,1);
            % Initialize momentum
            mW = struct();
            mW.feat = cell(1,length(self.feat_net.layer_weights));
            for i=1:length(mW.feat),
                mW.feat{i} = zeros(size(self.feat_net.layer_weights{i}));
            end
            mW.decode = cell(1,length(self.decode_net.layer_weights));
            for i=1:length(mW.decode),
                mW.decode{i} = zeros(size(self.decode_net.layer_weights{i}));
            end
            % Compute a length scale for gradient regularization. Sample random
            % pairs of observations and set the length to a function of some
            % approximate quantile of the pairwise inter-observation distances.
            dists = zeros(10000,1);
            for i=1:10000,
                x1 = X(randi(obs_count),:);
                x2 = X(randi(obs_count),:);
                dists(i) = sqrt(sum((x1 - x2).^2));
            end
            grad_len = (quantile(dists,0.01) / 2);
            for i=1:rounds,
                %%%%%%%%%%%%%%%%%%%%
                % DECODER TRAINING %
                %%%%%%%%%%%%%%%%%%%%
                [Xl Xc Xr Yc smpl_wts] = ...
                    CompLMNN.sample_points(X, Y, 250, grad_len);
                % Get weights from each underlying BaseNet
                [w_feat w_decode] = self.get_drop_weights(1);
                % Get activations for FD estimation points
                [ac_feat ac_decode] = self.feedforward(Xc,w_feat,w_decode);
                [al_feat al_decode] = self.feedforward(Xl,w_feat,w_decode);
                [ar_feat ar_decode] = self.feedforward(Xr,w_feat,w_decode);
                % Compute intrinsic gradients for these activations
                [g_feat g_decode] = self.int_grads(ac_feat, ac_decode);
                % Add extrinsic gradients for activations on decode_net
                g_decode = self.loss_grads_decode(Xc, ac_decode, g_decode);
                % Backprop the combined intrinsic/extrinsic gradients.
                dW = self.bp_decode(ac_feat,ac_decode,w_feat,w_decode,g_decode);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % FEATURE/DECODER REGULARIZATION %
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Get regularization grads on the feature transform
                if (self.lam_fgrad > 1e-8)
                    [dWf dNf Lf] = self.bp_grad_feat(Xl, Xr, al_feat, ...
                        ar_feat, w_feat, smpl_wts);
                end
%                 else
%                     error('UNDER CONSTRUCTION!\n');
%                 end
%                 if (self.lam_fhess > 1e-8)
%                     error('UNDER CONSTRUCTION!\n');
%                 else
%                     error('UNDER CONSTRUCTION!\n');
%                 end
                %%%%%%%%%%%%%%%%%%
                % JOINT TRAINING %
                %%%%%%%%%%%%%%%%%%
                for j=1:length(dW.feat),
                    dW.feat{j} = (self.lam_decode * dW.feat{j}) +...
                        (self.lam_fgrad * dWf{j});
                end
                for j=1:length(dW.decode),
                    dW.decode{j} = (self.lam_decode * dW.decode{j});
                end
                % Update weights in each network
                if (i == 1)
                    mW = self.update_weights(dW, mW, 0.0, step);
                else
                    mW = self.update_weights(dW, mW, 0.75, step);
                end
                % Check loss every once in a while
                if ((i == 1) || (mod(i, 50) == 0))
                    idx = randsample(obs_count, 2000);
                    Xv = X(idx,:);
                    [a_feat a_decode] = self.evaluate(Xv);
                    Ld = CompLMNN.loss_lsq(a_decode,Xv);
                    fprintf('Round %d: tr=(%.4f, %.4f)\n', i, ...
                        mean(Ld(:)), mean(Lf(:)));
                end
            end
            % SHTOOOOOOOOOF!
            result = struct();
            return
        end
        
        function [ result ] =  train_lmnn(self, X, Y, rounds, step)
            % Train all layers of this CompLMNN.
            %
            obs_count = size(X,1);
            % Initialize weight update auxiliary structures
            dW = struct();
            mW = struct();
            dW.feat = cell(1,length(self.feat_net.layer_weights));
            mW.feat = cell(1,length(self.feat_net.layer_weights));
            for i=1:length(dW.feat),
                dW.feat{i} = zeros(size(self.feat_net.layer_weights{i}));
                mW.feat{i} = zeros(size(self.feat_net.layer_weights{i}));
            end
            dW.decode = cell(1,length(self.decode_net.layer_weights));
            mW.decode = cell(1,length(self.decode_net.layer_weights));
            for i=1:length(dW.decode),
                dW.decode{i} = zeros(size(self.decode_net.layer_weights{i}));
                mW.decode{i} = zeros(size(self.decode_net.layer_weights{i}));
            end
            % Compute a length scale for gradient regularization. Sample random
            % pairs of observations and set the length to a function of some
            % approximate quantile of the pairwise inter-observation distances.
            dists = zeros(10000,1);
            for i=1:10000,
                x1 = X(randi(obs_count),:);
                x2 = X(randi(obs_count),:);
                dists(i) = sqrt(sum((x1 - x2).^2));
            end
            grad_len = (quantile(dists,0.01) / 2);
            %%%%%%%%%%%%%%%%%%%%
            % TEST INITIAL KNN %
            %%%%%%%%%%%%%%%%%%%%
            a_feat = self.evaluate(X);
            Y_c = class_cats(Y);
            Y_nn = knn(a_feat, a_feat, Y_c, 10, 1, 0);
            acc = sum(bsxfun(@eq,Y_nn,Y_c)) / numel(Y_c);
            fprintf('Start acc: (%.4f, %.4f, %.4f)\n',acc(1),acc(2),acc(3));
            for i=1:rounds,
                %%%%%%%%%%%%%%%%%
                % LMNN TRAINING %
                %%%%%%%%%%%%%%%%%
                if ((i == 1) || (mod(i,250) == 0))
                    % Recompute neighbor constraints from current embedding
                    Xf = self.evaluate(X);
                    nc_lmnn = CompLMNN.neighbor_constraints(Xf,Y,8,16,0);
                    clear('Xf');
                    nc_count = size(nc_lmnn,1);
                end
                nc_batch = nc_lmnn(randsample(nc_count,500,false),:);
                Xc = X(nc_batch(:,1),:);
                Xn = X(nc_batch(:,2),:);
                Xf = X(nc_batch(:,3),:);
                % Get weights from each underlying BaseNet
                [w_feat w_decode] = self.get_drop_weights(1);
                % Compute activations for center/near/far points
                ac_feat = self.feedforward(Xc,w_feat,w_decode);
                an_feat = self.feedforward(Xn,w_feat,w_decode);
                af_feat = self.feedforward(Xf,w_feat,w_decode);
                % Compute LMNN loss and gradients on these activations
                [L_lmnn dXc dXn dXf] = CompLMNN.lmnn_grads_euc(...
                    ac_feat{end}, an_feat{end}, af_feat{end}, 1.0);
                dWc = self.bp_lmnn(ac_feat, w_feat, dXc);
                dWn = self.bp_lmnn(an_feat, w_feat, dXn);
                dWf = self.bp_lmnn(af_feat, w_feat, dXf);
                %[Ls steps] = self.test_lmnn_grads(Xb_c, Xb_n, Xb_f);
                %plot(steps,Ls); drawnow(); pause(0.1);
                dW_lmnn = cell(1,length(dWc));
                for j=1:length(dW_lmnn),
                    dW_lmnn{j} = (dWc{j} + dWn{j} + dWf{j});
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%
                % REGULARIZATION TRAINING %
                %%%%%%%%%%%%%%%%%%%%%%%%%%%
                [Xl Xc Xr Yc smpl_wts] = ...
                    CompLMNN.sample_points(X, Y, 250, grad_len);
                % Get activations for FD estimation points
                ac_feat = self.feedforward(Xc,w_feat,w_decode);
                al_feat = self.feedforward(Xl,w_feat,w_decode);
                ar_feat = self.feedforward(Xr,w_feat,w_decode);
                % Get regularization grads on the feature transform
                if (self.lam_fgrad > 1e-8)
                    [dWf dNf Lf] = self.bp_grad_feat(Xl, Xr, al_feat, ...
                        ar_feat, w_feat, smpl_wts);
                end
%                 else
%                     error('UNDER CONSTRUCTION!\n');
%                 end
%                 if (self.lam_fhess > 1e-8)
%                     error('UNDER CONSTRUCTION!\n');
%                 else
%                     error('UNDER CONSTRUCTION!\n');
%                 end
                %%%%%%%%%%%%%%%%%%
                % JOINT TRAINING %
                %%%%%%%%%%%%%%%%%%
                for j=1:length(dW.feat),
                    dW.feat{j} = (self.lam_lmnn * dW_lmnn{j}) + ...
                        (self.lam_fgrad * dWf{j});
                end
                % Update weights in each network
                if (i == 1)
                    mW = self.update_weights(dW, mW, 0.0, step);
                else
                    mW = self.update_weights(dW, mW, 0.75, step);
                end
                % Check loss every once in a while
                if ((i == 1) || (mod(i, 50) == 0))
                    fprintf('Round %d: tr=(%.4f, %.4f)\n', i,...
                        mean(L_lmnn(:)), mean(Lf(:)));
                end
            end
            % SHTOOOOOOOOOF!
            result = struct();
            return
        end
        
        function [ step_losses steps ] = test_lmnn_grads(self, Xc, Xn, Xf)
            % Test gradients computed for LMNN objective at these points
            [w_feat w_decode] = self.get_drop_weights(1);
            % Compute activations for center/near/far points
            ac_feat = self.feedforward(Xc,w_feat,w_decode);
            an_feat = self.feedforward(Xn,w_feat,w_decode);
            af_feat = self.feedforward(Xf,w_feat,w_decode);
            % Compute LMNN loss and gradients on these activations
            [L_pre dXc dXn dXf] = CompLMNN.lmnn_grads_euc(...
                ac_feat{end}, an_feat{end}, af_feat{end}, 1.0);
            dWc = self.bp_lmnn(ac_feat, w_feat, dXc);
            dWn = self.bp_lmnn(an_feat, w_feat, dXn);
            dWf = self.bp_lmnn(af_feat, w_feat, dXf);
            % Aggregate center/near/far gradients
            dW = cell(1,length(dWc));
            for i=1:length(dWc),
                dW{i} = (dWc{i} + dWn{i} + dWf{i}) ./ 3;
            end
            % Check quality of gadient update for various step sizes
            steps = logspace(-3,1,200);
            step_losses = zeros(size(steps));
            for i=1:numel(steps),
                s = steps(i);
                wf_new = cell(1,length(w_feat));
                for j=1:length(w_feat),
                    wf_new{j} = w_feat{j} - (s * dW{j});
                end
                Ac = self.feedforward(Xc,wf_new,w_decode);
                An = self.feedforward(Xn,wf_new,w_decode);
                Af = self.feedforward(Xf,wf_new,w_decode);
                Ls = CompLMNN.lmnn_grads_euc(Ac{end}, An{end}, Af{end}, 1.0);
                step_losses(i) = mean(Ls(:));
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
        
        function [ samples ] = weighted_sample( values, sample_count, weights )
            % Do weighted sampling of the vlaues in 'value' using a probability
            % distribution determined by 'weights', without replacement.
            samples = zeros(1, sample_count);
            free_values = values;
            free_weights = weights;
            for i=1:sample_count,
                s_idx = randsample(numel(free_weights), 1, true, free_weights);
                free_weights(s_idx) = 0;
                samples(i) = free_values(s_idx);
            end
            return
        end
        
        function [ Xl Xc Xr Yc smpl_wts ] = sample_points(X, Y, smpls, grd_len)
            % Sample a batch of training observations and compute gradients
            % for gradient and hessian parts of loss.
            obs_count = size(X,1);
            idx = randsample(1:obs_count, smpls, true);
            Xc = X(idx,:);
            Yc = Y(idx,:);
            Xd = randn(size(Xc));
            Xd = bsxfun(@rdivide, Xd, sqrt(sum(Xd.^2,2))+1e-8);
            Xd = Xd .* grd_len;
            Xl = Xc - (Xd ./ 2);
            Xr = Xc + (Xd ./ 2);
            smpl_wts = ones(smpls,1);
            return
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % NEURAL NET LOSS FUNCTIONS %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
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
        
        function [ L dLl dLr ] = loss_grad(Xl, Xr, Fl, Fr)
            % Compute approximate functional gradient loss for the point tuples
            % (Xl, Xr) with outputs (Fl, Fr). Loss is equal to the square of the
            % finite-differences-estimated first-order (directional)
            % derivatives, with gradients computed accordingly.
            %
            d_norms = sqrt(sum((Xl - Xr).^2,2));
            d_scale = d_norms.^2;
            f_diffs = Fl - Fr;
            % Compute squared (fd-based) directional derivatives
            L = bsxfun(@rdivide, f_diffs.^2, 2*d_scale);
            if (nargout > 1)
                % Compute gradients
                dLl = bsxfun(@rdivide, f_diffs, d_scale);
                dLr = bsxfun(@rdivide, -f_diffs, d_scale);
            end
            return
        end
        
        function [ L dLl dLc dLr ] = loss_hess(Xl, Xc, Xr, Fl, Fc, Fr)
            % Compute approximate functional gradient loss for the point tuples
            % (Xl, Xc, Xr) with outputs (Fl, Fc, Fr). Loss is equal to the
            % square of the finite-differences-estimated second-order
            % (directional) derivatives, with gradients computed accordingly.
            %
            d_norms = 0.5 * (sqrt(sum((Xl-Xc).^2,2)) + sqrt(sum((Xr-Xc).^2,2)));
            d_scale = (d_norms.^2).^2; % weirdly written, for semantics
            f_diffs = (Fl + Fr) - (2 * Fc);
            % Compute squared (fd-based) directional derivatives
            L = bsxfun(@rdivide, f_diffs.^2, 2*d_scale);
            if (nargout > 1)
                % Compute gradients
                dLl = bsxfun(@rdivide, f_diffs, d_scale);
                dLr = bsxfun(@rdivide, f_diffs, d_scale);
                dLc = bsxfun(@rdivide, -2*f_diffs, d_scale);
            end
            return
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Metric Learning Functions %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [ I_nn ] = knn_ind( Xte, Xtr, k, do_loo, do_dot )
            % Find indices of the knn for points in Xte measured with respect to
            % the points in Xtr.
            %
            % (do_loo == 1) => do "leave-one-out" knn, assuming Xtr == Xte
            % (do_dot == 1) => use max dot-products instead of min euclideans.
            %
            if ~exist('do_loo','var')
                do_loo = 0;
            end
            if ~exist('do_dot','var')
                do_dot = 0;
            end
            obs_count = size(Xte,1);
            I_nn = zeros(obs_count,k);
            fprintf('Computing knn:');
            for i=1:obs_count,
                if (mod(i,round(obs_count/50)) == 0)
                    fprintf('.');
                end
                if (do_dot == 1)
                    d = Xtr * Xte(i,:)';
                    if (do_loo == 1)
                        d(i) = min(d) - 1;
                    end
                    [d_srt i_srt] = sort(d,'descend');
                else
                    d = sum(bsxfun(@minus,Xtr,Xte(i,:)).^2,2);
                    if (do_loo == 1)
                        d(i) = max(d) + 1;
                    end
                    [d_srt i_srt] = sort(d,'ascend');
                end
                I_nn(i,:) = i_srt(1:k);
            end
            fprintf('\n');
            return
        end
        
        function [ n_const ] = neighbor_constraints( X, Y, k_in, k_out, do_dot )
            % Generate "neighbor constraints" with which to train a metric.
            if ~exist('do_dot','var')
                do_dot = 0;
            end
            [xxx Y] = max(Y,[],2);
            I_nn = CompLMNN.knn_ind(X, X, (3*(k_in+k_out)), 1, do_dot);
            o_count = size(X,1);
            n_const = zeros(o_count*k_in*k_out,3);
            idx_c = 1;
            fprintf('Computing constraints:');
            for i=1:size(X,1),
                if (mod(i,floor(o_count/50)) == 0)
                    fprintf('.');
                end
                % Get indices of in-class and out-class nns for X(i,:)
                knn_i = I_nn(i,:);
                idx_i = knn_i(Y(knn_i) == Y(i));
                idx_o = knn_i(Y(knn_i) ~= Y(i));
                % Use the in/out nn indices to compute neighbor constraints
                for j_i=1:min(k_in,numel(idx_i)),
                    for j_o=1:min(k_out,numel(idx_o)),
                        n_const(idx_c,1) = i;
                        n_const(idx_c,2) = idx_i(j_i);
                        n_const(idx_c,3) = idx_o(j_o);
                        idx_c = idx_c + 1;
                    end
                end
            end
            n_const = n_const(1:(idx_c-1),:);
            fprintf('\n');
            return
        end

        function [ L dXc dXn dXf ] = lmnn_grads_euc( Xc, Xn, Xf, margin )
            % Compute gradients of standard LMNN using Euclidean distance.
            %
            % Parameters:
            %   Xc: central/source points
            %   Xn: points that should be closer to those in Xc
            %   Xf: points that should be further from those in Xc
            %   margin: desired margin between near/far distances w.r.t. Xc
            % Outputs:
            %   L: loss for each LMNN triplet (Xc(i,:),Xn(i,:),Xf(i,:))
            %   dXc: gradient of L w.r.t. Xc
            %   dXn: gradient of L w.r.t. Xn
            %   dXf: gradient of L w.r.t. Xf
            %
            On = Xn - Xc;
            Of = Xf - Xc;
            % Compute (squared) norms of the offsets On/Of
            Dn = 0.5 * sum(On.^2,2);
            Df = 0.5 * sum(Of.^2,2);
            % Get losses and indicators for violated LMNN constraints
            L = max(0, (Dn - Df) + margin);
            % Compute gradients for violated constraints
            dXn = bsxfun(@times, On, (L > 1e-10));
            dXf = bsxfun(@times, -Of, (L > 1e-10));
            dXc = -dXn - dXf;
            % Clip gradients
            dXc = max(-2, min(dXc, 2));
            dXn = max(-2, min(dXn, 2));
            dXf = max(-2, min(dXf, 2));
            return
        end
        
        function [ L dXc dXn dXf ] = lmnn_grads_dot( Xc, Xn, Xf, margin )
            % Compute gradients of standard LMNN using dot-product distance.
            %
            % Parameters:
            %   Xc: central/source points
            %   Xn: points that should be closer to those in Xc
            %   Xf: points that should be further from those in Xc
            %   margin: desired margin between near/far distances w.r.t. Xc
            % Outputs:
            %   L: loss for each LMNN triplet (Xc(i,:),Xn(i,:),Xf(i,:))
            %   dXc: gradient of L w.r.t. Xc
            %   dXn: gradient of L w.r.t. Xn
            %   dXf: gradient of L w.r.t. Xf
            %
            % Compute dot-products between center/near and center/far points
            Dn = sum(Xc .* Xn,2);
            Df = sum(Xc .* Xf,2);
            % Get losses and indicators for violated LMNN constraints
            m_viol = max(0, (Df - Dn) + margin);
            L = 0.5 * m_viol.^2;
            % Compute gradients for violated constraints
            dXn = bsxfun(@times, -Xc, m_viol);
            dXf = bsxfun(@times, Xc, m_viol);
            dXc = bsxfun(@times, (Xf - Xn), m_viol);
            return
        end
        
    end 
    
end










%%%%%%%%%%%%%%
% EYE BUFFER %
%%%%%%%%%%%%%%

