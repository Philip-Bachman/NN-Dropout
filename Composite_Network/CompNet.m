classdef CompNet < handle
    % This class controls a compound neural-net comprising multiple BaseNets.
    %
    % For now, this will be constrained to a "Siamese Network" architecture,
    % comprising a feature encoding network which feeds into two networks: a
    % classification network and a decoder network.
    %
    
    properties
        % feat_net is a handle for the BaseNet instance which acts as a feature
        % encoder for this CompNet.
        feat_net
        % class_net is a handle for the BaseNet instance which acts as a
        % classification network for this CompNet
        class_net
        % decode_net is a handle for the BaseNet instance which acts as a
        % nonlinear "decoder" for this CompNet
        decode_net
        % lam_class gives relative weight of grads on class_net
        lam_class
        % lam_decode gives relative weight of grads on decode_net
        lam_decode
    end
    
    methods
        function [self] = CompNet(X, Y, feat_net, class_net, decode_net)
            % Constructor for CompNet class
            self.feat_net = feat_net;
            self.class_net = class_net;
            self.decode_net = decode_net;
            % Check size compatability to make sure the BaseNets form a valid
            % dual-headed network architecture.
            if (feat_net.layer_nsizes(1) ~= size(X,2))
                error('Wrong size at feat_net input layer.');
            end
            if (feat_net.layer_nsizes(end) ~= class_net.layer_nsizes(1))
                error('Mismatch between feat_net/class_net out/in layers.');
            end
            if (feat_net.layer_nsizes(end) ~= decode_net.layer_nsizes(1))
                error('Mismatch between feat_net/decode_net out/in layers.');
            end
            if (class_net.layer_nsizes(end) ~= size(Y,2))
                error('Wrong size at class_net output layer.');
            end
            if (decode_net.layer_nsizes(end) ~= size(X,2))
                error('Wrong size at decode_net output layer.');
            end
            % Set relative weights of gradients (note: feat_net fixed to 1)
            self.lam_class = 1.0;
            self.lam_decode = 1.0;
            return
        end
        
        function [ result ] = init_weights(self, weight_scale)
            % Initialize the connection weights for this CompNet.
            %
            if ~exist('weight_scale','var')
                weight_scale = 0.1;
            end
            self.feat_net.init_weights(weight_scale);
            self.class_net.init_weights(weight_scale);
            self.decode_net.init_weights(weight_scale);
            result = weight_scale;
            return
        end
        
        function [ result ] = set_drop_rates(self, drop_rate)
            % Set drop rates in the underlying BaseNet networks.
            %
            self.feat_net.drop_rate = drop_rate;
            self.class_net.drop_rate = drop_rate;
            self.class_net.drop_input = drop_rate;
            self.decode_net.drop_rate = drop_rate;
            self.decode_net.drop_input = drop_rate;
            result = drop_rate;
            return
        end
        
        function [ w_feat w_class w_decode ] = get_drop_weights(self, do_noise)
            % Get droppy/fuzzy weights for the BaseNets in this CompNet.
            if ~exist('do_noise','var')
                do_noise = 0;
            end
            w_feat = self.feat_net.get_drop_weights(do_noise);
            w_class = self.class_net.get_drop_weights(do_noise);
            w_decode = self.decode_net.get_drop_weights(do_noise);
            return
        end
        
        function [ a_feat a_class a_decode ] = evaluate(self, X)
            % Do a simple feed-forward computation for the inputs in X. This
            % computation performs neither dropout nor weight fuzzing. For some
            % drop rates, it approximates ensemble averaging using Hinton's
            % suggestion of weight-halving.
            %
            % Note: the "feedforward(X, weights)" function can be used to
            %       evaluate this network with droppy/fuzzy weights.
            a_feat = self.feat_net.evaluate(X);
            a_class = self.class_net.evaluate(a_feat);
            a_decode = self.decode_net.evaluate(a_feat);
            return
        end
        
        function [ a_feat a_class a_decode ] = ...
                feedforward(self, X, w_feat, w_class, w_decode)
            % Get per-layer activations for the observations in X, given the
            % weights in w_feat, w_class, and w_decode.
            %
            a_feat = self.feat_net.feedforward(X, w_feat);
            a_class = self.class_net.feedforward(a_feat{end}, w_class);
            a_decode = self.decode_net.feedforward(a_feat{end}, w_decode);
            return
        end
        
        function [ dW dN ] = backprop(self, a_feat, a_class, a_decode, ...
                w_feat, w_class, w_decode, g_feat, g_class, g_decode)
            % Get per-layer weight gradients, based on the given per-layer
            % activations, per-layer weights, and loss gradients at the output
            % layer (i.e., perform backprop). SUPER IMPORTANT FUNCTION!!
            %
            % Parameters:
            %   a_---: per-layer post-transform activations for ---
            %   w_---: inter-layer weights w.r.t. which to gradient for ---
            %   g_---: per layer grads on post-transform activations for ---
            %            note: for backpropping basic loss on net output, only
            %                  g_class{end} and/or g_decode{end} will ~= 0.
            %
            % Outputs:
            %   dW: grad on each inter-layer weight (size of l_weights)
            %   dN: grad on each pre-transform activation (size of l_grads)
            %       note: dN{1} will be grads on inputs to network
            %
            [dW_class dN_class] = ...
                self.class_net.backprop(a_class, w_class, g_class);
            [dW_decode dN_decode] = ...
                self.decode_net.backprop(a_decode, w_decode, g_decode);
            % Add the gradients coming from losses on the classification and
            % decoding networks onto the intrisic gradient on the feature net.
            g_feat{end} = g_feat{end} + (self.lam_class * dN_class{1});
            g_feat{end} = g_feat{end} + (self.lam_decode * dN_decode{1});
            % Backprop per-layer gradients through the feature net.
            [dW_feat dN_feat] = ...
                self.feat_net.backprop(a_feat, w_feat, g_feat);
            % Pack structs with the results of BP
            dW = struct();
            dW.class = dW_class;
            dW.decode = dW_decode;
            dW.feat = dW_feat;
            dN = struct();
            dN.class = dW_class;
            dN.decode = dN_decode;
            dN.feat = dN_feat;
            return
        end
        
        function [ g_feat g_class g_decode ] = ...
                int_grads(self, a_feat, a_class, a_decode)
            % Compute gradients intrinsic to the activations of each BaseNet
            % underlying this CompNet (i.e. _not_ classification/encoding).
            %
            g_feat = cell(1,self.feat_net.depth);
            for i=1:self.feat_net.depth,
                g_feat{i} = zeros(size(a_feat{i}));
            end
            g_class = cell(1,self.class_net.depth);
            for i=1:self.class_net.depth,
                g_class{i} = zeros(size(a_class{i}));
            end
            g_decode = cell(1,self.decode_net.depth);
            for i=1:self.decode_net.depth,
                g_decode{i} = zeros(size(a_decode{i}));
            end
            return
        end
        
        function [ g_feat ] = ext_grads_feat(self, X, Y, a_feat, g_feat)
            %
            g_feat{end} = g_feat{end} + 0;
            return
        end
        
        function [ g_class ] = ext_grads_class(self, Y, a_class, g_class)
            %
            [L dL] = CompNet.loss_mclr(a_class{end}, Y);
            g_class{end} = g_class{end} + dL;
            return
        end
        
        function [ g_decode ] = ext_grads_decode(self, X, a_decode, g_decode)
            %
            [L dL] = CompNet.loss_lsq(a_decode{end}, X);
            g_decode{end} = g_decode{end} + dL;
            return
        end
        
        function [ result ] = ...
                update_weights(self, dW_feat, dW_class, dW_decode, step)
            % Update weights in each of the underlying networks using the given
            % cell arrays of updates to each network's inter-layer weights.
            %
            for i=1:(self.feat_net.depth-1),
                Wi = self.feat_net.layer_weights{i};
                dWi = dW_feat{i};
                self.feat_net.layer_weights{i} = Wi - (step * dWi);
            end
            for i=1:(self.class_net.depth-1),
                Wi = self.class_net.layer_weights{i};
                dWi = dW_class{i};
                self.class_net.layer_weights{i} = Wi - (step * dWi);
            end
            for i=1:(self.decode_net.depth-1),
                Wi = self.decode_net.layer_weights{i};
                dWi = dW_decode{i};
                self.decode_net.layer_weights{i} = Wi - (step * dWi);
            end
            result = 1;
            return
        end
        
        function [ result ] =  train(self, X, Y, rounds)
            % Train all layers of this CompNet.
            %
            obs_count = size(X,1);
            for i=1:rounds,
                idx = randsample(obs_count, 200);
                Xb = X(idx,:);
                Yb = Y(idx,:);
                % Get weights from each underlying BaseNet
                [w_feat w_class w_decode] = self.get_drop_weights(1);
                % Get activations using these weights
                [a_feat a_class a_decode] = ...
                    self.feedforward(Xb, w_feat, w_class, w_decode);
                % Compute intrinsic gradients for these activations
                [g_feat g_class g_decode] = ...
                    self.int_grads(a_feat, a_class, a_decode);
                % Add extrinsic gradients for activations on class_net
                g_class = self.ext_grads_class(Yb, a_class, g_class);
                % Add extrinsic gradients for activations on decode_net
                g_decode = self.ext_grads_decode(Xb, a_decode, g_decode);
                % Add extrinsic gradients for activations on feat_net
                g_feat = self.ext_grads_feat(Xb, Yb, a_feat, g_feat);
                % Backprop the combined intrinsic/extrinsic gradients.
                %
                % Note: backprop does reweighting of gradients coming from the
                % classification and decoding networks into the feature network
                % using weights self.lam_class and self.lam_decode.
                %
                dW = self.backprop(a_feat, a_class, a_decode, w_feat,...
                    w_class, w_decode, g_feat, g_class, g_decode);
                % Update weights in each network
                self.update_weights(dW.feat, dW.class, dW.decode, 0.01);
                % Check loss every once in a while
                if ((i == 1) || (mod(i, 25) == 0))
                    idx = randsample(obs_count, 2000);
                    Xv = X(idx,:);
                    Yv = Y(idx,:);
                    [a_feat a_class a_decode] = self.evaluate(Xv);
                    Lc = CompNet.loss_mclr(a_class,Yv);
                    [xxx Yi] = max(Yv,[],2);
                    [xxx Yhi] = max(a_class,[],2);
                    Ac = sum(Yhi == Yi) / numel(Yi);
                    Ld = CompNet.loss_lsq(a_decode,Xv);
                    fprintf('Round %d: tr=(%.4f, %.4f, %.4f)\n',...
                        i,mean(Lc(:)),Ac,mean(Ld(:)));
                end
            end
            
            % SHTOOOOOOOOOF!
            result = struct();
            
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
        
        function [ L dLl dLr ] = loss_grads(Xl, Xr, Fl, Fr)
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
        
        function [ params ] = process_params(params)
            % Process parameters to use in training of some sort.
            if ~isfield(params, 'rounds')
                params.rounds = 10000;
            end
            if ~isfield(params, 'start_rate')
                params.start_rate = 1.0;
            end
            if ~isfield(params, 'decay_rate')
                params.decay_rate = 0.995;
            end
            if ~isfield(params, 'momentum')
                params.momentum = 0.5;
            end
            if ~isfield(params, 'lam_l2')
                params.lam_l2 = 1e-5;
            end
            if ~isfield(params, 'lam_grad')
                params.lam_grad = 0;
            end
            if ~isfield(params, 'lam_hess')
                params.lam_hess = 0;
            end
            if ~isfield(params, 'batch_size')
                params.batch_size = 100;
            end
            if ~isfield(params, 'do_validate')
                params.do_validate = 0;
            end
            if (params.do_validate == 1)
                if (~isfield(params, 'Xv') || ~isfield(params, 'Yv'))
                    error('Validation set required for doing validation.');
                end
            end
            params.momentum = min(1, max(0, params.momentum));
            return
        end
        
    end 
    
end










%%%%%%%%%%%%%%
% EYE BUFFER %
%%%%%%%%%%%%%%

