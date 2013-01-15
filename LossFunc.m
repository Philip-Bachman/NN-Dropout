classdef LossFunc < handle
    % This is a class for managing loss functions to be used at the output layer
    % of a multi-layer neural-net.
    %
    
    properties
        % func_type determines which loss function to use
        func_type
    end
    
    methods
        function [self] = LossFunc( func_type )
            % Constructor for LossFunc class
            if ~exist('func_type','var')
                func_type = 1;
            end
            self.func_type = func_type;
            return
        end
        
        function [ L dL ] = evaluate(self, Yh, Y)
            % Compute feed-forward activations according to some function
            if (size(Yh) ~= size(Y))
                error('LossFunc: estimate and target vector size mismatch.');
            end
            switch self.func_type
                case 1
                    [L dL] = LossFunc.least_squares(Yh, Y);
                case 2
                    [L dL] = LossFunc.cross_entropy(Yh, Y);
                case 3
                    [L dL] = LossFunc.binomial_deviance(Yh, Y);
                case 4
                    [L dL] = LossFunc.svm_hinge(Yh, Y);
                otherwise
                    error('No valid loss function type selected.');
            end
            return
        end
    end
    
    methods (Static = true)
        % The static methods for LossFunc compute loss values and gradients for
        % various loss functions.
        %
        function [ L dL ] = least_squares(Yh, Y)
            % Compute loss and loss gradient for least-squares error
            L = (1 / 2) * (Yh - Y).^2;
            dL = Yh - Y;
            return
        end
        
        function [ L dL ] = cross_entropy(Yh, Y)
            % Compute loss and loss gradient for cross-entropy error
            Yf = Y(:);
            Yhf = Yh(:);
            if ((sum(Yf == 1) + sum(Yf == 0)) < numel(Yf))
                error('LossFunc.cross_entropy: values in Y 0/1.');
            end
            if ((sum(Yhf < 0) + sum(Yhf > 1)) > 0)
                error('LossFunc.cross_entropy: Yh rows must be distributions.');
            end
            L = -Y .* log(Yh);
            dL = -Y ./ Yh;
            return
        end
        
        function [ L dL ] = binomial_deviance(Yh, Y)
            % Compute loss and loss gradient for binomial-deviance error
            Yf = Y(:);
            if ((sum(Yf == 1) + sum(Yf == -1)) < numel(Yf))
                error('LossFunc.binomial_deviance: Y values should be +/- 1.');
            end
            if (size(Y,2) == 1)
                error('LossFunc.binomial_deviance: classes need own outputs.');
            end
            lam = 1e-5;
            obs_count = size(Y,1);
            % Get the true class index for each observation and the value
            % predicted for this class
            [c_val c_idx] = max(Y,[],2);
            true_idx = sub2ind(size(Y), (1:obs_count)', c_idx);
            true_val = reshape(Yh(true_idx),obs_count,1);
            % Compute error for each prediction
            err = bsxfun(@minus, Yh, true_val);
            % Bound error range to avoid exp() overflowing
            err = min(max(err, -10), 10);
            % Use error to compute binomial-deviance loss
            L = log(exp(err) + 1);
            dL = exp(err) ./ (exp(err) + 1);
            L(true_idx) = 0;
            dL(true_idx) = 0;
            L(true_idx) = sum(L,2);
            dL(true_idx) = -sum(dL,2);
            L = L + ((lam/2) .* (Yh.^2));
            dL = dL + (lam .* Yh);
            return
        end
        
        function [ L dL ] = svm_hinge(Yh, Y)
            % Compute loss and loss gradient for svm-like hinge loss
            Yf = Y(:);
            if ((sum(Yf == 1) + sum(Yf == -1)) < numel(Yf))
                error('LossFunc.svm_hinge: Y values should be +/- 1.');
            end
            if (size(Y,2) == 1)
                error('LossFunc.svm_hinge: each class needs an output.');
            end
            lam = 1e-5;
            obs_count = size(Y,1);
            % Get the true class index for each observation and the value
            % predicted for this class
            [c_val c_idx] = max(Y,[],2);
            true_idx = sub2ind(size(Y), (1:obs_count)', c_idx);
            true_val = reshape(Yh(true_idx),obs_count,1);
            % Compute error for each prediction
            err = bsxfun(@minus, Yh, true_val);
            % Use error to compute hinge loss
            L = max(err + 1, 0);
            dL = L > 0;
            L(true_idx) = 0;
            dL(true_idx) = 0;
            L(true_idx) = sum(L,2);
            dL(true_idx) = -sum(dL,2);
            L = L + ((lam/2) .* (Yh.^2));
            dL = dL + (lam .* Yh);
            return
        end
    end
    
end






%%%%%%%%%%%%%%
% EYE BUFFER %
%%%%%%%%%%%%%%

