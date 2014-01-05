classdef RecSys < handle
    % Namespace for piling up miscellaneous "recommender systems" functions
    
    methods (Static = true)
        
        function [ Itr Jtr Ktr Ite Jte Kte Rtr Rte ] = ...
                trte_split(I, J, K, tr_frac)
            % Split the user/item/rating tuples (I(i),J(i),K(i)) into training
            % and testing sets, with a fraction tr_frac of each user's examples
            % going into the training set. The split is random, subject to the
            % constraint imposed by tr_frac.
            %
            % Inputs:
            %   I: vector of user ids
            %   J: vector of item ids
            %   K: vector of user ratings for items
            %   tr_frac: portion of each user's ratings for training
            % Outputs:
            %   Itr: training vector of user ids
            %   Jtr: training vector of item ids
            %   Ktr: training vector of user ratings for items
            %   Ite: testing vector of user ids
            %   Jte: testing vector of item ids
            %   Kte: testing vector of user ratings for items
            %
            user_ids = unique(I);
            Itr = [];
            Ite = [];
            Jtr = [];
            Jte = [];
            Ktr = [];
            Kte = [];
            for i=1:numel(user_ids),
                uid = user_ids(i);
                u_idx = find(I == uid);
                ur_count = numel(u_idx);
                tr_count = ceil(tr_frac * ur_count);
                u_idx = u_idx(randperm(ur_count));
                Itr = [Itr; I(u_idx(1:tr_count))];
                Jtr = [Jtr; J(u_idx(1:tr_count))];
                Ktr = [Ktr; K(u_idx(1:tr_count))];
                Ite = [Ite; I(u_idx((tr_count+1):end))];
                Jte = [Jte; J(u_idx((tr_count+1):end))];
                Kte = [Kte; K(u_idx((tr_count+1):end))];
            end
            m = max(I);
            n = max(J);
            Rtr = sparse(Itr,Jtr,Ktr,m,n);
            Rte = sparse(Ite,Jte,Kte,m,n);
            return
        end
        
        function [ A B ] = factor_huber( X, r, s, lam_l2, opts )
            % Compute a factorization of the matrix X as X = A*B, in which A
            % and B are both rank r matrices. Use Huberized squared loss, 
            % setting the l2-l1 transition so that numel(X)*s entries incur l1, 
            % rather than l2 loss. Use simultaneous descent on A and B.
            %
            % Inputs:
            %   X: matrix to factorize (obs_count x obs_dim)
            %   r: rank of the factorization
            %   s: estimate of the fraction of outlier entries
            %   lam_l2: L2 regularization strength on A and B
            %   opts: an options structure
            %     opts.Ai: initializer for A
            %     opts.Bi: initializer for B
            %     opts.outer_iters: max number of outer optimization iterations
            %     opts.inner_iters: max number of inner optimization iterations
            % Outputs:
            %   Xh: learned approximation Xh = A*B
            %   A: factor matrix of dimension (obs_count x r)
            %   B: factor matrix of dimension (r x obs_dim)
            %

            obs_count = size(X,1);
            obs_dim = size(X,2);
            s_count = ceil(s * obs_count * obs_dim);
            if ~exist('opts','var')
                opts = struct();
            end
            if isfield(opts,'Ai')
                A = opts.Ai;
            else
                A = randn(obs_count,r);
            end
            if isfield(opts,'Bi')
                B = opts.Bi;
            else
                B = randn(r,obs_dim);
            end
            if isfield(opts,'outer_iters')
                outer_iters = opts.outer_iters;
            else
                outer_iters = 20;
            end
            if isfield(opts,'inner_iters')
                inner_iters = opts.inner_iters;
            else
                inner_iters = 5;
            end
            if (size(A,2) ~= r || size(B,1) ~= r)
                error('factor_huber(): initializing matrices should agree with r.\n');
            end

            options = struct();
            options.Display = 'iter';
            options.Method = 'lbfgs';
            options.Corr = 10;
            options.LS = 3;
            options.LS_init = 3;
            options.MaxIter = inner_iters;
            options.MaxFunEvals = 10*inner_iters;
            options.TolX = 1e-5;

            nz_mask = find(X);
            for i=1:outer_iters,
                % Check error of reconstruction and set l2->l2 threshold
                Xh = A * B;
                Rsq = (Xh - X).^2;
                Rsq = sort(Rsq(nz_mask),'descend');
                d = 100; %sqrt(Rsq(s_count));
                % Optimize A and B simultaneously
                funObj = @( w ) RecSys.objfun_AB(w, X, r, d, lam_l2);
                [AB fval flag output] = minFunc(funObj, [A(:); B(:)], options);
                A = reshape(AB(1:numel(A)), obs_count, r);
                B = reshape(AB(numel(A)+1:end), r, obs_dim);
                fprintf('Iter %d, fval=%.8f\n',i,fval);
                if (output.iterations == 1)
                    break;
                end
            end
            return
        end

        function [ L dLdW ] = objfun_AB( w, X, r, d, lam_l2 )
            % Compute the loss and gradients for approximating the matrix X as
            % a product of two rank r matrices, whose elements are stored
            % (linearly) in w, using robust Huberized squared loss.
            %
            % Parameters:
            %   w: linearized encoding of the matrices A and B in X = A*B
            %   X: the observation matrix being "factorized"
            %   r: the dimension of the factorization space
            %   d: the threshold at which to transition from l2 to l1 loss
            %   lam_l2: L2 regularization weight on A and B
            % Outputs:
            %   L: loss for reconstruction of X and for regularization of A/B
            %   dLdW: gradients of loss with respect to elements of w
            %
            p_count = numel(X);
            A_dim = [size(X,1) r];
            B_dim = [r size(X,2)];
            A = reshape(w(1:A_dim(1)*A_dim(2)), A_dim(1), A_dim(2));
            B = reshape(w(A_dim(1)*A_dim(2)+1:end), B_dim(1), B_dim(2));
            R = A*B - X;
            % Create a mask of entries of X_res that incur l1 loss
            loss_mask = ((R > d) | (R < -d));
            L_rec = sum(sum(R(~loss_mask).^2));
            L_rec = L_rec + sum(sum((2 * d * abs(R(loss_mask))) - d^2));
            L_reg = lam_l2 * (sum(sum(A.^2)) + sum(sum(B.^2)));
            L = (L_rec / p_count) + L_reg;
            if (nargout > 1)
                % Stick l2 loss gradient with respect to residuals in all spots
                dLdR = 2 * R;
                % Overwrite with l1 loss gradient where relevant
                dLdR(loss_mask) = 2 * d * sign(R(loss_mask));
                % Backpropagate loss gradient with respect to residuals and add
                % gradients w.r.t. L2 regularization terms.
                dLdA = ((2 / p_count) * dLdR * B') + (lam_l2 * 2 * A);
                dLdB = ((2 / p_count) * A' * dLdR) + (lam_l2 * 2 * B);
                % Linearize gradients
                dLdW = [dLdA(:); dLdB(:)];
            end
            return
        end

    end
    
end

