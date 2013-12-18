function [ opt_theta accs svm_eval ] = svm_train_test( Xtr, Ytr, Xte, Yte, lams )
% Compute svm training/testing error over a range of regularization weights
%
% Parameters:
%   Xtr: training observations
%   Ytr: classes (integer-valued) for training observations
%   Xtr: testing observations
%   Ytr: classes (integer-valued) for testing observations
%   lams: sequence of lambda/regularizations weights to test
% Outputs:
%   opt_theta: learned parameters for lambda with best test accuracy
%   accs: array of size (numel(lams) x 2), with train/test acc for each lambda
%   svm_eval: a function that outputs labels for a set of test samples, this
%             function uses the theta with best test accuracy
%

% Set options for minFunc to reasonable values
opts = struct();
opts.MaxIter = 750;
opts.Display = 'iter';
opts.Method = 'lbfgs';
opts.Corr = 20;
opts.Damped = 1;
opts.LS = 0;
opts.LS_type = 0;
opts.LS_init = 3;
opts.use_mex = 0;

% Create a mapping of class labels => 1:label_count
label_map = sort(unique(Ytr),'ascend');
label_count = numel(label_map);
Ytr_i = zeros(size(Ytr));
Yte_i = zeros(size(Yte));
for l=1:label_count,
    Ytr_i(Ytr == label_map(l)) = l;
    Yte_i(Yte == label_map(l)) = l;
end

% For each regularization weight, train and test an SVM
opt_acc = 0.0;
opt_theta = [];
accs = zeros(numel(lams),2);
fprintf('Training/testing with %d lambdas:\n',numel(lams));
for l_num=1:numel(lams),
    % Train an SVM using one of the given lambdas
    theta = svm_coates(Xtr, Ytr_i, lams(l_num), opts);
    % Evaluate SVM with theta learned for this lambda
    [vals, class_tr] = max(Xtr*theta, [], 2);
    [vals, class_te] = max(Xte*theta, [], 2);
    % Compute and record train/test accuracy
    train_acc = 100 * (sum(class_tr == Ytr_i) / numel(Ytr_i));
    test_acc = 100 * (sum(class_te == Yte_i) / numel(Yte_i));
    accs(l_num,1) = train_acc;
    accs(l_num,2) = test_acc;
    fprintf('  lam: %.4f, train: %.4f, test: %.4f\n',...
        lams(l_num), train_acc, test_acc);
    % Record test_acc and theta if a new optimum was found
    if (test_acc > opt_acc)
        opt_acc = test_acc;
        opt_theta = theta;
    end
end

% Define a classifier using the optimal theta via closure
function classes = svm_class(X, T)
    [v c] = max(X*T, [], 2);
    classes = label_map(c);
    return
end
svm_eval = @( X ) svm_class(X, opt_theta);

return
end

function theta = svm_coates(trainXC, trainY, C, opts)
% Train an SVM using general function minimization
%
% Based on code by Adam Coates (from sc_vq_demo.tgz)
%
numClasses = max(trainY);
w0 = zeros(size(trainXC,2)*numClasses, 1);
w = minFunc(@l2svmloss_coates, w0, opts, trainXC, trainY, numClasses, C);
theta = reshape(w, size(trainXC,2), numClasses);
return
end

function [loss, g] = l2svmloss_coates(w, X, y, K, C)
% 1-vs-all L2-svm loss function: similar to LibLinear.
%
% Based on coded by Adam Coates (from sc_vq_demo.tgz)
%
[M,N] = size(X);
theta = reshape(w, N,K);
Y = bsxfun(@(y,ypos) 2*(y==ypos)-1, y, 1:K);

margin = max(0, 1 - Y .* (X*theta));
loss = (0.5 * sum(theta.^2)) + C*mean(margin.^2);
loss = sum(loss);   
g = theta - 2*C/M * (X' * (margin .* Y));

% adjust for intercept term
loss = loss - 0.5 * sum(theta(end,:).^2);
g(end,:) = g(end, :) - theta(end,:);
g = g(:);
return
end

