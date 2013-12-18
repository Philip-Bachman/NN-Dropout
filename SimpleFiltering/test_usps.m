load('usps.mat');
X = ZMUN(X);

test_count = 3;
train_frac = 0.75;
train_rounds = 500;
svm_lams = [0.25 0.5 1.0 2.0 4.0 8.0];
spdf_accs = zeros(1,test_count);
rand_accs = zeros(1,test_count);

for i=1:test_count,
    [Xtr Ytr Xte Yte] = trte_split(X, Y, train_frac);
    SF = SimpleFilter(1,@SimpleFilter.actfun_rehu);
    [opts W] = SF.train(Xtr, 512, train_rounds, 1);
    % Test first with learned filters
    Ftr = SF.evaluate(Xtr);
    Fte = SF.evaluate(Xte);
    max_acc = 0;
    for j=1:numel(svm_lams),
        svm_str = sprintf('-s 1 -c %.4f -B 1 -q',svm_lams(j));
        SVM = train(Ytr,sparse(Ftr),svm_str);
        [xx1 acc xx2] = predict(Yte,sparse(Fte),SVM);
        if (acc > max_acc)
            max_acc = acc;
        end
    end
    spdf_accs(i) = max_acc;
    % Then test with random initializing filters
    SF.filters = SF.pre_filters;
    Ftr = SF.evaluate(Xtr);
    Fte = SF.evaluate(Xte);
    max_acc = 0;
    for j=1:numel(svm_lams),
        svm_str = sprintf('-s 1 -c %.4f -B 1 -q',svm_lams(j));
        SVM = train(Ytr,sparse(Ftr),svm_str);
        [xx1 acc xx2] = predict(Yte,sparse(Fte),SVM);
        if (acc > max_acc)
            max_acc = acc;
        end
    end
    rand_accs(i) = max_acc;
    
    % Output for tracking test progress
    fprintf('============================================================\n');
    fprintf('ROUND %d, spdf: %.4f, rand: %.4f\n',i,spdf_accs(i),rand_accs(i));
    fprintf('============================================================\n');
    
end








