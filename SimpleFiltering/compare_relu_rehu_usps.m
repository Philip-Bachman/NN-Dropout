% Compare performance of ReLu and ReHu activation functions on USPS digits, 
% using random and k-means filters. Random filters are set to unit norm, and
% their +/- signs are flipped to maximize variance on the data. A bias term is
% added, to set the response sparsity of each feature to a target amount.
%
%

load('usps.mat');
X = ZMUN(X);

test_count = 50;
sparse_vals = [1.0 0.5];
filt_counts = [100 250 500 750 1000];
c_vals = [0.1 0.25 0.5 1.0 2.0 5.0 10.0 20.0];
avg_acts = [0.10 0.15 0.20];

rehu_accs = zeros(numel(sparse_vals),numel(filt_counts),numel(avg_acts),test_count);
relu_accs = zeros(numel(sparse_vals),numel(filt_counts),numel(avg_acts),test_count);
for t_num=1:test_count,
    % Sample a train/test split, and further split for a validation set
    [Xtr Ytr Xte Yte] = trte_split(X,Y,0.8);
    Xtr_v = Xtr(1:1000,:);
    Ytr_v = Ytr(1:1000,:);
    Xtr_t = Xtr(1001:end,:);
    Ytr_t = Ytr(1001:end,:);
    for s_num=1:numel(sparse_vals),
        s_val = sparse_vals(s_num);
        for f_num=1:numel(filt_counts),
            f_val = filt_counts(f_num);
            for a_num=1:numel(avg_acts),
                a_val = avg_acts(a_num);
                fprintf('==================================================\n');
                fprintf('TEST: %d\n',t_num);
                fprintf('S: %.2f, F: %d, A: %.2f\n',s_val,f_val,a_val);
                fprintf('==================================================\n');
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Generate random filters according with desired params %
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                clear SF;
                SF = SimpleFilter(1, @SimpleFilter.actfun_rehu, 1.0);
                SF.init_rand_filters(Xtr_t, f_val, s_val, a_val);

                %%%%%%%%%%%%%%%%%%%%%%%%
                % Test ReHu activation %
                %%%%%%%%%%%%%%%%%%%%%%%%
                fprintf('Testing ReHu:\n');
                % Find best c value on the validation set
                Ftr_v = SF.evaluate(Xtr_v);
                Ftr_t = SF.evaluate(Xtr_t);
                fprintf('Checking c values on validation set...\n');
                max_acc = 0;
                c_opt = c_vals(1);
                for c_num=1:numel(c_vals),
                    c = c_vals(c_num);
                    svm_str = sprintf('-s 1 -c %.4f -B 1 -q',c);
                    SVM = train(Ytr_t,sparse(Ftr_t),svm_str);
                    [aaa bbb] = predict(Ytr_v,sparse(Ftr_v),SVM);
                    if (bbb > max_acc)
                        max_acc = bbb;
                        c_opt = c_vals(c_num);
                    end
                end

                % Retrain on the full training set, using validated c value
                Ftr = SF.evaluate(Xtr);
                Fte = SF.evaluate(Xte);
                svm_str = sprintf('-s 1 -c %.4f -B 1 -q',c_opt);
                SVM = train(Ytr,sparse(Ftr),svm_str);
                % Test on test and record result
                [aaa bbb] = predict(Yte,sparse(Fte),SVM);
                rehu_accs(s_num,f_num,a_num,t_num) = bbb;

                %%%%%%%%%%%%%%%%%%%%%%%%
                % Test ReLu activation %
                %%%%%%%%%%%%%%%%%%%%%%%%
                fprintf('Testing ReLu:\n');
                SF.act_func = @SimpleFilter.actfun_soft_relu;
                % Find best c value on the validation set
                Ftr_v = SF.evaluate(Xtr_v);
                Ftr_t = SF.evaluate(Xtr_t);
                fprintf('Checking c values on validation set...\n');
                max_acc = 0;
                c_opt = c_vals(1);
                for c_num=1:numel(c_vals),
                    c = c_vals(c_num);
                    svm_str = sprintf('-s 1 -c %.4f -B 1 -q',c);
                    SVM = train(Ytr_t,sparse(Ftr_t),svm_str);
                    [aaa bbb] = predict(Ytr_v,sparse(Ftr_v),SVM);
                    if (bbb > max_acc)
                        max_acc = bbb;
                        c_opt = c_vals(c_num);
                    end
                end

                % Retrain on the full training set, using validated c value
                Ftr = SF.evaluate(Xtr);
                Fte = SF.evaluate(Xte);
                svm_str = sprintf('-s 1 -c %.4f -B 1 -q',c_opt);
                SVM = train(Ytr,sparse(Ftr),svm_str);
                % Test on test and record result
                [aaa bbb] = predict(Yte,sparse(Fte),SVM);
                relu_accs(s_num,f_num,a_num,t_num) = bbb;
            end
        end
    end
    
    save('result_relu_rehu_usps.mat');
    
end
    
    