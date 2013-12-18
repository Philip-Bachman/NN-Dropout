% Test on pre-extracted centroids

%% Load CIFAR training data
load('cifar_color_data.mat');
Ytr_cifar = Ytr_cifar + 1;
load('cifar_centroids_1000.mat');

alphas = [0.4 0.2 0.0];
test_count = 5;
lambdas = [0.1 1.0 2.0 4.0];

relu_accs = cell(numel(test_count),numel(alphas));
rehu_accs = cell(numel(test_count),numel(alphas));
test_alphas = cell(numel(test_count),numel(alphas));

for t_num=1:test_count,
    
    for a_num=1:numel(alphas),
    
        [Xtr Ytr Xte Yte] = trte_sample(Xtr_cifar,Ytr_cifar,400,400);

        Xtr = double(Xtr);
        Ytr = double(Ytr);
        Xte = double(Xte);
        Yte = double(Yte);
        
        % Set the alpha (i.e. negative feature bias) for this test
        alpha = alphas(a_num);
        test_alphas{t_num,a_num} = alpha;

        %%%%%%%%%%%%%%%%%%%%%%%%
        % Test ReLu activation %
        %%%%%%%%%%%%%%%%%%%%%%%%
        enc = 'relu';        
        % extract training features
        Ftr = extract_features(Xtr,centroids,rfSize,CIFAR_DIM,M,P,enc,alpha);
        Fte = extract_features(Xte,centroids,rfSize,CIFAR_DIM,M,P,enc,alpha);
        % standardize data
        F_mean = mean(Ftr);
        F_sd = sqrt(var(Ftr) + 1e-3);
        Ftr = bsxfun(@rdivide, bsxfun(@minus, Ftr, F_mean), F_sd);
        Ftr = [Ftr ones(size(Ftr,1),1)];
        Fte = bsxfun(@rdivide, bsxfun(@minus, Fte, F_mean), F_sd);
        Fte = [Fte ones(size(Fte,1),1)];
        % train and test svm on multiple lambdas
        [opt_theta accs] = svm_train_test(Ftr,Ytr,Fte,Yte,lambdas);
        relu_accs{t_num,a_num} = accs;
        clear Ftr Fte;
        
        %%%%%%%%%%%%%%%%%%%%%%%%
        % Test ReHu activation %
        %%%%%%%%%%%%%%%%%%%%%%%%
        enc = 'rehu';      
        % extract training features
        Ftr = extract_features(Xtr,centroids,rfSize,CIFAR_DIM,M,P,enc,alpha);
        Fte = extract_features(Xte,centroids,rfSize,CIFAR_DIM,M,P,enc,alpha);
        % standardize data
        F_mean = mean(Ftr);
        F_sd = sqrt(var(Ftr) + 1e-3);
        Ftr = bsxfun(@rdivide, bsxfun(@minus, Ftr, F_mean), F_sd);
        Ftr = [Ftr ones(size(Ftr,1),1)];
        Fte = bsxfun(@rdivide, bsxfun(@minus, Fte, F_mean), F_sd);
        Fte = [Fte ones(size(Fte,1),1)];
        % train and test svm on multiple lambdas
        [opt_theta accs] = svm_train_test(Ftr,Ytr,Fte,Yte,lambdas);
        rehu_accs{t_num,a_num} = accs;
        clear Ftr Fte;
        
        save('cifar_stuff_result.mat','relu_accs','rehu_accs');
        
    end
    
    
end