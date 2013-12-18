% Test on pre-extracted centroids

%% Load CIFAR training data
load('cifar_color_data.mat');
Ytr_cifar = Ytr_cifar + 1;
load('cifar_centroids_1000.mat');
load('cifar_spdf_filters.mat');

% Use the same alpha as in the original paper by Coates et. al.
alpha = 0.25;
test_count = 1;
lambdas = [0.1 1.0 2.0 4.0];

kmeans_accs = cell(1,numel(test_count));
spdf_st_accs = cell(1,numel(test_count));
spdf_un_accs = cell(1,numel(test_count));


for t_num=1:test_count,

    % Get a train/test split of the CIFAR 10 images, with 400 images per
    % class in both the training and testing splits.
    [Xtr Ytr Xte Yte] = trte_sample(Xtr_cifar,Ytr_cifar,400,400);
    % Convert sampled subset of images to double (from uint8).
    Xtr = double(Xtr);
    Ytr = double(Ytr);
    Xte = double(Xte);
    Yte = double(Yte);

    %%%%%%%%%%%%%%%%%%%%%%
    % Test SPDF Features %
    %%%%%%%%%%%%%%%%%%%%%% 
    % extract features from training/testing images
    Ftr = extract_spdf_features(Xtr,spdf_filters,rfSize,CIFAR_DIM,M,P,bias_val);
    Fte = extract_spdf_features(Xte,spdf_filters,rfSize,CIFAR_DIM,M,P,bias_val);
    % train and test svm on unstandardized data
    [opt_theta accs] = svm_train_test(Ftr,Ytr,Fte,Yte,(lambdas./40));
    spdf_un_accs{t_num} = accs;
    % standardize data
    F_mean = mean(Ftr);
    F_sd = sqrt(var(Ftr) + 1e-3);
    Ftr = bsxfun(@rdivide, bsxfun(@minus, Ftr, F_mean), F_sd);
    Ftr = [Ftr ones(size(Ftr,1),1)];
    Fte = bsxfun(@rdivide, bsxfun(@minus, Fte, F_mean), F_sd);
    Fte = [Fte ones(size(Fte,1),1)];
    % train and test svm on standardized data
    [opt_theta accs] = svm_train_test(Ftr,Ytr,Fte,Yte,lambdas);
    spdf_st_accs{t_num} = accs;
    clear Ftr Fte;

    %%%%%%%%%%%%%%%%%%%%%%%%
    % Test KMeans Features %
    %%%%%%%%%%%%%%%%%%%%%%%%
    enc = 'relu';        
    % extract features from training/testing images
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
    kmeans_accs{t_num} = accs;
    clear Ftr Fte;

    save('cifar_stuff_result.mat','relu_accs','rehu_accs');

    
end