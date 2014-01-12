clear;

% Load USPS digits data
load('mnist_data.mat');
X = double(X_mnist) ./ max(max(double(X_mnist)));
Y = LDNet.class_inds(double(Y_mnist));

test_count = 3;
sup_counts = [500 1000 1500 2000 2500 5000 10000 20000];

dev_accs = zeros(numel(sup_counts),test_count);
sde_accs = zeros(numel(sup_counts),test_count);
raw_accs = zeros(numel(sup_counts),test_count);
dev_loss = zeros(numel(sup_counts),test_count);
sde_loss = zeros(numel(sup_counts),test_count);
raw_loss = zeros(numel(sup_counts),test_count);

for i=1:test_count,
    for j=1:numel(sup_counts),
        sup_count = sup_counts(j);
        fprintf('RUNNING TEST %d, SUP_COUNT %d:\n', i, sup_count);
        
        % Get a train/test split for this 
        [Xtr Ytr Xte Yte] = trte_split(X,Y,0.8);

        % Split training data into supervised and unsupervised portions
        Xtr_u = Xtr((sup_count+1):end,:);
        Ytr_u = Ytr((sup_count+1):end,:);
        Xtr = Xtr(1:sup_count,:);
        Ytr = Ytr(1:sup_count,:);

        % Train DEV, SDE, and RAW nets
        LDN_dev = train_mnist_ss(Xtr, Ytr, [Xtr; Xtr_u], Xte, Yte, 'dev');
        LDN_sde = train_mnist_ss(Xtr, Ytr, [Xtr; Xtr_u], Xte, Yte, 'sde');
        LDN_raw = train_mnist_ss(Xtr, Ytr, [Xtr; Xtr_u], Xte, Yte, 'raw');

        % Check performance of the learned parameters
        [L_dev A_dev] = LDN_dev.check_loss(Xte,Yte);
        [L_sde A_sde] = LDN_sde.check_loss(Xte,Yte);
        [L_raw A_raw] = LDN_raw.check_loss(Xte,Yte);

        % Record performance
        dev_accs(j,i) = A_dev;
        dev_loss(j,i) = L_dev;
        sde_accs(j,i) = A_sde;
        sde_loss(j,i) = L_sde;
        raw_accs(j,i) = A_raw;
        raw_loss(j,i) = L_raw;

        save('res_test_mnist_ss.mat','dev_accs','dev_loss','sde_accs','sde_loss',...
            'raw_accs','raw_loss');
    end
end






