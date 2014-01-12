clear;

% Load USPS digits data
load('mnist_data.mat');
X = double(X_mnist) ./ max(max(double(X_mnist)));
Y = LDNet.class_inds(double(Y_mnist));

test_count = 10;
dev_accs = zeros(1,test_count);
sde_accs = zeros(1,test_count);
raw_accs = zeros(1,test_count);
dev_loss = zeros(1,test_count);
sde_loss = zeros(1,test_count);
raw_loss = zeros(1,test_count);

for i=1:test_count,
    
    fprintf('RUNNING TEST %d:\n',i);
    
    % Get a train/test split for this 
    [Xtr Ytr Xte Yte] = trte_split(X,Y,0.8);
    
    % Train DEV, SDE, and RAW nets
    LDN_dev = train_mnist(Xtr, Ytr, Xte, Yte, 'dev');
    LDN_sde = train_mnist(Xtr, Ytr, Xte, Yte, 'sde');
    LDN_raw = train_mnist(Xtr, Ytr, Xte, Yte, 'raw');
    
    % Check performance of the learned parameters
    [L_dev A_dev] = LDN_dev.check_loss(Xte,Yte);
    [L_sde A_sde] = LDN_sde.check_loss(Xte,Yte);
    [L_raw A_raw] = LDN_raw.check_loss(Xte,Yte);
    
    % Record performance
    dev_accs(i) = A_dev;
    dev_loss(i) = L_dev;
    sde_accs(i) = A_sde;
    sde_loss(i) = L_sde;
    raw_accs(i) = A_raw;
    raw_loss(i) = L_raw;
    
    save('res_test_mnist.mat','dev_accs','dev_loss','sde_accs','sde_loss',...
        'raw_accs','raw_loss');
end






