% Generate some simple data with which to test the SimpleNet class
obs_dim = 10;
obs_count = 2000;
train_count = 250;
obs_noise = 0.10;
label_noise = 0.05;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate observations and labels by some random process %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% b = randn(obs_dim, 1);
% X = randn(obs_count, obs_dim);
% Y = sign(X*b);
% % Add noise to observations and labels
% X = X + (randn(size(X)) .* obs_noise);
% for i=1:numel(Y),
%     if (rand() < label_noise)
%         Y(i) = -Y(i);
%     end
% end

[ X Yi ] = data_grid( obs_count, 0.35, 0.45, 4, 3 );
for i=1:numel(Yi),
    if (rand() < label_noise)
        Yi(i) = -Yi(i);
    end
end
Y = zeros(obs_count, 2);
X = [X zeros(obs_count,obs_dim-2)];
rbf_idx = randsample(obs_count,obs_dim-2,false);
for i=1:obs_count,
    for j=1:numel(rbf_idx),
        r = rbf_idx(j);
        %X(i,j+2) = exp(-sum((X(i,1:2)-X(r,1:2)).^2));
        X(i,j+2) = randn();
    end
    if (Yi(i) == 1)
        Y(i,1) = 1;
        Y(i,2) = -1;
    else
        Y(i,1) = -1;
        Y(i,2) = 1;
    end
end
X = ZMUV(X);

% Split data into training and testing portions
X_tr = X(1:train_count,:);
X_te = X(train_count+1:end,:);
Y_tr = Y(1:train_count,:);
Y_te = Y(train_count+1:end,:);

% Use sigmoid activation in hidden layers, linear activation in output layer,
% and binomial-deviance loss at output layer.
act_func = ActFunc(2);
out_func = ActFunc(1);
loss_func = LossFunc(3);
layer_sizes = [obs_dim 80 80 2];

% Generate a SimpleNet instance
net = SimpleNet(layer_sizes, act_func, out_func, loss_func);
net.init_weights(1.0);
net.drop_stride = 2;

% Set up parameter struct for updates
params = struct();
params.epochs = 10000;
params.start_rate = 1.0;
params.decay_rate = 0.1^(1 / params.epochs);
params.momentum = 0.0;
params.weight_bound = 20;
params.batch_size = 100;
params.batch_rounds = 1;
params.dr_obs = 0.0;
params.dr_node = 0.5;
params.do_validate = 1;
params.X_v = X_te;
params.Y_v = Y_te;






%%%%%%%%%%%%%%
% EYE BUFFER %
%%%%%%%%%%%%%%