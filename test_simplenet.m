% Generate some simple data with which to test the SimpleNet class
obs_dim = 20;
obs_count = 4000;
train_count = 500;
obs_noise = 0.1;
label_noise = 0.0;

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

[ X Y ] = data_grid( obs_count, 0.4, 0.5, 4, 5 );
X = [X zeros(obs_count,obs_dim-2)];
rbf_idx = randsample(obs_count,obs_dim-2,false);
for i=1:obs_count,
    for j=1:numel(rbf_idx),
        r = rbf_idx(j);
        X(i,j+2) = exp(-sum((X(i,1:2)-X(r,1:2)).^2));
    end
end
X = ZMUV(X);

% Split data into training and testing portions
X_tr = X(1:train_count,:);
X_te = X(train_count+1:end,:);
Y_tr = Y(1:train_count);
Y_te = Y(train_count+1:end);

% Use sigmoid activation in hidden layers, linear activation in output layer,
% and binomial-deviance loss at output layer.
act_func = ActFunc(2);
out_func = ActFunc(1);
loss_func = LossFunc(3);
layer_sizes = [obs_dim 5*obs_dim 5*obs_dim 1];

% Generate a SimpleNet instance
net = SimpleNet(layer_sizes, act_func, out_func, loss_func);
net.init_weights(1.0);

% Set up parameter struct for updates
params = struct();
params.epochs = 7500;
params.start_rate = 1.0;
params.decay_rate = 0.1^(1 / params.epochs);
params.weight_bound = 10;
params.batch_size = 50;
params.batch_rounds = 1;
params.dr_obs = 0.0;
params.dr_node = 0.5;
params.do_validate = 1;
params.X_v = X_te(1:1000,:);
params.Y_v = Y_te(1:1000);






%%%%%%%%%%%%%%
% EYE BUFFER %
%%%%%%%%%%%%%%