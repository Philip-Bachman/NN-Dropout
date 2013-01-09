%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file is for testing the effect of input dimension on drop stride on the
% learning process.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
obs_dim = 4;
obs_count = 2500;
train_count = 500;
obs_noise = 0.00;
label_noise = 0.00;
data_count = 5;
tt_cycles = 5;
%obs_dims = [2 8];
drop_strides = [1 4 8];

% Set up parameters common to all training and testing
params = struct();
params.epochs = 10000;
params.start_rate = 1.0;
params.decay_rate = 0.1^(1 / params.epochs);
params.momentum = 0.5;
params.weight_bound = 20;
params.batch_size = 100;
params.batch_rounds = 1;
params.dr_obs = 0.0;
params.dr_node = 0.5;
params.do_validate = 1;

data_res_accs = zeros(data_count, numel(drop_strides), params.epochs);
data_res_loss = zeros(data_count, numel(drop_strides), params.epochs);
for data_num=1:data_count,
    fprintf('DATA NUMBER %d OF %d\n',data_num,data_count);
    % Generate griddy two-class data for this set of train/test cycles
    [ Xi Yi ] = data_grid( obs_count, 0.35, 0.45, 4, 3 );
    for i=1:numel(Yi),
        if (rand() < label_noise)
            Yi(i) = -Yi(i);
        end
    end
    for stride_num=1:numel(drop_strides),
        fprintf('STRIDE NUMBER %d OF %d\n',stride_num,numel(drop_strides));
        % Augment the data for this observation dimension
        Y = zeros(obs_count, 2);
        X = [Xi zeros(obs_count,obs_dim-2)];
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
        % Generate a SimpleNet instance
        act_func = ActFunc(2);
        out_func = ActFunc(1);
        loss_func = LossFunc(3);
        layer_sizes = [obs_dim 64 64 2];
        net = SimpleNet(layer_sizes, act_func, out_func, loss_func);
        net.drop_stride = drop_strides(stride_num);
        % Set the validation data in the parameters struct
        params.X_v = X_te;
        params.Y_v = Y_te;
        tt_res_accs = zeros(1, params.epochs);
        tt_res_loss = zeros(1, params.epochs);
        for tt_cycle=1:tt_cycles,
            fprintf('CYCYLE %d OF %d\n',tt_cycle,tt_cycles);
            % Run a train/test cycle for this net with this data
            net.init_weights(1.0);
            result = net.complex_update(X_tr, Y_tr, params);
            tt_res_accs = tt_res_accs + result.test_accs;
            tt_res_loss = tt_res_loss + result.test_loss;
        end
        data_res_accs(data_num,stride_num,:) = tt_res_accs ./ tt_cycles;
        data_res_loss(data_num,stride_num,:) = tt_res_loss ./ tt_cycles;
    end
end






%%%%%%%%%%%%%%
% EYE BUFFER %
%%%%%%%%%%%%%%