clear comp_net feat_net class_net decode_net;

% Load data to run through network

% Create the BaseNet instances
feat_dims = [size(Xtr,2) 50 32 16];
decode_dims = [feat_dims(end) 32 size(Xtr,2)];

feat_net = BaseNet(feat_dims, ActFunc(5), ActFunc(5));
decode_net = BaseNet(decode_dims, ActFunc(5), ActFunc(1));

% Create the CompNet using the BaseNet instances
comp_net = CompLMNN(Xtr, feat_net, decode_net);

% Test the CompNet, simply
comp_net.init_weights(0.1);
comp_net.set_drop_rates(0.0);