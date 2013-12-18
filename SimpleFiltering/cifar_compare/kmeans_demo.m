%% Configuration
rfSize = 6;
numCentroids=1600;
whitening=true;
numPatches = 300000;
CIFAR_DIM=[32 32 3];

%% Load CIFAR training data
load('cifar_color_data.mat');

trainX = Xtr_cifar;
trainY = Ytr_cifar + 1;
clear Xtr_cifar Ytr_cifar;

% extract random patches
patches = zeros(numPatches, rfSize*rfSize*3);
for i=1:numPatches
  if (mod(i,10000) == 0)
      fprintf('Extracting patch: %d / %d\n', i, numPatches);
  end
  r = random('unid', CIFAR_DIM(1) - rfSize + 1);
  c = random('unid', CIFAR_DIM(2) - rfSize + 1);
  img = randi(size(trainX,1));
  patch = reshape(trainX(img,:),CIFAR_DIM);
  patch = patch(r:r+rfSize-1,c:c+rfSize-1,:);
  patches(i,:) = double(patch(:)');
end

% normalize for contrast
patches = bsxfun(@rdivide,...
    bsxfun(@minus, patches, mean(patches,2)), sqrt(var(patches,[],2)+10));

save('cifar_patches.mat','patches','rfSize','CIFAR_DIM');

% whiten
if (whitening)
  C = cov(patches);
  M = mean(patches);
  [V,D] = eig(C);
  P = V * diag(sqrt(1./(diag(D) + 0.1))) * V';
  patches = bsxfun(@minus, patches, M) * P;
end

% run K-means
centroids = run_kmeans(patches, numCentroids, 50);
show_centroids(centroids, rfSize); drawnow;

save('cifar_centroids.mat','centroids','rfSize','CIFAR_DIM','M','P');

% % extract training features
% if (whitening)
%   trainXC = extract_features(trainX, centroids, rfSize, CIFAR_DIM, M,P);
% else
%   trainXC = extract_features(trainX, centroids, rfSize, CIFAR_DIM);
% end
% 
% clear('patches','trainX');
% 
% % standardize data
% trainXC_mean = mean(trainXC);
% trainXC_sd = sqrt(var(trainXC) + 1e-3);
% trainXCs = bsxfun(@rdivide, bsxfun(@minus, trainXC, trainXC_mean), trainXC_sd);
% trainXCs = sparse([trainXCs, ones(size(trainXCs,1),1)]);
% 
% % Train liblinear classifier
% train(trainY,trainXCs,'-s 4 -v 5');

% % train classifier using SVM
% C = 100;
% theta = train_svm(trainXCs, trainY, C);
% 
% [val,labels] = max(trainXCs*theta, [], 2);
% fprintf('Train accuracy %f%%\n', 100 * (1 - sum(labels ~= trainY) / length(trainY)));

% %%%%% TESTING %%%%%
% 
% %% Load CIFAR test data
% fprintf('Loading test data...\n');
% f1=load([CIFAR_DIR '/test_batch.mat']);
% testX = double(f1.data);
% testY = double(f1.labels) + 1;
% clear f1;
% 
% % compute testing features and standardize
% if (whitening)
%   testXC = extract_features(testX, centroids, rfSize, CIFAR_DIM, M,P);
% else
%   testXC = extract_features(testX, centroids, rfSize, CIFAR_DIM);
% end
% testXCs = bsxfun(@rdivide, bsxfun(@minus, testXC, trainXC_mean), trainXC_sd);
% testXCs = [testXCs, ones(size(testXCs,1),1)];
% 
% % test and print result
% [val,labels] = max(testXCs*theta, [], 2);
% fprintf('Test accuracy %f%%\n', 100 * (1 - sum(labels ~= testY) / length(testY)));

