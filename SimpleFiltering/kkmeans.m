function [ means member_counts ] = kkmeans( X, K, kk, round_count, do_dot )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

if (do_dot == 1)
    X = bsxfun(@rdivide, X, sqrt(sum(X.^2,2)+1e-8));
end

obs_count = size(X,1);

% Initialize means to random samples from training data
idx = randsample(obs_count, K);
means = X(idx,:);

% Set up blocks to use in updates due to memory constraints
block_starts = 1:1000:obs_count;
block_starts = [block_starts obs_count+1];
block_count = numel(block_starts);

Xsq = sum(X.^2,2);
member_counts = zeros(round_count, K);
% Perform round_count rounds of kk-means updating.
fprintf('Doing %d k-means updates:\n', round_count);
for i=1:round_count,
    Msq = sum(means.^2,2);
    mi = zeros(obs_count,kk);
    md = zeros(obs_count,kk);
    fprintf('%d blocks:',block_count);
    for j=2:block_count,
        block_idx = block_starts(j-1):(block_starts(j)-1);
        dots = X(block_idx,:) * means';
        if (do_dot == 1)
            [bmd bmi] = sort(dots,2,'descend');
        else
            norms = bsxfun(@plus, Xsq(block_idx), Msq) - (2*dots);
            [bmd bmi] = sort(norms,2,'ascend');
        end
        md(block_idx,:) = bmd(:,1:kk);
        mi(block_idx,:) = bmi(:,1:kk);
        fprintf('.');
    end
    for j=1:K,
        k_idx = (sum(bsxfun(@eq, mi, j),2) > 0.5);
        member_counts(i,j) = sum(k_idx);
        means(j,:) = sum(X(k_idx,:),1) ./ sum(k_idx);
        if (sum(k_idx) == 0)
            fprintf('  +RESET CENTROID\n');
            means(j,:) = X(randi(obs_count),:);
        end
    end 
    means = bsxfun(@rdivide, means, sqrt(sum(means.^2,2)+1e-8));
    d = mean(md,1);
    fprintf('  %d: ',i);
    for j=1:kk,
        fprintf('%.4f ',d(j));
    end
    fprintf('\n');
end


end

