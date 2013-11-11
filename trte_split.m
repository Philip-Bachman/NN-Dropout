function [ Xtr Ytr Xte Yte ] = trte_split( X, Y, tr_frac )
% Make a train/test split of the given set of observations
%
obs_count = size(X,1);
tr_count = round(obs_count * tr_frac);
tr_idx = randsample(1:obs_count, tr_count);
te_idx = setdiff(1:obs_count, tr_idx);
Xtr = X(tr_idx,:);
Ytr = Y(tr_idx,:);
Xte = X(te_idx,:);
Yte = Y(te_idx,:);
return
end

