function [ Xtr Ytr Xte Yte ] = trte_sample( X, Y, tr_count, te_count )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

c_labels = unique(Y);

Xtr = [];
Ytr = [];
Xte = [];
Yte = [];
trte_count = tr_count + te_count;
for i=1:numel(c_labels),
    label = c_labels(i);
    idx = find(Y == label);
    idx = randsample(idx,trte_count,false);
    Xtr = [Xtr; X(idx(1:tr_count),:)];
    Ytr = [Ytr; Y(idx(1:tr_count),:)];
    Xte = [Xte; X(idx((tr_count+1):end),:)];
    Yte = [Yte; Y(idx((tr_count+1):end),:)];
end

return
end

