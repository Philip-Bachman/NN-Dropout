function [ Yi ] = class_inds(Y, class_count)
% Convet categorical class values into +1/-1 indicator matrix

class_labels = sort(unique(Y),'ascend');
if ~exist('class_count','var')
    class_count = numel(class_labels);
end

Yi = -ones(size(Y,1),class_count);
for i=1:numel(class_labels),
    c_idx = (Y == class_labels(i));
    Yi(c_idx,i) = 1;
end
return
end