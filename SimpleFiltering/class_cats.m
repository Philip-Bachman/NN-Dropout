function [ Yc ] = class_cats(Yi)
% Convert +1/-1 indicator class matrix to a vector of categoricals
[vals Yc] = max(Yi,[],2);
return
end