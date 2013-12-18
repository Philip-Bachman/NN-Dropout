function [ fig ] = draw_usps_filters( filters, draw_count, has_bias, in_order )

if ~exist('has_bias','var')
    has_bias = 0;
end

if (has_bias == 1)
    filters = filters(:,1:(end-1));
end
if ~exist('in_order','var')
    in_order = 0;
end

% Get either ordered or randomly sampled list of indices
if (in_order == 1)
    idx = 1:draw_count;
else
    idx = randsample(size(filters,1),draw_count);
end

fig = figure();
sq_size = ceil(sqrt(draw_count));
im_size = round(sqrt(size(filters,2)));
for j=1:draw_count,
    subplot(sq_size,sq_size,j);
    imagesc(reshape(filters(idx(j),:),im_size,im_size)');
    set(gca,'xtick',[],'ytick',[]);
    axis square;
    colormap('gray');
end

return
end