function [ fig ] = draw_usps_poolers( W2, W1, w2_count, w1_count, has_bias )
% Do some nice plotting for second layer pooling features.
if ~exist('has_bias','var')
    has_bias = 0;
end

if (has_bias == 1)
    W2 = W2(:,1:(end-1));
    W1 = W1(:,1:(end-1));
end

fig = figure();

im_size = round(sqrt(size(W1,2)));
w2_idx = randsample(size(W2,1),w2_count);
sp_idx = 1;
for i=1:w2_count,
    [w2_wts w1_idx] = sort(abs(W2(w2_idx(i),:)),'descend');
    subplot(w2_count,(w1_count+1),sp_idx);
    plot(w2_wts(1:100));
    axis tight;
    set(gca,'xtick',[],'ytick',[]);
    sp_idx = sp_idx + 1;
    for j=1:w1_count,
        subplot(w2_count,(w1_count+1),sp_idx);
        imagesc(reshape(W1(w1_idx(j),:),im_size,im_size)');
        set(gca,'xtick',[],'ytick',[]);
        axis square;
        colormap('gray');
        sp_idx = sp_idx + 1;
    end
end

return
end