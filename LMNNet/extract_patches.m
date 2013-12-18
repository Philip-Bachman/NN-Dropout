function [ patches ] = extract_patches( ims, w, patch_count, rows, cols, rgb )
% Extract patches from images of dimension rows x cols x rgb. Postprocess
% patches to be zero mean and unit norm.
%
% Parameters:
%   ims: image matrix (im_count x rows*cols*rgb)
%   w: size of square patches to extract
%   patch_count: number of patches to extract
%   rows: number of rows in each image
%   cols: number of columns in each image
%   rgb: number of colors in each image (e.g. 1 or 3)
%
% Outputs:
%   patches: ZMUNed randomly selected patches, each of size (1 x w*w*rgb)
%
im_count = size(ims,1);
sample_count = round(patch_count*1.25);
ws = w - 1;
min_coord = 1;
max_coord = min(rows, cols) - ws;

patches = zeros(sample_count, w*w*rgb);
coords = randi([min_coord max_coord], sample_count, 2);
fprintf('Extracting %d patches:', patch_count);
for i=1:sample_count,
    if (mod(i, max(round(sample_count/50),1)) == 0)
        fprintf('.');
    end
    idx = randi(im_count);
    im = reshape(ims(idx,:),rows,cols,rgb);
    row = coords(i,1);
    col = coords(i,2);
    patch = reshape(im(row:(row+ws),col:(col+ws),:),1,w*w*rgb);
    patches(i,:) = double(patch);
end
fprintf('\n');

patches = bsxfun(@minus, patches, mean(patches,2));
vars = sum(patches.^2,2);
[vals idx] = sort(vars, 'descend');
patches = patches(idx(1:patch_count),:);
patches = ZMUN(patches);
    
return 

end

