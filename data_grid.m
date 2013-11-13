function [ X Y ] = data_grid( ...
    obs_count, sigma_min, sigma_max, grid_width, class_centers )
% Make a binary-classed data set with class clusters laid out on a grid.
%

if ~exist('grid_width','var')
    grid_width = 3;
end

grid_size = grid_width^2;

if ~exist('class_centers','var')
    class_centers = floor(grid_size / 2);
else
    if (class_centers > (grid_size / 2))
        error('make_grid_data: too many requested class centers.\n');
    end
end

grid_idx = randperm(grid_size);
c0_idx = grid_idx(1:class_centers);
c1_idx = grid_idx(class_centers+1:class_centers+class_centers);

[grid_x grid_y] = meshgrid(1:grid_width,1:grid_width);

grid_sigma = ones(size(grid_x)).*sigma_min + rand(size(grid_x)).*(sigma_max - sigma_min);

X = zeros(obs_count,2);
Y = zeros(obs_count,1);

for i=1:obs_count,
    if (randi(2) == 1)
        Y(i) = -1;
        if (numel(c0_idx) == 1)
            mu_idx = c0_idx(1);
        else
            mu_idx = randsample(c0_idx,1);
        end
    else
        Y(i) = 1;
        if (numel(c1_idx) == 1)
            mu_idx = c1_idx(1);
        else
            mu_idx = randsample(c1_idx,1);
        end
    end
    mu = [grid_x(mu_idx), grid_y(mu_idx)];
    X(i,:) = mu + randn(1,2).*grid_sigma(mu_idx);
end

theta = rand() * 2 * pi;
rot_mat = [cos(theta) -sin(theta); sin(theta) cos(theta)];
X = X * rot_mat;

X = bsxfun(@minus, X, mean(X));
X = bsxfun(@rdivide, X, std(X));

return

end

