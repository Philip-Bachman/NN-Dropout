function [ fig ] = plot_netfunc( X, Y, net, grid_res, grid_buffer, fig )
%

assert((size(X,2) == 2), 'Inputs in X must be 2d.');
assert((size(Y,2) == 1), 'Outputs in Y must be 1d.');
assert((net.layers{1}.dim_input == 3), 'net must take 2d inputs (+ a bias).');
assert((net.layers{end}.dim_output == 1), 'net must make 1d outputs.');

% Compute the extremal coordinates of the evaluation grid
x_min = min(X(:,1)) - grid_buffer;
x_max = max(X(:,1)) + grid_buffer;
y_min = min(X(:,2)) - grid_buffer;
y_max = max(X(:,2)) + grid_buffer;

% Compute a suitable set of grid points at which to evaluate the trees
[ Xg Yg ] = meshgrid(...
    linspace(x_min, x_max, grid_res), linspace(y_min, y_max, grid_res));
Fg = zeros(size(Xg));
fprintf('Computing values for grid:');
for col=1:grid_res,
    if (mod(col,round(max(1.0,grid_res/60))) == 0)
        fprintf('.');
    end
    col_points = [Xg(:,col) Yg(:,col)];
    Y_col = net.evaluate(col_points);
    Fg(:,col) = Y_col;
end
fprintf('\n');

fprintf('Plotting the learned function...\n');

% Setup figure and axes
if ~exist('fig','var')
    fig = figure();
else
    figure(fig);
    cla;
    axis auto;
end
hold on;

% Temporary stuff for surface plotting.
colormap('jet');
surfc(Xg, Yg, Fg);
axis square;
axis equal;
colorbar();

% figure();
% hold on;
% plot(Fg(:,1),'b-');
% plot(Fg(1,:),'r-');

return

end



