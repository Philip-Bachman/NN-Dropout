function [ X Y ] = data_ring( obs_count, ring_count, width, sigma, ring_class  )
% Generate data for (possibly) alternating class rings
%
% Parameters:
%   ring_count: number of rings to generate
%   width: the thickness of each ring
%   sigma: the variance of point distance from the ring central ring
%   ring_class: (optional) assignment of a class to each ring
%
% Output:
%   X: a set of observations drawn from the ring-like object
%   Y: the class of each observation
%

if ~exist('ring_class','var')
    ring_class = zeros(1,ring_count);
    rc = 1;
    for i=1:ring_count,
        ring_class(i) = rc;
        if (rc == -1)
            rc = 1;
        else
            rc = -1;
        end
    end
end

if (numel(ring_class) ~= ring_count)
    error('make_ring_data: mismatched ring_count and ring_class length.\n');
end 

ring_rads = (1:ring_count) .* width;
ring_weights = zeros(1,numel(ring_rads));
for i=1:numel(ring_rads),
    r = ring_rads(i);
    ring_weights(i) = pi*(r + width)^2 - pi*(r - width)^2;
end
ring_weights = ring_weights ./ sum(ring_weights);
% ring_weights = ring_rads ./  sum(ring_rads);    

X = zeros(obs_count,2);
Y = zeros(obs_count,1);
for i=1:obs_count,
    ring_num = randsample(1:ring_count,1,true,ring_weights);
    % Get a unit vector in a random direction
    vec = randn(1,2);
    vec = vec ./ norm(vec);
    % Get a noise displacement along the random direction
    vec_noise = vec .* (randn() * sigma);
    % Get the final observation vector
    vec = (vec .* ring_rads(ring_num)) + vec_noise;
    % Store observation and class
    X(i,:) = vec;
    Y(i) = ring_class(ring_num);
end

X = ZMUV(X);

return

end

