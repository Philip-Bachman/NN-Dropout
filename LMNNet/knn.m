function [ Y_nn I_nn ] = knn( Xte, Xtr, Ytr, k, do_loo, do_cos )
% Do knn classification of the points in Xte using the points in Xtr/Ytr.
%
% When do_loo == 1, this does "leave-one-out" knn, assuming Xtr == Xte
% When do_cos == 1, this uses max normalized dot-products (i.e. cosine).
%
if ~exist('do_loo','var')
    do_loo = 0;
end
if ~exist('do_cos','var')
    do_cos = 0;
end
if (do_cos == 1)
    Xtr = bsxfun(@rdivide, Xtr, sqrt(sum(Xtr.^2,2))+1e-8);
    Xte = bsxfun(@rdivide, Xte, sqrt(sum(Xte.^2,2))+1e-8);
end
obs_count = size(Xte,1);
Y_nn = zeros(obs_count,k);
I_nn = zeros(obs_count,k);
Xtr_n = sqrt(sum(Xtr.^2,2)+1e-8);
Xtr_s = sum(Xtr.^2,2);
Xte_s = sum(Xte.^2,2);
fprintf('Computing knn:');
for i=1:obs_count,
    if (mod(i,round(obs_count/50)) == 0)
        fprintf('.');
    end
    if (do_cos == 1)
        d = Xtr * Xte(i,:)';
        d = d ./ Xtr_n;
        if (do_loo == 1)
            d(i) = min(d) - 1;
        end
        [d_srt i_srt] = sort(d,'descend');
    else
        d = (Xtr_s + Xte_s(i)) - (2 * (Xtr * Xte(i,:)'));
        if (do_loo == 1)
            d(i) = max(d) + 1;
        end
        [d_srt i_srt] = sort(d,'ascend');
    end
    I_nn(i,:) = i_srt(1:k);
    Y_nn(i,:) = Ytr(I_nn(i,:));
end
fprintf('\n');
return
end

