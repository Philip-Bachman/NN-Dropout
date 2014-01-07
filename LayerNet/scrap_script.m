% SCRAP SCRIPT

L_dev = zeros(1,30);
L_sde = zeros(1,30);
L_raw = zeros(1,30);

fprintf('Testing: ');
for t=1:30,
    Mte1 = LDN_dev.get_drop_masks(size(Xte,1),1);
    Mte2 = LDN_dev.get_drop_masks(size(Xte,1),0);
    Mte = cell(1,LDN_dev.layer_count);
    for i=1:LDN_dev.layer_count,
        Mte{i} = [Mte1{i}; Mte2{i}];
    end
    L = LDN_dev.dev_loss(LDN_dev.struct_weights(),[Xte;Xte],Yte,Mte,size(Xte,1),2);
    L_dev(t) = L(2);
    L = LDN_dev.dev_loss(LDN_sde.struct_weights(),[Xte;Xte],Yte,Mte,size(Xte,1),2);
    L_sde(t) = L(2);
    L = LDN_dev.dev_loss(LDN_raw.struct_weights(),[Xte;Xte],Yte,Mte,size(Xte,1),2);
    L_raw(t) = L(2);
    fprintf('.');
end
fprintf('\n');
fprintf('mean(L_dev): %.4f\n',mean(L_dev));
fprintf('mean(L_sde): %.4f\n',mean(L_sde));
fprintf('mean(L_raw): %.4f\n',mean(L_raw));




