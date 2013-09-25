function [ L dXc dXn dXf ] = lmnn_grads_euc( Xc, Xn, Xf, margin )
    % Compute gradients of standard LMNN using Euclidean distance.
    %
    % In addition to the standard penalty on margin transgression by
    % impostor neighbors, impose an attractive penalty on distance between
    % true neighbors and a repulsive penalty between false neighbors.
    %
    % Parameters:
    %   Xc: central/source points
    %   Xn: points that should be closer to those in Xc
    %   Xf: points that should be further from those in Xc
    %   margin: desired margin between near/far distances w.r.t. Xc
    % Outputs:
    %   L: loss for each LMNN triplet (Xc(i,:),Xn(i,:),Xf(i,:))
    %   dXc: gradient of L w.r.t. Xc
    %   dXn: gradient of L w.r.t. Xn
    %   dXf: gradient of L w.r.t. Xf
    %

    On = Xn - Xc;
    Of = Xf - Xc;
    % Compute (squared) norms of the offsets On/Of
    Dn = sum(On.^2,2);
    Df = sum(Of.^2,2);
    % Add a penalty on neighbor distances > margin.
    yes_pen = max(0, Dn - margin);
    % Add a penalty on non-neighbor distances < margin.
    non_pen = max(0, margin - Df);
    % Get losses and indicators for violated LMNN constraints
    m_viol = max(0, (Dn - Df) + margin);
    L = (0.5 * m_viol) + (0.5 * yes_pen) + (0.5 * non_pen);
    % Compute gradients for violated constraints
    dXn = bsxfun(@times,On,(m_viol > 1e-10)) + ...
        bsxfun(@times,On,(yes_pen > 1e-10));
    dXf = bsxfun(@times,-Of,(m_viol > 1e-10)) + ...
        bsxfun(@times,-Of,(non_pen > 1e-10));
    dXc = -dXn - dXf;
    % Clip gradients
    dXc = max(-2, min(dXc, 2));
    dXn = max(-2, min(dXn, 2));
    dXf = max(-2, min(dXf, 2));
    return
end
    
    
function [ L dXc dXn dXf ] = lmnn_grads_dot( Xc, Xn, Xf, margin )
    % Compute gradients of standard LMNN using dot-product distance.
    %
    % Parameters:
    %   Xc: central/source points
    %   Xn: points that should be closer to those in Xc
    %   Xf: points that should be further from those in Xc
    %   margin: desired margin between near/far distances w.r.t. Xc
    % Outputs:
    %   L: loss for each LMNN triplet (Xc(i,:),Xn(i,:),Xf(i,:))
    %   dXc: gradient of L w.r.t. Xc
    %   dXn: gradient of L w.r.t. Xn
    %   dXf: gradient of L w.r.t. Xf
    %
    % Compute dot-products between center/near and center/far points
    Dn = sum(Xc .* Xn,2);
    Df = sum(Xc .* Xf,2);
    % Get losses and indicators for violated LMNN constraints
    m_viol = max(0, (Df - Dn) + margin);
    m_mask = m_viol > 1e-10;
    L = m_viol;
    % Compute gradients for violated constraints
    dXn = bsxfun(@times, -Xc, m_mask);
    dXf = bsxfun(@times, Xc, m_mask);
    dXc = bsxfun(@times, (Xf - Xn), m_mask);
    return
end