classdef LNLayer < handle

    properties
        act_func
        weights
    end % END PROPETIES
    
    methods (Abstract)
        % feedforward computes a feedforward activation 
        A = feedforward( X )
        
        [dLdW dLdX] = backprop( dLdA )
        
        
        
    end
end





%%%%%%%%%%%%%%
% EYE BUFFER %
%%%%%%%%%%%%%%
