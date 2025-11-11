classdef System_CLASS < handle & matlab.mixin.Heterogeneous
    %SYSTEM_CLASS Abstract base class for defining a system for iLQR.
    
    properties (SetAccess = protected)
        % These properties MUST be set by the subclass constructor
        n_x       % State dimension
        n_u       % Control dimension
        dt        % Time step (now defined by system)
    end
    
    methods (Abstract)
        % Dynamics
        f = f_fcn(obj, x, u);
        f_x = f_x_fcn(obj, x, u);
        f_u = f_u_fcn(obj, x, u);
        
        % Stage Cost
        l = l_fcn(obj, x, u);
        l_x = l_x_fcn(obj, x, u);
        l_u = l_u_fcn(obj, x, u);
        l_xx = l_xx_fcn(obj, x, u);
        l_ux = l_ux_fcn(obj, x, u);
        l_uu = l_uu_fcn(obj, x, u);
        
        % Terminal Cost
        l_f = l_f_fcn(obj, x);
        l_f_x = l_f_x_fcn(obj, x);
        l_f_xx = l_f_xx_fcn(obj, x);
    end
end