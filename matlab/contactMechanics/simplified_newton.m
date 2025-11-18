function [x,f,dfdx,conv,k] = simplified_newton(fun, dfdx,x0,tol,maxiter)
% Newton's method to find zeros
%
% Input
%  fun     : function which gives f and dfdx
%  x0      : initial guess
%  tol     : tolerance
%  maxiter : maximum number of iterations
%
% Output
%  x       : solution for which f(x) = 0 within tolerance
%  f       : function value f(x)
%  dfdx    : Jacobian matrix
%  conv    : boolean to indicate convergence
%  k       : number of iterations
x = x0;
k = 0;
conv = 0;
while ~conv && k<maxiter
    f = fun(x);
    if norm(f) < tol %norm of f is smaller than tol
        conv = 1; % converged
    else    
        % increment k
        k = k+1;
        % update x with dx
        dx = -dfdx\f;
        x = x + dx;
    end
end