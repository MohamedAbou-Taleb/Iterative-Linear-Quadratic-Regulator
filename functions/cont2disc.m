function [A_discrete, B_discrete] = cont2disc(A_cont, B_cont, dt)
% discretize linear system assuming piecewise constant control inputs
n_x = length(A_cont);
n_u = size(B_cont, 2);
H_cont = [A_cont, B_cont; zeros(n_u, n_x + n_u)];
H_discrete = expm(H_cont*dt);
A_discrete = H_discrete(1:n_x, 1:n_x);
B_discrete = H_discrete(1:n_x, n_x+1:end);
end

