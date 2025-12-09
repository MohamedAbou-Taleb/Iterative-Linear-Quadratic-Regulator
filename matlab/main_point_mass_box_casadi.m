clear all

close all

clc

%%

m_ball = 1;

m_box = 0.5;

box_width = 0.5;

box_height = 0.3;

mu_box = 0.1;

mu_ball = 0.5;

g = 9.81;

M = blkdiag(eye(4)*m_ball, eye(2)*m_box);

invM = inv(M);

w_N1 = [-1; 0; 0; 0; 1; 0];

w_N2 = [0; 0; 1; 0; -1; 0];

w_N3 = [0; 0; 0; 0; 0; 1];

w_T1 = [0; -1; 0; 0; 0; 1];

w_T2 = [0; 0; 0; -1; 0; 1];

w_T3 = [0; 0; 0; 0; 1; 0];

W = [w_T1, w_N1, w_T2, w_N2, w_T3, w_N3];

f_g = [0; 0; 0; 0; 0; -m_box*g];

ball_radius = 0.01;

gap_fcn = @(q) [q(5)-q(1)-ball_radius; q(3)-q(5)-ball_radius; q(6)-box_height/2];

dynamics_fun = @(q, v, lambda) [v; invM*(f_g + W*lambda)];

dt = 0.01;

T = 2;

tspan = 0:dt:T;

N = length(tspan)-1;

q_0 = [-1; box_height/2;

1; box_height/2;

0; 4*box_height/2];

v_0 = [0; 0; 0; 0; 1.0; 0];

x_0 = [q_0; v_0];

n_q = 6;

n_v = 6;

n_x = n_q + n_v;

n_la = length(gap_fcn(q_0))*2;

rho = 1e-3;

%%

import casadi.*

opti = casadi.Opti();

opts = struct;

opts.ipopt.max_iter = 2000;

opts.ipopt.hessian_approximation = 'limited-memory';

opti.solver('ipopt', opts);

x = opti.parameter(n_x, 1);

xPlus = opti.variable(n_x, 1);

lambda = opti.variable(n_la,1);

q = x(1:n_q);

v = x(n_q+1:end);

qPlus = xPlus(1:n_q);

vPlus = xPlus(n_q+1:end);

lambda_N1 = lambda(2);

lambda_N2 = lambda(4);

lambda_N3 = lambda(6);

lambda_T1 = lambda(1);

lambda_T2 = lambda(3);

lambda_T3 = lambda(5);

opti.subject_to( xPlus == x + dt*dynamics_fun(qPlus, vPlus, lambda) )

opti.subject_to( lambda_N1 >= 0 )

opti.subject_to( lambda_N2 >= 0 )

opti.subject_to( lambda_N3 >= 0 )

opti.subject_to( -mu_ball*lambda_N1 <= lambda_T1 )

opti.subject_to( mu_ball*lambda_N1 >= lambda_T1 )

opti.subject_to( -mu_ball*lambda_N2 <= lambda_T2 )

opti.subject_to( mu_ball*lambda_N2 >= lambda_T2 )

opti.subject_to( -mu_box*lambda_N3 <= lambda_T3 )

opti.subject_to( mu_box*lambda_N3 >= lambda_T3 )

g_NPlus = gap_fcn(qPlus);

opti.subject_to( g_NPlus >= zeros(3,1) )

opti.subject_to( g_NPlus(1)*lambda_N1 == rho )

opti.subject_to( g_NPlus(2)*lambda_N2 == rho )

opti.subject_to( g_NPlus(3)*lambda_N3 == rho )

% opti.minimize( lambda'*lambda )

opti.minimize( (W(:, [1,3,5])'*vPlus)'*lambda([1,3,5]) )

X = zeros(n_x, N+1);

X(:, 1) = x_0;

Lambda = zeros(n_la, N+1);

run_times = zeros(1, N);

for k = 1:N

    tic; % Start timer

    xk = X(:, k);

    opti.set_value(x, xk);

    opti.set_initial(xPlus, xk);

    opti.set_initial(lambda, Lambda(:, k));

    sol = opti.solve();

    xPlus_sol = sol.value(xPlus);

    lambda_sol = sol.value(lambda);

    X(:, k+1) = xPlus_sol;

    Lambda(:, k+1) = lambda_sol;

    run_times(k) = toc; % Record elapsed time

end

average_run_time = mean(run_times);

disp(['Average run time: ', num2str(average_run_time), ' seconds']);

%% plot the results

figure()

plot(tspan, X(6,:))
xlabel('time [s]')
ylabel('y_{box}')
title('box height')

figure()
plot(tspan, X(5,:))
xlabel('time [s]')
ylabel('x_{box}')
title('box horizontal displacement')

figure()
plot(tspan, Lambda(5,:))
xlabel('time [s]')
ylabel('lambda_{T,box}')
title('Tangential force box')

figure()
plot(tspan, Lambda(6,:))
xlabel('time [s]')
ylabel('lambda_{N,box}')
title('Normal force box')