clc
clear all
close all
addpath(genpath(pwd))

%%
dt = 0.01;
T = 2;
tspan = 0:dt:T;
N = length(tspan) - 1; 

n_x = 2;
n_u = 1;

g = 9.81;
l = 1;
d = 0;
Q = diag([10, 1]);
R = 1*eye(n_u);
Q_f = eye(n_x)*10;

x_target = [pi;0];

pendulum_sys = Pendulum_System_CLASS(g=g, l=l, d=d, dt=dt, Q=Q, R=R, Q_f=Q_f, x_target=x_target);
pendulum_sys_sim = Pendulum_System_CLASS(g=g, l=l, d=d, dt=dt, Q=Q, R=R, Q_f=Q_f, x_target=x_target);

x_0 = [pi*0;0];



U_ff = zeros(n_u, N);
tol = 1e-6;
maxiter = 10;

iLQR= iLQR_CLASS( ...
    U_init=U_ff, ...
    T = T, ...
    x_0 = x_0, ...
    system=pendulum_sys, ...
    tol=tol, ...
    maxiter=maxiter...
    );


T_sim = 10;
tspan_sim = 0:dt:T_sim;
N_sim = length(tspan_sim)-1;
X_sim = zeros(n_x, N_sim + 1);
X_sim(:, 1) = x_0;
U_sim = zeros(n_u, N_sim);
start = tic;
for k = 1:N_sim
    xk = X_sim(:, k);
    iLQR.x_0 = xk;

    iLQR.optimize_trajectory();
    X_bar = iLQR.X;
    U_bar = iLQR.U;

    % warm start
    iLQR.U = [U_bar(:, 2:end), U_bar(:, end)];
    
    uk = U_bar(:, 1);

    xkPlusOne = pendulum_sys_sim.f_fcn(xk, uk);

    U_sim(:, k) = uk;
    X_sim(:, k+1) = xkPlusOne;
end
elapsed_time = toc(start);
disp(['elapsed time for MPC simulation: ', num2str(elapsed_time), ' seconds'])
%% --- 4. Plotting ---


figure;
set(gcf, 'Color', 'w');

% Plot States
subplot(3, 1, 1);
plot(tspan_sim, X_sim(1, :), 'b-', 'LineWidth', 2);
hold on;
plot([0, T_sim], [x_target(1), x_target(1)], LineWidth=2);
title('Closed loop trajectory');
xlabel('Time (s)');
ylabel('State');
grid on;

subplot(3, 1, 2)
plot(tspan_sim, X_sim(2, :), 'b-', 'LineWidth', 2);
hold on;
plot([0, T_sim], [x_target(2), x_target(2)], LineWidth=2);
xlabel('Time (s)');
ylabel('State');
grid on;

% Plot Control
subplot(3, 1, 3);
% Plot control inputs (only up to time N)
hold on
plot(tspan_sim(1:N_sim), U_sim, 'k-', 'LineWidth', 2);
title('Optimal Control Input');
xlabel('Time (s)');
ylabel('Control (u)');
grid on;