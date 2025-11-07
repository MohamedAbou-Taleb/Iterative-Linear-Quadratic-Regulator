clc
clear all
close all
addpath(genpath(pwd))

%%
A_cont = [0, 1; 0, 0];
B_cont = [0; 1];
dt = 0.01;
T = 3;
tspan = 0:dt:T;
N = length(tspan) - 1;
[A, B] = cont2disc(A_cont, B_cont, dt);
[n, m] = size(B);

Q = diag([1, 1]);
R = eye(m);
Q_f = eye(n)*0;

x_0 = [1;0];

U_ff = zeros(m, N);
linear_iLQR = Linear_iLQR_CLASS(A=A, B=B, Q=Q, R=R, Q_f=Q_f, x_0=x_0, U_ff=U_ff, dt=dt, T = T);

for i = 1:100
    linear_iLQR.optimize_trajectory();
end
X_bar = linear_iLQR.X;
U_bar = linear_iLQR.U;

%% Casadi solution


import casadi.*

opti = casadi.Opti();
opts = struct;
opts.ipopt.max_iter = 2000;

opts.ipopt.hessian_approximation = 'limited-memory';

opti.solver('ipopt', opts);

X_bar_casadi = opti.variable(n, N+1);
U_bar_casadi = opti.variable(m, N);

opti.subject_to( X_bar_casadi(:, 1) == x_0 )

cost = 0;
for k = 1:N
    xk = X_bar_casadi(:, k);
    uk = U_bar_casadi(:, k);
    xkPlusOne = A * xk + B * uk;
    opti.subject_to(X_bar_casadi(:, k + 1) == xkPlusOne);
    cost = cost + (0.5*xk'*Q*xk + 0.5*uk'*R*uk)*dt;
end

cost = cost + X_bar_casadi(:, end)'*Q_f*X_bar_casadi(:, end);
sol = opti.solve();

X_bar_casadi = sol.value(X_bar_casadi);
U_bar_casadi = sol.value(U_bar_casadi);

%% --- 4. Plotting ---


figure;
set(gcf, 'Color', 'w');

% Plot States
subplot(3, 1, 1);
plot(tspan, X_bar(1, :), 'b-', 'LineWidth', 2);
hold on;
plot(tspan, X_bar_casadi(1, :), 'r--', 'LineWidth', 2);
title('Optimal State Trajectory');
xlabel('Time (s)');
ylabel('State');
legend('iLQR', 'collocation');
grid on;

subplot(3, 1, 2)
plot(tspan, X_bar(2, :), 'b-', 'LineWidth', 2);
hold on;
plot(tspan, X_bar_casadi(2, :), 'r--', 'LineWidth', 2);
title('Optimal State Trajectory');
xlabel('Time (s)');
ylabel('State');
% legend('x_1 (Position)', 'x_2 (Velocity)');
grid on;

% Plot Control
subplot(3, 1, 3);
% Plot control inputs (only up to time N)
hold on
plot(tspan(1:N), U_bar, 'k-', 'LineWidth', 2);
plot(tspan(1:N), U_bar_casadi, LineStyle="--", LineWidth=2)
title('Optimal Control Input');
xlabel('Time (s)');
ylabel('Control (u)');
grid on;
legend('iLQR', 'collocation');