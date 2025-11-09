clc
clear all
close all
addpath(genpath(pwd))

%%
dt = 0.01;
T = 5;
tspan = 0:dt:T;
N = length(tspan) - 1; 

n = 2;
m = 1;

f_fcn = @(x, u) [x(1) + dt*x(2); x(2) - sin(x(1))*dt + u*dt];
f_x_fcn = @(x, u) [1, dt; -cos(x(1))*dt, 1];
f_u_fcn = @(x, u) [0; dt];

Q = diag([1, 1]);
R = eye(m);
Q_f = eye(n)*0;

x_0 = [0;0];
x_target = [pi;0];

l_fcn = @(x, u) 0.5 * ((x-x_target)' * Q * (x-x_target) + u' * R * u)*dt;
l_x_fcn = @(x, u) (x-x_target)'*Q*dt;
l_u_fcn = @(x, u) u'*R*dt;
l_xx_fcn = @(x, u) Q*dt;
l_ux_fcn = @(x, u) zeros(m, n);
l_uu_fcn = @(x, u) R*dt;

l_f_fcn = @(x) 0.5 * (x-x_target)' * Q_f * (x-x_target);
l_f_x_fcn = @(x) (x-x_target)'*Q_f;
l_f_xx_fcn = @(x) Q_f;

U_ff = zeros(m, N);
tol = 1e-6;
maxiter = 100;

iLQR= iLQR_CLASS( ...
        dt = dt, ...
        T = T, ...
        x_0 = x_0, ...
        U_ff = U_ff, ...
        f_fcn = f_fcn, ...
        f_x_fcn = f_x_fcn, ...
        f_u_fcn = f_u_fcn, ...
        l_fcn = l_fcn, ...
        l_x_fcn = l_x_fcn, ...
        l_u_fcn = l_u_fcn, ...
        l_xx_fcn = l_xx_fcn, ...
        l_ux_fcn = l_ux_fcn, ...
        l_uu_fcn = l_uu_fcn, ...
        l_f_fcn = l_f_fcn, ...
        l_f_x_fcn = l_f_x_fcn, ...
        l_f_xx_fcn = l_f_xx_fcn, ...
        tol=tol, ...
        maxiter=maxiter...
    );

startTime = tic; % Start timing
for i = 1:1
    iLQR.optimize_trajectory();
end
elapsedTime = toc(startTime); % Calculate elapsed time
disp(['Time taken to execute: ', num2str(elapsedTime), ' seconds']);
X_bar = iLQR.X;
U_bar = iLQR.U;

%% Casadi solution


import casadi.*

opti = casadi.Opti();
opts = struct;
opts.ipopt.max_iter = 2000;

opts.ipopt.hessian_approximation = 'limited-memory';

opti.solver('ipopt', opts);

U_bar_casadi = opti.variable(m, N);
X_bar_casadi = opti.variable(n, N+1);
opti.subject_to( X_bar_casadi(:, 1) == x_0 )
% 2. Create symbolic trajectory and cost
cost = 0;

for k = 1:N
    % Get the control variable for this step
    uk = U_bar_casadi(:, k);
    xk = X_bar_casadi(:, k);
    % Add stage cost (based on current state xk and control uk)
    cost = cost + l_fcn(xk, uk);
    
    % Simulate the next state using the dynamics
    % This is now part of the cost/constraint graph, not a constraint itself
    xkPlusOne = f_fcn(xk, uk);
    opti.subject_to( X_bar_casadi(:, k+1) == xkPlusOne )
end

% 3. Add terminal cost (based on the final simulated state)
% xk now holds the expression for the final state X_sim(:, N+1)
cost = cost + l_f_fcn(X_bar_casadi(:, end));

% 4. Set the objective
opti.minimize(cost);

% (Note: The initial state x_0 is enforced by starting the simulation
% from it. The dynamics are enforced by being used to build the
% cost function. There are no other constraints.)

% 5. Solve the problem
startTime_casadi = tic; % Start timing for Casadi solver
sol = opti.solve();
elapsedTime_casadi = toc(startTime_casadi); % Calculate elapsed time for Casadi solver

% 6. Retrieve results
U_bar_casadi = sol.value(U_bar_casadi);
% We get the state trajectory by evaluating the symbolic
% expression 'X_sim' with the optimal control values
X_bar_casadi = sol.value(X_bar_casadi);
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

disp(['Time taken to execute iLQR: ', num2str(elapsedTime), ' seconds']);
disp(['Time taken to execute collocation: ', num2str(elapsedTime_casadi), ' seconds']);