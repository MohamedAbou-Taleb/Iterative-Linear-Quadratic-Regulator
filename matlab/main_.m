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
Q_f = eye(n)*10000;

x_0 = [10;0];

U_ff = zeros(m, N);
linear_iLQR = Linear_iLQR_CLASS(A=A, B=B, Q=Q, R=R, Q_f=Q_f, x_0=x_0, U_ff=U_ff, dt=dt, T = T);

startTime = tic; % Start timing
for i = 1:1
    linear_iLQR.optimize_trajectory();
end
elapsedTime = toc(startTime); % Calculate elapsed time
disp(['Time taken to execute: ', num2str(elapsedTime), ' seconds']);
X_bar = linear_iLQR.X;
U_bar = linear_iLQR.U;

%% Casadi solution


import casadi.*

opti = casadi.Opti();
opts = struct;
opts.ipopt.max_iter = 2000;

opts.ipopt.hessian_approximation = 'limited-memory';

opti.solver('ipopt', opts);

U_bar_casadi = opti.variable(m, N);

% 2. Create symbolic trajectory and cost
cost = 0;
xk = x_0; % Initialize state with the numeric initial condition
X_sim = [xk]; % Store the simulated state trajectory (as an expression)

for k = 1:N
    % Get the control variable for this step
    uk = U_bar_casadi(:, k);
    
    % Add stage cost (based on current state xk and control uk)
    cost = cost + (0.5*xk'*Q*xk + 0.5*uk'*R*uk)*dt;
    
    % Simulate the next state using the dynamics
    % This is now part of the cost/constraint graph, not a constraint itself
    xk_next = A * xk + B * uk;
    
    % Update xk for the next loop iteration
    xk = xk_next;
    
    % Store the simulated state
    X_sim = [X_sim, xk_next];
end

% 3. Add terminal cost (based on the final simulated state)
% xk now holds the expression for the final state X_sim(:, N+1)
cost = cost + xk'*Q_f*xk;

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
X_bar_casadi = sol.value(X_sim);
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