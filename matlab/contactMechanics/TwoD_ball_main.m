clear all
close all
clc

%%
m = 1;
dt = 0.01;
g = 9.81;
W_T = [1; 0];
W_N = [0; 1];
W = [W_T, W_N];
M = diag([m, m]); % Mass matrix for the system
tol_newton = 1e-5;
max_iter = 1000;
tol_fixed_point = 1e-5;
mu = 0.1;
T = 3;
tspan = 0:dt:T;
N = length(tspan);

q0 = [0; 1];
u0 = [1; 0];
PN0 = 0;
PT0 = 0;
% Initialize arrays to store results
q = zeros(2, N);
u = zeros(2, N);
PN = zeros(1, N);
PT = zeros(1, N);
q(:, 1) = q0;
u(:, 1) = u0;
PN(:, 1) = PN0;
PT(:, 1) = PT0;
R_x = @(x, y, qk, uk) [x(1:2) - dt*(x(3:4)+uk);
               m*x(3) - y(1);
               m*x(4) + dt*m*g - y(2)];
J_x = @(x, y) [eye(2), -dt*eye(2);
               zeros(2), m*eye(2)];
xk_0 = zeros(4, 1);
yk_0 = zeros(2, 1);
start = tic;
for k = 1:(N-1)
    qk = q(:, k);
    uk = u(:, k);
    [xkPlusOne_0, ~,~, conv_newton] = newton(@(x) R_x(x, yk_0, qk, uk), @(x) J_x(x, yk_0), xk_0, tol_newton, max_iter);
    % [xkPlusOne_0, ~,~, conv_newton] = fsolve(@(x) R_x(x, yk_0, qk, uk), xk_0);

        % --- STEP 1: Compute Optimal r Vector ---
    % This works for ANY M(q)
    
    % G_diag = diag(W' * M^-1 * W)
    % Since M is diagonal here, it's easy:
    % G_ii = (w_i_x^2 / m) + (w_i_y^2 / m)
    invM = inv(M);
    G_diag = diag(W' * invM * W);
    
    % Optimal r = 1 / (dt * G_ii)
    r_vec = 1.0 ./ (dt * G_diag);
    
    r_T = r_vec(1)*dt;
    r_N = r_vec(2);

    conv = 0;
    i = 1;
    ykPlusOne_i = yk_0;
    xkPlusOne_i = xkPlusOne_0;

    while ~conv && i <= max_iter
        dPN_iPlusOne = -prox_R0minus(-ykPlusOne_i(2) + r_N*(qk(2) + xkPlusOne_i(2)));
        dPT_iPlusOne = -prox_CT(-ykPlusOne_i(1) + r_T*(uk(1) + xkPlusOne_i(3)), mu*dPN_iPlusOne);
        ykPlusOne_iPlusOne = [dPT_iPlusOne; dPN_iPlusOne];
        xkPlusOne_iPlusOne = newton(@(x) R_x(x, ykPlusOne_iPlusOne, qk, uk), @(x) J_x(x, ykPlusOne_iPlusOne),...
            xkPlusOne_i, tol_newton, max_iter);
        conv = norm(xkPlusOne_iPlusOne - xkPlusOne_i) < tol_fixed_point;
        xkPlusOne_i = xkPlusOne_iPlusOne;
        ykPlusOne_i = ykPlusOne_iPlusOne;
        i=i+1;
    end
    if ~conv
        warning('fixed point iteration has not converged')
    end
    
    fprintf('Number of iterations for fixed point iteration at step %d: %d\n', k, i-1);
    dq = xkPlusOne_iPlusOne(1:2);
    du = xkPlusOne_iPlusOne(3:4);
    dP = ykPlusOne_iPlusOne;
    q(:, k+1) = qk + dq;
    u(:, k+1) = uk + du;
    PT(:, k+1) = dP(1);
    PN(:, k+1) = dP(2);
    yk_0 = dP;
    xk_0 = xkPlusOne_iPlusOne;
end
disp(['time elapsed: ', num2str(toc(start))])

%% Plotting Results
figure('Color', 'w', 'Name', 'Simulation Dynamics');

% 1. Trajectory (Position Y vs Position X)
% This visualizes the actual path of the ball in space.
subplot(2, 2, 1);
plot(q(1, :), q(2, :), 'b-o', 'MarkerSize', 3, 'LineWidth', 1.0);
yline(0, 'k-', 'LineWidth', 2); % Ground line
title('Trajectory (Position Space)');
xlabel('Distance X (m)');
ylabel('Height Y (m)');
axis equal; grid on;
% Add a little buffer to the Y-limits so the ground is visible
ylim([-0.1, max(q(2,:))*1.1]); 

% 2. Velocities (u_x and u_y vs Time)
% u_x (Red) should step down due to friction.
% u_y (Blue) should show the sawtooth gravity pattern.
subplot(2, 2, 2);
plot(tspan, u(1, :), 'r-', 'LineWidth', 1.5); hold on;
plot(tspan, u(2, :), 'b--', 'LineWidth', 1.0);
title('Velocities');
legend('Horizontal v_x', 'Vertical v_y', 'Location', 'best');
xlabel('Time (s)');
ylabel('m/s');
grid on;

% 3. Normal Percussion (PN)
% The "Bounce" force.
subplot(2, 2, 3);
plot(tspan, PN, 'g-', 'LineWidth', 1.5);
title('Normal Percussion (P_N)');
xlabel('Time (s)');
ylabel('Impulse (N*s)');
grid on;

% 4. Friction Percussion (PT)
% The "Braking" force.
subplot(2, 2, 4);
plot(tspan, PT, 'm-', 'LineWidth', 1.5);
title('Friction Percussion (P_T)');
xlabel('Time (s)');
ylabel('Impulse (N*s)');
grid on;

sgtitle('Complete System State: Position, Velocity, and Forces');