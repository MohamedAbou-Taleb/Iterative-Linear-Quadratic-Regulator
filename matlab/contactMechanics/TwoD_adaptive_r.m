clear all; close all; clc;

%% Parameters
m = 1; 
dt = 0.01;
g = 9.81;
mu = 0.3;
tol = 1e-6;
max_iter = 100;

% Mass Matrix (Constant for this problem, but general formulation)
M = [m 0; 0 m]; 

T = 3; 
tspan = 0:dt:T; 
N = length(tspan);

q0 = [0; 2]; 
u0 = [2; 0]; 

q = zeros(2, N); u = zeros(2, N);
PN = zeros(1, N); PT = zeros(1, N);
q(:,1) = q0; u(:,1) = u0;

% Jacobians (Constant for simple particle)
% W_N: Normal acts on Y (Direction [0; 1])
W_N = [0; 1];
% W_T: Friction acts on X (Direction [1; 0])
W_T = [1; 0];

% Helper Functions
prox_R0minus = @(x) min(0, x);
prox_CT = @(x, lim) max(-lim, min(lim, x));

R_x = @(x, y, qk, uk) [x(1:2) - dt*(x(3:4)+uk);
                       M*x(3:4) - [y(1); 0] - [0; y(2)] + [0; m*g*dt]];
                   
J_x = [eye(2), -dt*eye(2); zeros(2), M];

xk_0 = zeros(4,1); 
yk_0 = zeros(2,1);

%% Simulation Loop
start = tic;
for k = 1:(N-1)
    qk = q(:,k); uk = u(:,k);
    
    % --- STEP 1: Compute Optimal r Vector ---
    % This works for ANY M(q)
    W_all = [W_T, W_N]; % [Tangent, Normal]
    
    % G_diag = diag(W' * M^-1 * W)
    % Since M is diagonal here, it's easy:
    % G_ii = (w_i_x^2 / m) + (w_i_y^2 / m)
    invM = inv(M);
    G_diag = diag(W_all' * invM * W_all);
    
    % Optimal r = 1 / (dt * G_ii)
    r_vec = 1.0 ./ (dt * G_diag);
    
    r_T = r_vec(1);
    r_N = r_vec(2);
    % ----------------------------------------

    % [xk_i, ~,~,~] = newton(@(x) R_x(x, yk_0, qk, uk), @(x) J_x, xk_0, tol, max_iter);
    % [xk_i, ~,~,~] = simplified_newton(@(x) R_x(x, yk_0, qk, uk), J_x, xk_0, tol, max_iter);
    [xk_i, ~,~,~] = fsolve(@(x) R_x(x, yk_0, qk, uk), xk_0);
    yk_i = yk_0;
    
    conv = false;
    iter = 0;
    
    while ~conv && iter < max_iter
        iter = iter + 1;
        
        % 1. Normal Update (Using r_N)
        gap = qk(2) + xk_i(2); 
        PN_new = -prox_R0minus(-yk_i(2) + r_N * gap);
        
        % 2. Friction Update (Using r_T)
        vel_tan = uk(1) + xk_i(3);
        PT_new = -prox_CT(-yk_i(1) + r_T * vel_tan, mu * PN_new);
        
        yk_new = [PT_new; PN_new];
        
        % 3. Solve Dynamics
        xk_new = newton(@(x) R_x(x, yk_new, qk, uk), @(x) J_x, xk_i, tol, max_iter);
        
        if norm(xk_new - xk_i) < tol && norm(yk_new - yk_i) < tol
            conv = true;
        end
        xk_i = xk_new;
        yk_i = yk_new;
    end
    
    dq = xk_i(1:2); du = xk_i(3:4);
    q(:,k+1) = qk + dq; u(:,k+1) = uk + du;
    PT(k+1) = yk_i(1); PN(k+1) = yk_i(2);
    
    xk_0 = xk_i; yk_0 = yk_i;
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

% function [x, f, f_norm, converged] = newton(fun, jac, x0, tol, max_iter)
%     x = x0; converged = false; f_norm = inf; f = fun(x);
%     for k = 1:max_iter
%         if norm(f, inf) < tol, converged = true; return; end
%         x = x - jac(x) \ f;
%         f = fun(x);
%     end
% end