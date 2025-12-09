clear all
close all
clc

%% 1. Physics Parameters
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
dt = 0.01;
rho = 1e-3;

% Initial State
q_0 = [-1; box_height/2; 1; box_height/2; 0; 4*box_height/2];
v_0 = [0; 0; 0; 0; 1.0; 0];
x_0 = [q_0; v_0];

n_q = 6; n_v = 6; n_x = n_q + n_v;
n_la = 6; % 3 contacts * 2 (Normal + Tangential)

%% 2. Define Problem with SX (Standard CasADi)
import casadi.*

% --- Symbolic Variables ---
p = SX.sym('p', n_x);        % Parameter: x_k (current state)
x = SX.sym('x', n_x + n_la); % Decision: [x_k+1; lambda]

% Split decision vector for readability
xPlus = x(1:n_x);
lambda = x(n_x+1:end);

qPlus = xPlus(1:n_q);
vPlus = xPlus(n_q+1:end);

% Previous state (Parameter)
q_prev = p(1:n_q);
v_prev = p(n_q+1:end);

% Unpack Lambda
lam_T1 = lambda(1); lam_N1 = lambda(2);
lam_T2 = lambda(3); lam_N2 = lambda(4);
lam_T3 = lambda(5); lam_N3 = lambda(6);

% --- Dynamics & Geometry Functions ---
gap_fcn = [qPlus(5)-qPlus(1)-ball_radius; ...
           qPlus(3)-qPlus(5)-ball_radius; ...
           qPlus(6)-box_height/2];
       
dynamics = xPlus - (p + dt * [vPlus; invM*(f_g + W*lambda)]);

% --- Constraints (g) ---
% We stack all equality and inequality constraints into one vector 'g'
% 1. Dynamics (Equality)
g_eq_dyn = dynamics;

% 2. Friction Cone (Inequalities handled as: lb <= Expression <= ub)
g_fric_1 = [lam_T1 + mu_ball*lam_N1; -lam_T1 + mu_ball*lam_N1];
g_fric_2 = [lam_T2 + mu_ball*lam_N2; -lam_T2 + mu_ball*lam_N2];
g_fric_3 = [lam_T3 + mu_box*lam_N3;  -lam_T3 + mu_box*lam_N3];

% 3. Gap Condition (Inequality: gap >= 0)
g_gap = gap_fcn;

% 4. Complementarity Relaxation (Equality: gap * lambda == rho)
g_compl = [gap_fcn(1)*lam_N1 - rho;
           gap_fcn(2)*lam_N2 - rho;
           gap_fcn(3)*lam_N3 - rho];

% Combine all 'g' functions
g = [g_eq_dyn; g_fric_1; g_fric_2; g_fric_3; g_gap; g_compl];

% --- Objective (f) ---
f = (W(:, [1,3,5])'*vPlus)' * lambda([1,3,5]);

% --- NLP Structure ---
nlp = struct('x', x, 'p', p, 'f', f, 'g', g);

%% 3. Setup Solver & Sensitivity Factory
opts = struct;
opts.ipopt.print_level = 0;
opts.ipopt.max_iter = 2000;
opts.ipopt.hessian_approximation = 'exact'; % Required for sensitivity

solver = nlpsol('solver', 'ipopt', nlp, opts);

% Create Sensitivity Function
% We want Jacobian of Decision 'x' w.r.t Parameter 'p'
sensitivity = solver.factory('sensitivity', {'p', 'x0', 'lam_x0', 'lam_g0'}, {'jac:x:p'});

%% 4. Define Bounds (lbg, ubg, lbx, ubx)
% Helper to construct bounds vectors
n_dyn = length(g_eq_dyn);
n_fric = length(g_fric_1) * 3;
n_gap = 3;
n_compl = 3;

% Constraints Bounds
lbg = [zeros(n_dyn,1); zeros(n_fric,1); zeros(n_gap,1); zeros(n_compl,1)];
ubg = [zeros(n_dyn,1); inf(n_fric,1);   inf(n_gap,1);   zeros(n_compl,1)];

% Variable Bounds
% xPlus: -inf to inf
% lambda: Normal forces >= 0, Tangential unbounded (handled by friction cone constraints)
lbx = [-inf(n_x,1); -inf; 0; -inf; 0; -inf; 0]; 
ubx = inf(n_x + n_la, 1);

%% 5. Simulation Loop
T_sim = 2;
N = length(0:dt:T_sim)-1;

X_hist = zeros(n_x, N+1);
X_hist(:, 1) = x_0;

% Jacobians Storage: d(xPlus)/d(x_current)
% Since decision 'x' includes [xPlus; lambda], we extract the top-left block later
Jac_hist = zeros(n_x, n_x, N); 

% Initial Guesses
w0 = zeros(n_x + n_la, 1);
w0(1:n_x) = x_0; 
lam_x0 = zeros(length(w0), 1);
lam_g0 = zeros(length(lbg), 1);

fprintf('Starting Simulation...\n');
for k = 1:N
    
    p_val = X_hist(:, k);
    
    % A. Solve NLP
    arg = struct('p', p_val, 'x0', w0, 'lbx', lbx, 'ubx', ubx, 'lbg', lbg, 'ubg', ubg, ...
                 'lam_x0', lam_x0, 'lam_g0', lam_g0);
    sol = solver(arg);
    
    % Extract Solution
    w_opt = full(sol.x);
    lam_x_opt = full(sol.lam_x);
    lam_g_opt = full(sol.lam_g);
    
    xPlus_val = w_opt(1:n_x);
    X_hist(:, k+1) = xPlus_val;
    
    % B. Compute Gradient
    % Pass the OPTIMAL duals (lam_x_opt, lam_g_opt) to the sensitivity factory
    res = sensitivity('p', p_val, 'x0', w_opt, 'lam_x0', lam_x_opt, 'lam_g0', lam_g_opt);
    
    % res.jac_x_p is (n_dec_vars) x (n_params)
    % We only care about d(xPlus)/d(p), which is the top n_x rows
    J_full = full(res.jac_x_p);
    Jac_hist(:, :, k) = J_full(1:n_x, :);
    
    % Warm Start for next iter
    w0 = w_opt;
    lam_x0 = lam_x_opt;
    lam_g0 = lam_g_opt;
end

fprintf('Done.\n');

%% 6. Plotting
tspan = 0:dt:T_sim;
figure; 
subplot(2,1,1); plot(tspan, X_hist(6,:)); title('Box Y Position'); grid on;
subplot(2,1,2); plot(tspan(1:end-1), squeeze(Jac_hist(6,6,:))); title('Sensitivity d(y_{box}^+) / d(y_{box})'); grid on;