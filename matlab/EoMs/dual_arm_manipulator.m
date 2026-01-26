clc
clear all
close all

% ---------------------------------------------------------
% 1. System Definitions and Parameters
% ---------------------------------------------------------

% Generalized coordinates:
% q(1:3): Left Arm joints (Relative angles)
% q(4:6): Right Arm joints (Relative angles)
% q(7:9): Box (x, y, alpha)
syms q [9, 1] real
syms dqdt [9, 1] real

% Actuation Torques (R6)
% tau(1:3): Left Arm (Shoulder, Elbow, Wrist)
% tau(4:6): Right Arm (Shoulder, Elbow, Wrist)
syms tau [6, 1] real

% Arm Geometry & Mass Parameters
syms l1 l2 real             % Link lengths
syms lc1 lc2 real           % CoM distances
syms m1 m2 m_EE real        % Masses

% Inertias (Named theta)
syms theta1 theta2 theta_EE real 

% Box Parameters
syms w_box h_box real
syms m_box theta_box real   % theta_box is the inertia

% EE Geometry
syms w_EE h_EE real

% Gravity
syms g real

% Base positions
syms x_base_L y_base_L real
syms x_base_R y_base_R real

% ---------------------------------------------------------
% 2. Forward Kinematics (Arms)
% ---------------------------------------------------------

A_IB_fcn = @(angle) [cos(angle), -sin(angle); sin(angle), cos(angle)];

% --- LEFT ARM (q1-q3) ---
alpha_L1 = q(1);
alpha_L2 = q(1) + q(2);
alpha_L3 = q(1) + q(2) + q(3);

pos_cm_L1 = [x_base_L; y_base_L] + A_IB_fcn(alpha_L1) * [lc1; 0];
pos_joint_L2 = [x_base_L; y_base_L] + A_IB_fcn(alpha_L1) * [l1; 0];
pos_cm_L2 = pos_joint_L2 + A_IB_fcn(alpha_L2) * [lc2; 0];
pos_wrist_L = pos_joint_L2 + A_IB_fcn(alpha_L2) * [l2; 0];

x_EE1 = pos_wrist_L(1);
y_EE1 = pos_wrist_L(2);
alpha_EE1 = alpha_L3;
pos_EE1 = [x_EE1; y_EE1];

% --- RIGHT ARM (q4-q6) ---
alpha_R1 = q(4);
alpha_R2 = q(4) + q(5);
alpha_R3 = q(4) + q(5) + q(6);

pos_cm_R1 = [x_base_R; y_base_R] + A_IB_fcn(alpha_R1) * [lc1; 0];
pos_joint_R2 = [x_base_R; y_base_R] + A_IB_fcn(alpha_R1) * [l1; 0];
pos_cm_R2 = pos_joint_R2 + A_IB_fcn(alpha_R2) * [lc2; 0];
pos_wrist_R = pos_joint_R2 + A_IB_fcn(alpha_R2) * [l2; 0];

x_EE2 = pos_wrist_R(1);
y_EE2 = pos_wrist_R(2);
alpha_EE2 = alpha_R3;
pos_EE2 = [x_EE2; y_EE2];

% --- BOX (q7-q9) ---
x_box = q(7);
y_box = q(8);
alpha_box = q(9);
I_r_Obox = [x_box; y_box];

% ---------------------------------------------------------
% 3. Contact Kinematics Setup
% ---------------------------------------------------------

A_IB_EE1 = A_IB_fcn(alpha_EE1);
A_IB_EE2 = A_IB_fcn(alpha_EE2);
A_IB_box = A_IB_fcn(alpha_box);

I_r_OEE1 = pos_EE1;
I_r_OEE2 = pos_EE2;

I_e_x_B_EE1 = A_IB_EE1(:, 1); I_e_y_B_EE1 = A_IB_EE1(:, 2);
I_e_x_B_EE2 = A_IB_EE2(:, 1); I_e_y_B_EE2 = A_IB_EE2(:, 2);
I_e_x_B_box = A_IB_box(:, 1); I_e_y_B_box = A_IB_box(:, 2);

I_r_Obox_left = I_r_Obox - w_box/2*I_e_x_B_box;
I_r_Obox_right = I_r_Obox + w_box/2*I_e_x_B_box;
I_r_Obox_bottom_left = I_r_Obox_left - h_box/2*I_e_y_B_box;
I_r_Obox_bottom_right = I_r_Obox_right - h_box/2*I_e_y_B_box;

B_EE1_r_SP_upper1 = [w_EE/2; h_EE/2];
I_r_SP_upper1 = A_IB_EE1 * B_EE1_r_SP_upper1;
B_EE1_r_SP_lower1 = [w_EE/2; -h_EE/2];
I_r_SP_lower1 = A_IB_EE1 * B_EE1_r_SP_lower1;

B_EE2_r_SP_upper2 = [-w_EE/2; h_EE/2];
I_r_SP_upper2 = A_IB_EE2 * B_EE2_r_SP_upper2;
B_EE2_r_SP_lower2 = [-w_EE/2; -h_EE/2];
I_r_SP_lower2 = A_IB_EE2 * B_EE2_r_SP_lower2;

I_r_OP_upper1 = I_r_OEE1 + I_r_SP_upper1;
I_r_OP_lower1 = I_r_OEE1 + I_r_SP_lower1;
I_r_OP_upper2 = I_r_OEE2 + I_r_SP_upper2;
I_r_OP_lower2 = I_r_OEE2 + I_r_SP_lower2;

I_r_P_upper1_box_left = I_r_Obox_left - I_r_OP_upper1;
I_r_P_lower1_box_left = I_r_Obox_left - I_r_OP_lower1;
I_r_P_upper2_box_right = I_r_Obox_right - I_r_OP_upper2;
I_r_P_lower2_box_right = I_r_Obox_right - I_r_OP_lower2;

I_n_box_left = -I_e_x_B_box; I_n_box_right = I_e_x_B_box;
I_n_ground_left = [0; 1]; I_n_ground_right = [0; 1];

% ---------------------------------------------------------
% 4. Contact Jacobians (W)
% ---------------------------------------------------------

g_N_upper1 = simplify(I_n_box_left' * (-I_r_P_upper1_box_left));
g_N_lower1 = simplify(I_n_box_left' * (-I_r_P_lower1_box_left));
g_N_upper2 = simplify(I_n_box_right' * (-I_r_P_upper2_box_right));
g_N_lower2 = simplify(I_n_box_right' * (-I_r_P_lower2_box_right));
g_N_ground_left = I_n_ground_left' * I_r_Obox_bottom_left;
g_N_ground_right = I_n_ground_right' * I_r_Obox_bottom_right;

w_N_upper1 = jacobian(g_N_upper1, q)'; w_N_lower1 = jacobian(g_N_lower1, q)';
w_N_upper2 = jacobian(g_N_upper2, q)'; w_N_lower2 = jacobian(g_N_lower2, q)';
w_N_ground_left = jacobian(g_N_ground_left, q)'; w_N_ground_right = jacobian(g_N_ground_right, q)';

I_t_left = I_e_y_B_box; I_t_right = I_e_y_B_box; I_t_ground = [1; 0];

I_v_box_left = jacobian(I_r_Obox_left, q)*dqdt;
I_v_box_right = jacobian(I_r_Obox_right, q)*dqdt;
I_v_P_upper1 = jacobian(I_r_OP_upper1, q)*dqdt;
I_v_P_lower1 = jacobian(I_r_OP_lower1, q)*dqdt;
I_v_P_upper2 = jacobian(I_r_OP_upper2, q)*dqdt;
I_v_P_lower2 = jacobian(I_r_OP_lower2, q)*dqdt;
I_v_box_bottom_left = jacobian(I_r_Obox_bottom_left, q)*dqdt;
I_v_box_bottom_right = jacobian(I_r_Obox_bottom_right, q)*dqdt;

gamma_T_upper1 = simplify( I_t_left'* (I_v_box_left - I_v_P_upper1) );
gamma_T_lower1 = simplify( I_t_left'* (I_v_box_left - I_v_P_lower1) );
gamma_T_upper2 = simplify( I_t_right'* (I_v_box_right - I_v_P_upper2) );
gamma_T_lower2 = simplify( I_t_right'* (I_v_box_right - I_v_P_lower2) );
gamma_T_ground_left = simplify( I_t_ground' * I_v_box_bottom_left );
gamma_T_ground_right = simplify( I_t_ground' * I_v_box_bottom_right );

w_T_upper1 = simplify( jacobian(gamma_T_upper1, dqdt)' );
w_T_lower1 = simplify( jacobian(gamma_T_lower1, dqdt)' );
w_T_upper2 = simplify( jacobian(gamma_T_upper2, dqdt)' );
w_T_lower2 = simplify( jacobian(gamma_T_lower2, dqdt)' );
w_T_ground_left = simplify( jacobian(gamma_T_ground_left, dqdt)' );
w_T_ground_right = simplify( jacobian(gamma_T_ground_right, dqdt)' );

W = [w_T_upper1, w_N_upper1, w_T_lower1, w_N_lower1, ...
     w_T_upper2, w_N_upper2, w_T_lower2, w_N_lower2, ...
     w_T_ground_left, w_N_ground_left, w_T_ground_right, w_N_ground_right];
W_dot_transpose_dqdt = jacobian(W'*dqdt, q) * dqdt;
g_N = [g_N_upper1; g_N_lower1; g_N_upper2; g_N_lower2; g_N_ground_left; g_N_ground_right];

% ---------------------------------------------------------
% 5. Dynamics (Mass Matrix and Forces)
% ---------------------------------------------------------

% Velocities and Energy
v_cm_L1 = jacobian(pos_cm_L1, q) * dqdt;
v_cm_L2 = jacobian(pos_cm_L2, q) * dqdt;
v_cm_EE1 = jacobian(pos_EE1, q) * dqdt;
v_cm_R1 = jacobian(pos_cm_R1, q) * dqdt;
v_cm_R2 = jacobian(pos_cm_R2, q) * dqdt;
v_cm_EE2 = jacobian(pos_EE2, q) * dqdt;
v_cm_box = jacobian(I_r_Obox, q) * dqdt;

omega_L1 = jacobian(alpha_L1, q) * dqdt;
omega_L2 = jacobian(alpha_L2, q) * dqdt;
omega_EE1 = jacobian(alpha_L3, q) * dqdt;
omega_R1 = jacobian(alpha_R1, q) * dqdt;
omega_R2 = jacobian(alpha_R2, q) * dqdt;
omega_EE2 = jacobian(alpha_R3, q) * dqdt;
omega_box = jacobian(alpha_box, q) * dqdt;

T_total = 0.5*m1*(v_cm_L1.'*v_cm_L1) + 0.5*theta1*omega_L1^2 + ...
          0.5*m2*(v_cm_L2.'*v_cm_L2) + 0.5*theta2*omega_L2^2 + ...
          0.5*m_EE*(v_cm_EE1.'*v_cm_EE1) + 0.5*theta_EE*omega_EE1^2 + ...
          0.5*m1*(v_cm_R1.'*v_cm_R1) + 0.5*theta1*omega_R1^2 + ...
          0.5*m2*(v_cm_R2.'*v_cm_R2) + 0.5*theta2*omega_R2^2 + ...
          0.5*m_EE*(v_cm_EE2.'*v_cm_EE2) + 0.5*theta_EE*omega_EE2^2 + ...
          0.5*m_box*(v_cm_box.'*v_cm_box) + 0.5*theta_box*omega_box^2;

M = simplify(jacobian(gradient(T_total, dqdt), dqdt));

V_total = m1*g*pos_cm_L1(2) + m2*g*pos_cm_L2(2) + m_EE*g*pos_EE1(2) + ...
          m1*g*pos_cm_R1(2) + m2*g*pos_cm_R2(2) + m_EE*g*pos_EE2(2) + ...
          m_box*g*y_box;

% Generalized Forces (Gravity + Convective)
gen_force = -jacobian(V_total, q)'; % Gravity

% Helper for convective term: -J' * (m * J_dot * dqdt)
add_conv = @(J_pos, m_val, v_val) -J_pos' * (m_val * (jacobian(v_val, q) * dqdt));

gen_force = gen_force + add_conv(jacobian(pos_cm_L1, q), m1, v_cm_L1);
gen_force = gen_force + add_conv(jacobian(pos_cm_L2, q), m2, v_cm_L2);
gen_force = gen_force + add_conv(jacobian(pos_EE1, q), m_EE, v_cm_EE1);
gen_force = gen_force + add_conv(jacobian(pos_cm_R1, q), m1, v_cm_R1);
gen_force = gen_force + add_conv(jacobian(pos_cm_R2, q), m2, v_cm_R2);
gen_force = gen_force + add_conv(jacobian(pos_EE2, q), m_EE, v_cm_EE2);
gen_force = gen_force + add_conv(jacobian(I_r_Obox, q), m_box, v_cm_box);

gen_force = simplify(gen_force);

% ---------------------------------------------------------
% 6. Actuation Jacobian
% ---------------------------------------------------------
% B projects tau (6x1) into the generalized coordinate space q (9x1).
% q(1:6) are the relative joint angles which correspond 1-to-1 with tau.
% q(7:9) are the box coordinates (unactuated).

B = sym(zeros(9, 6));
B(1:6, 1:6) = eye(6); 

% The term B*tau can now be added to the RHS of the dynamics
gen_force_actuated = B * tau;

% Display result
disp('Derivation complete.');
disp('B matrix (Actuation Jacobian):');
disp(B);