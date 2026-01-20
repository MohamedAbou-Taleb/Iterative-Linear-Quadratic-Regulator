clc
clear all
close all

syms q [9, 1] real
syms dqdt [9, 1] real

syms w_box h_box real
syms w_EE h_EE real
syms m_box m_EE theta_box theta_EE real
syms g real

x_EE1 = q(1);
y_EE1 = q(2);
phi_EE1 = q(3);
x_EE2 = q(4);
y_EE2 = q(5);
phi_EE2 = q(6);
x_box = q(7);
y_box = q(8);
phi_box = q(9);

A_IB_fcn = @(phi) [cos(phi), -sin(phi);
                   sin(phi), cos(phi)];
A_IB_EE1 = A_IB_fcn(phi_EE1);
A_IB_EE2 = A_IB_fcn(phi_EE2);
A_IB_box = A_IB_fcn(phi_box);

I_r_OEE1 = [x_EE1; y_EE1];
I_r_OEE2 = [x_EE2; y_EE2];
I_r_Obox = [x_box; y_box];

I_e_x_B_EE1 = A_IB_EE1(:, 1);
I_e_y_B_EE1 = A_IB_EE1(:, 2);
I_e_x_B_EE2 = A_IB_EE2(:, 1);
I_e_y_B_EE2 = A_IB_EE2(:, 2);
I_e_x_B_box = A_IB_box(:, 1);
I_e_y_B_box = A_IB_box(:, 2);

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

I_n_box_left = -I_e_x_B_box;
I_n_box_right = I_e_x_B_box;

I_n_ground_left = [0; 1];
I_n_ground_right = [0; 1];

g_N_upper1 = simplify(I_n_box_left' * (-I_r_P_upper1_box_left));
g_N_lower1 = simplify(I_n_box_left' * (-I_r_P_lower1_box_left));
g_N_upper2 = simplify(I_n_box_right' * (-I_r_P_upper2_box_right));
g_N_lower2 = simplify(I_n_box_right' * (-I_r_P_lower2_box_right));

g_N_ground_left = I_n_ground_left' * I_r_Obox_bottom_left;
g_N_ground_right = I_n_ground_right' * I_r_Obox_bottom_right;

w_N_upper1 = jacobian(g_N_upper1, q)';
w_N_lower1 = jacobian(g_N_lower1, q)';
w_N_upper2 = jacobian(g_N_upper2, q)';
w_N_lower2 = jacobian(g_N_lower2, q)';

w_N_ground_left = jacobian(g_N_ground_left, q)';
w_N_ground_right = jacobian(g_N_ground_right, q)';

I_t_left = I_e_y_B_box;
I_t_right = I_e_y_B_box;
I_t_ground = [1; 0];

I_v_EE1 = [dqdt1; dqdt2];
I_v_EE2 = [dqdt4; dqdt5];
I_v_box = [dqdt7; dqdt8];
Omega_EE1 = dqdt3;
Omega_EE2 = dqdt6;
Omega_box = dqdt9;

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

w_T_ground_left = simplify( jacobian(gamma_T_ground_left, q)' );
w_T_ground_right = simplify( jacobian(gamma_T_ground_right, q)' );


W = [w_T_upper1, w_N_upper1, w_T_lower1, w_N_lower1, ...
    w_T_upper2, w_N_upper2, w_T_lower2, w_N_lower2, ...
    w_T_ground_left, w_N_ground_left, w_T_ground_right, w_N_ground_right];
W_dot_transpose_dqdt = jacobian(W'*dqdt, q);
g_N = [g_N_upper1; g_N_lower1; g_N_upper2; g_N_lower2; g_N_ground_left; g_N_ground_right];
M = blkdiag(m_EE*eye(2), theta_EE, m_EE*eye(2), theta_EE, m_box*eye(2), theta_box);
gen_force = sym(zeros(9, 1)); gen_force(8) = -m_box*g;