clear all
close all
clc

syms q [7, 1] real
syms dqdt [7, 1] real
syms w h real
syms m_box m_ball ball_radius theta_box real
syms g real

phi = q(7);
A_IB = [cos(phi), -sin(phi);
        sin(phi), cos(phi)];

I_r_Oball1 = [q(1); q(2)];
I_r_Oball2 = [q(3); q(4)];
I_r_Obox = [q(5); q(6)];

I_e_x_B = A_IB(:, 1);
I_e_y_B = A_IB(:, 2);

I_r_P1ball1 = I_r_Oball1 - (I_r_Obox - w/2*I_e_x_B);
I_r_P2ball2 = I_r_Oball2 - (I_r_Obox + w/2*I_e_x_B);
B_r_P1ball1 = A_IB'*I_r_P1ball1;
B_r_P2ball2 = A_IB'*I_r_P2ball2;
I_r_OP3 = I_r_Obox - w/2*I_e_x_B - h/2*I_e_y_B;
I_r_OP4 = I_r_Obox + w/2*I_e_x_B - h/2*I_e_y_B;

B_n_1 = [-1; 0];
B_n_2 = [1; 0];

I_n_1 = A_IB*B_n_1;
I_n_2 = A_IB*B_n_2;
I_n_3 = [0; 1];
I_n_4 = [0; 1];

g_N1 = simplify(I_r_P1ball1'*I_n_1) - ball_radius;
g_N2 = simplify(I_r_P2ball2'*I_n_2) - ball_radius;
g_N3 = simplify(I_r_OP3'*I_n_3);
g_N4 = simplify(I_r_OP4'*I_n_4);

w_N1 = jacobian(g_N1, q)';
w_N2 = jacobian(g_N2, q)';
w_N3 = jacobian(g_N3, q)';
w_N4 = jacobian(g_N4, q)';


I_t_1 = I_e_y_B;
I_t_2 = I_e_y_B;
I_t_3 = [1; 0];
I_t_4 = [1; 0];

I_v_ball1 = [dqdt1; dqdt2];
I_v_ball2 = [dqdt3; dqdt4];
I_v_box = [dqdt5; dqdt6];
Omega_box = dqdt7;

I_r_boxP1 = -w/2*I_e_x_B;
I_r_boxP2 = +w/2*I_e_x_B;
I_r_OP1 = I_r_Obox + I_r_boxP1;
I_r_OP2 = I_r_Obox + I_r_boxP2;

I_v_P1 = jacobian(I_r_OP1, q)*dqdt;
I_v_P2 = jacobian(I_r_OP2, q)*dqdt;
I_v_P3 = jacobian(I_r_OP3, q)*dqdt;
I_v_P4 = jacobian(I_r_OP4, q)*dqdt;
I_v_ball1 = jacobian(I_r_Oball1, q)*dqdt;
I_v_ball2 = jacobian(I_r_Oball2, q)*dqdt;

gamma_T1 = simplify(I_t_1'*(I_v_ball1 - I_v_P1));
gamma_T2 = simplify(I_t_2'*(I_v_ball2 - I_v_P2));
gamma_T3 = simplify(I_t_3'*I_v_P3);
gamma_T4 = simplify(I_t_4'*I_v_P4);

w_T1 = jacobian(gamma_T1, dqdt)';
w_T2 = jacobian(gamma_T2, dqdt)';
w_T3 = jacobian(gamma_T3, dqdt)';
w_T4 = jacobian(gamma_T4, dqdt)';


%% Save the following variables as functions
W = [w_T1, w_N1, w_T2, w_N2, w_T3, w_N3, w_T4, w_N4];
g_N = [g_N1; g_N2; g_N3; g_N4];
gamma_T = [gamma_T1; gamma_T2; gamma_T3; gamma_T4];
M = blkdiag(m_ball*eye(4), m_box*eye(2), theta_box);
gen_force = [0; 0; 0; 0; 0; -m_box*g; 0];