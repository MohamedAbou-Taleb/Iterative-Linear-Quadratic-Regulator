clear all
close all
clc

%%
syms q [6,1] real
syms dqdt [6,1] real
syms tau [4,1] real
syms g real

syms m_B m_upper m_lower theta_upper theta_lower real
syms l_upper l_lower real
syms d_upper d_lower real

x_MB = q1;
y_MB = q2;
I_r_Omb = [x_MB; y_MB];


I_r_Ojoint1 = I_r_Omb + [l_upper*sin(q3); -l_upper*cos(q3)];
I_r_Ojoint2 = I_r_Omb + [l_upper*sin(q5); -l_upper*cos(q5)];

I_r_Oupper1 = I_r_Omb + [l_upper/2*sin(q3); -l_upper/2*cos(q3)];
I_r_Oupper2 = I_r_Omb + [l_upper/2*sin(q5); -l_upper/2*cos(q5)];

I_r_Olower1 = I_r_Ojoint1 + [l_lower/2*sin(q3 + q4); -l_lower/2*cos(q3+q4)];
I_r_Olower2 = I_r_Ojoint2 + [l_lower/2*sin(q5 + q6); -l_lower/2*cos(q5+q6)];

I_r_Ofoot1 = I_r_Ojoint1 + [l_lower*sin(q3 + q4); -l_lower*cos(q3+q4)];
I_r_Ofoot2 = I_r_Ojoint2 + [l_lower*sin(q5 + q6); -l_lower*cos(q5+q6)];

I_g_vec = [0; -g];

Theta_upper = diag([0,0,theta_upper]);
Theta_lower = diag([0,0,theta_lower]);
Omega_upper1 = [0;0;dqdt3];
Omega_upper2 = [0;0; dqdt5];
Omega_lower1 = [0; 0; dqdt3 + dqdt4];
Omega_lower2 = [0; 0; dqdt5 + dqdt6];

J_mb = jacobian(I_r_Omb, q);
J_s_upper1 = jacobian(I_r_Ojoint1, q);
J_s_upper2 = jacobian(I_r_Ojoint2, q);
J_s_lower1 = jacobian(I_r_Olower1, q);
J_s_lower2 = jacobian(I_r_Olower2, q);

J_R_upper1 = jacobian(Omega_upper1, dqdt);
J_R_upper2 = jacobian(Omega_upper2, dqdt);
J_R_lower1 = jacobian(Omega_lower1, dqdt);
J_R_lower2 = jacobian(Omega_lower2, dqdt);

M = J_mb'*m_B*J_mb + ...
    J_s_upper1' * m_upper * J_s_upper1 + J_R_upper1' * Theta_upper * J_R_upper1 + ...
    J_s_upper2' * m_upper * J_s_upper2 + J_R_upper2' * Theta_upper * J_R_upper2 + ...
    J_s_lower1' * m_lower * J_s_lower1 + J_R_lower1' * Theta_lower * J_R_lower1 + ...
    J_s_lower2' * m_lower * J_s_lower2 + J_R_lower2' * Theta_lower * J_R_lower2;
M = simplify(M);
f_c = -J_s_upper1' * jacobian(J_s_upper1*dqdt, q)*dqdt + ...
       -J_s_upper2' * jacobian(J_s_upper2*dqdt, q)*dqdt + ...
       -J_s_lower1' * jacobian(J_s_lower1*dqdt, q)*dqdt + ...
       -J_s_lower2' * jacobian(J_s_lower2*dqdt, q)*dqdt;

f_g = J_mb'*m_B*I_g_vec + ...
    J_s_upper1'*m_upper*I_g_vec + ...
    J_s_upper2'*m_upper*I_g_vec + ...
    J_s_lower1'*m_lower*I_g_vec + ...
    J_s_lower2'*m_lower*I_g_vec;

f_cg = simplify(f_c + f_g);

W_tau = [zeros(2, 4);
    eye(4)];
f_tau = W_tau*tau;

gen_force = f_cg + f_tau;

%% Contact dynamics
I_n_1 = [0; 1];
I_n_2 = [0; 1];

g_N1 = I_n_1'*I_r_Ofoot1;
g_N2 = I_n_2'*I_r_Ofoot2;

w_N1 = jacobian(g_N1, q)';
w_N2 = jacobian(g_N2, q)';

I_t_1 = [1; 0];
I_t_2 = [1; 0];

I_v_foot1 = jacobian(I_r_Ofoot1, q)*dqdt;
I_v_foot2 = jacobian(I_r_Ofoot2, q)*dqdt;

gamma_T1 = simplify(I_t_1'*I_v_foot1);
gamma_T2 = simplify(I_t_2'*I_v_foot2);

w_T1 = jacobian(gamma_T1, dqdt)';
w_T2 = jacobian(gamma_T2, dqdt)';

W = [w_T1, w_N1, w_T2, w_N2];

g_N = [g_N1; g_N2];
gamma_T = [gamma_T1; gamma_T2];