clear all
close all
clc

%%
syms q [2, 1] real
syms dqdt [2, 1] real
syms tau [2, 1] real
syms m1 m2 l1 l2 d1 d2 g theta1 theta2 real
I_Theta_s1 = diag([0,0,theta1]);
I_Theta_s2 = diag([0,0,theta2]);
%% kinematics

I_r_os1 = [0.5*l1*sin(q1); 
           -0.5*l1*cos(q1);
           0];
I_r_os2 = [l1*sin(q1) + 0.5*l2*sin(q1+q2);
            -l1*cos(q1) - 0.5*l2*cos(q1+q2);
            0];

I_J_s1 = jacobian(I_r_os1, q);
I_J_s2 = jacobian(I_r_os2, q);

I_v_s1 = I_J_s1*dqdt;
I_v_s2 = I_J_s2*dqdt;

I_Omega_s1 = [0; 0; dqdt1];
I_Omega_s2 = [0; 0; dqdt1 + dqdt2];

I_J_R1 = jacobian(I_Omega_s1, dqdt);
I_J_R2 = jacobian(I_Omega_s2, dqdt);

%%
M = I_J_s1'*m1*I_J_s1 ...
    + I_J_s2'*m2*I_J_s2 ...
    + I_J_R1'*I_Theta_s1*I_J_R1 ...
    + I_J_R2'*I_Theta_s2*I_J_R2;

f_c = -I_J_s1'*m1*jacobian(I_J_s1*dqdt, q)*dqdt - I_J_s2'*m2*jacobian(I_J_s2*dqdt, q)*dqdt ...
      -I_J_R1'*(cross(I_Omega_s1, I_Theta_s1*I_Omega_s1)) -I_J_R2'*(cross(I_Omega_s2, I_Theta_s2*I_Omega_s2));
I_g = [0; -g; 0];
F_g1 = m1*I_g;
F_g2 = m2*I_g;
f_g = I_J_s1'*F_g1 + I_J_s2'*F_g2;
f_cg = simplify(f_c + f_g);

W1 = I_J_R1';
W2 = (I_J_R2-I_J_R1)';
M_act1 = [0; 0; tau1];
M_act2 = [0; 0; tau2];
M_d1 = [0; 0; -d1*dqdt1];
M_d2 = [0; 0; -d2*dqdt2];
f_NP = W1*M_act1 + W2*M_act2 + W1*M_d1 + W2*M_d2;
h = simplify(f_cg + f_NP);

ddqdt = simplify(M\h);