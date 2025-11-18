clear all
close all
clc

%%
m = 1;
dt = 0.01;
g = 9.81;
tol_newton = 1e-5;
max_iter = 10000;
tol_fixed_point = 1e-5;
r = 1;
T = 3;
tspan = 0:dt:T;
N = length(tspan);

q0 = 1;
u0 = 0;
PN0 = 0;
% Initialize arrays to store results
q = zeros(1, N);
u = zeros(1, N);
PN = zeros(1, N);
q(:, 1) = q0;
u(:, 1) = u0;
PN(:, 1) = PN0;
R_x = @(x, y, qk, uk) [x(1) - dt*(x(2)+uk);
               m*x(2) + dt*m*g - y(1)];
J_x = @(x, y) [1, -dt;
               0, m];
xk_0 = [0;0];
yk_0 = 0;
for k = 1:(N-1)
    qk = q(:, k);
    uk = u(:, k);
    [xkPlusOne_0, ~,~, conv_newton] = newton(@(x) R_x(x, yk_0, qk, uk), @(x) J_x(x, yk_0), xk_0, tol_newton, max_iter);
    conv = 0;
    i = 1;
    ykPlusOne_i = yk_0;
    xkPlusOne_i = xkPlusOne_0;

    while ~conv && i <= max_iter
        ykPlusOne_iPlusOne = -prox_R0minus(-ykPlusOne_i + r*(qk + xkPlusOne_i(1)));
        xkPlusOne_iPlusOne = newton(@(x) R_x(x, ykPlusOne_iPlusOne, qk, uk), @(x) J_x(x, ykPlusOne_iPlusOne),...
            xkPlusOne_i, tol_newton, max_iter);
        conv = norm(xkPlusOne_iPlusOne - xkPlusOne_i) < tol_fixed_point;
        xkPlusOne_i = xkPlusOne_iPlusOne;
        ykPlusOne_i = ykPlusOne_iPlusOne;
        i=i+1;
    end
    disp(conv)
    dq = xkPlusOne_iPlusOne(1);
    du = xkPlusOne_iPlusOne(2);
    dP = ykPlusOne_iPlusOne(1);
    q(:, k+1) = qk + dq;
    u(:, k+1) = uk + du;
    PN(:, k+1) = dP; 
    yk_0 = dP;
    xk_0 = xkPlusOne_iPlusOne;
end

%%
figure;
subplot(3, 1, 1);
plot(tspan, q, 'b-', 'LineWidth', 1.5);
title('Position');
xlabel('Time (s)');
ylabel('q');
grid on;

subplot(3, 1, 2);
plot(tspan, u, 'r-', 'LineWidth', 1.5);
title('Velocity');
xlabel('Time (s)');
ylabel('u');
grid on;

subplot(3, 1, 3);
plot(tspan, PN, 'g-', 'LineWidth', 1.5);
title('Percussion');
xlabel('Time (s)');
ylabel('P_N');
grid on;
