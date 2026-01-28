% Example for 1-D rigid body mass estimation using the Lyapunov rule
clear all
close all
clc

%% parameters and dynamics
k_true = 1.0;
A = [0, 1; 0, 0];
C = eye(2);
B = [0; k_true];
B_k = @(k) [0; k];
observer_poles = [-20, -10];
L = place(A', C', observer_poles)';
gamma = 10.0;
sigma = 0;
khat0 = 1.5;

input_noise_fcn = @(t) 0.2*cos(1*t);

dyn_fcn = @(t, x, u) A*x + B*u(t,x);
observer_dyn_fcn = @(t, x_hat, u, x, k_hat) A*x_hat + B_k(k_hat)*( u(t, x) + input_noise_fcn(t) ) + L*C*(x-x_hat);
k_hat_dyn_fcn = @(t, x_hat, x, u, k_hat) gamma*(x(2) - x_hat(2))*( u(t, x) + input_noise_fcn(t) ) - sigma*(k_hat - khat0);

closed_loop_dyn_fcn = @(t, z, u) [dyn_fcn( t, z(1:2), u);
                                  observer_dyn_fcn( t, z(3:4), u, z(1:2), z(5) );
                                  k_hat_dyn_fcn( t, z(3:4), z(1:2), u, z(5) )];

x0 = [1;0];
xhat0 = [2.1;0.2];


z0 = [x0; xhat0; khat0];

dt = 0.01;
Tsim = 5;
tspan = 0:dt:Tsim;

% u_fcn = @(t, x) cos(0.1*t) + exp(-1*t);
u_fcn = @(t, x) 1;
[t, z] = ode45(@(t, z) closed_loop_dyn_fcn(t, z, u_fcn), tspan, z0);
z = z';
x = z(1:2, :);
xhat = z(3:4, :);
khat = z(5, :);

%%
figure()
subplot(3, 1, 1)
hold on
plot(tspan, x(1,:), LineWidth=2)
plot(tspan, xhat(1,:), LineWidth=2, LineStyle="--")
grid on
legend({'x', 'xhat'})
subplot(3, 1, 2)
hold on
plot(tspan, x(2,:), LineWidth=2)
plot(tspan, xhat(2,:), LineWidth=2, LineStyle="--")
grid on
legend({'x', 'xhat'})
subplot(3, 1, 3)
hold on
plot([0, Tsim], [k_true, k_true], LineWidth=2)
plot(tspan, khat(1,:), LineWidth=2, LineStyle="--")
grid on
legend({'k', 'khat'})

function [f_val, grad_f] = param_constraint_f(theta, theta_max, epsilon_0)
    % PARAM_CONSTRAINT_F Implements Example 1.7.8 from the text.
    %
    % Inputs:
    %   theta     : Current parameter vector (k x 1)
    %   theta_max : Maximum allowable norm bound
    %   epsilon_0 : Tolerance parameter for the smooth boundary
    %
    % Outputs:
    %   f_val     : Scalar value of the convex function f(theta)
    %   grad_f    : Gradient vector of f at theta (k x 1)

    % Ensure theta is a column vector
    theta = theta(:);
    
    % Pre-calculate norm squared
    norm_theta_sq = theta' * theta;
    theta_max_sq = theta_max^2;
    
    % Calculate f(theta)
    % Equation: (||theta||^2 - theta_max^2) / (epsilon_0 * theta_max^2)
    f_val = (norm_theta_sq - theta_max_sq) / (epsilon_0 * theta_max_sq);
    
    % Calculate Gradient of f(theta)
    % Equation: 2*theta / (epsilon_0 * theta_max^2)
    grad_f = (2 * theta) / (epsilon_0 * theta_max_sq);
end

function proj_y = projection_op(y, f_val, grad_f)
    % PROJECTION_OP Implements Definition 1.7.7 (Projection Operator)
    %
    % Inputs:
    %   y      : The update vector or derivative (k x 1) (e.g., from adaptive law)
    %   f_val  : Scalar value of the convex function f at current theta
    %   grad_f : Gradient vector of f at current theta (k x 1)
    %
    % Output:
    %   proj_y : The projected update vector

    % Ensure vectors are columns
    y = y(:);
    grad_f = grad_f(:);

    % Check the condition from Def 1.7.7:
    % if f(theta) > 0 AND y' * grad f(theta) > 0
    inner_prod = y' * grad_f;
    
    if (f_val > 0) && (inner_prod > 0)
        % Calculate the squared norm of the gradient
        grad_norm_sq = grad_f' * grad_f;
        
        % Avoid division by zero (safety check)
        if grad_norm_sq < eps
            proj_y = y;
            return;
        end
        
        % Apply the projection formula:
        % Proj = y - ( (grad * grad') / ||grad||^2 ) * y * f(theta)
        %
        % We use (grad * (grad' * y)) for efficient matrix-vector multiplication
        subtraction_term = (grad_f * inner_prod * f_val) / grad_norm_sq;
        
        proj_y = y - subtraction_term;
    else
        % Otherwise, return y unchanged
        proj_y = y;
    end
end