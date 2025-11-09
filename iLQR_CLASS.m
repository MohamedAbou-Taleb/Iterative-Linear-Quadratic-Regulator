classdef iLQR_CLASS < handle
    %LINEAR_ILQR_CLASS Summary of this class goes here

    properties
        f_fcn
        f_x_fcn
        f_u_fcn
        l_fcn
        l_x_fcn
        l_u_fcn
        l_xx_fcn
        l_ux_fcn
        l_uu_fcn
        l_f_fcn
        l_f_x_fcn
        l_f_xx_fcn
        dt
        T
        tspan
        N
        n_x
        n_u
        X
        U
        K_cell_array
        U_ff
        x_0
        tol = 1e-6; % Define tolerance for cost change
        maxiter = 100;
        alpha_factor = 0.5; % Factor to shrink alpha (e.g., 0.5)
        min_alpha = 1e-8;   % Minimum alpha to try
    end

    methods
        function obj = iLQR_CLASS(args)
            arguments
                args.f_fcn;
                args.f_x_fcn;
                args.f_u_fcn;
                args.l_fcn;
                args.l_x_fcn;
                args.l_u_fcn;
                args.l_xx_fcn;
                args.l_ux_fcn;
                args.l_uu_fcn;
                args.l_f_fcn;
                args.l_f_x_fcn;
                args.l_f_xx_fcn;
                args.x_0
                args.U_ff
                args.dt
                args.T
                args.tol
                args.maxiter
            end
            % Assign validated values to object properties
            props = fieldnames(args);
            for k = 1:numel(props)
                    obj.(props{k}) = args.(props{k});
            end
            obj.n_x = length(obj.x_0);
            [obj.n_u, ~] = size(obj.U_ff);
            obj.tspan = 0:obj.dt:obj.T;
            obj.N = length(obj.tspan) - 1;
            obj.X = zeros(obj.n_x, obj.N+1);
            obj.U = obj.U_ff;
            for i = 1:obj.N
                obj.K_cell_array{i} = zeros(obj.n_u, obj.n_x);
            end
        end

        function [xkPlusOne, f_x, f_u] = dynamics_fcn(obj, x, u)
            xkPlusOne = obj.f_fcn(x, u);
            f_x = obj.f_x_fcn(x, u);
            f_u = obj.f_u_fcn(x, u);

        end

        function [l, l_x, l_u, l_xx, l_ux, l_uu] = stage_cost_fcn(obj, x, u)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            l = obj.l_fcn(x, u);
            l_x = obj.l_x_fcn(x, u);
            l_u = obj.l_u_fcn(x, u);
            l_xx = obj.l_xx_fcn(x, u);
            l_ux = obj.l_ux_fcn(x, u);
            l_uu = obj.l_uu_fcn(x, u);
        end

        function [l_f, l_f_x, l_f_xx] = terminal_cost_fcn(obj, x)
            l_f = obj.l_f_fcn(x);
            l_f_x = obj.l_f_x_fcn(x);
            l_f_xx = obj.l_f_xx_fcn(x);
        end

        function [Q_x, Q_u, Q_xx, Q_ux, Q_uu] = Q_fcn(obj, x, u, V_x, V_xx)
            [~, l_x, l_u, l_xx, l_ux, l_uu] = obj.stage_cost_fcn(x, u);
            [~, f_x, f_u] = obj.dynamics_fcn(x, u);

            Q_x = l_x + V_x*f_x;
            Q_u = l_u + V_x*f_u;
            Q_xx = l_xx + f_x'*V_xx*f_x;
            Q_ux = l_ux + f_u'*V_xx*f_x;
            Q_uu = l_uu + f_u'*V_xx*f_u;
        end

        function [u_ff, K, V_x_prev, V_xx_prev] = u_opt_fcn(obj, x, u, V_x, V_xx)
            [Q_x, Q_u, Q_xx, Q_ux, Q_uu] = obj.Q_fcn(x, u, V_x, V_xx);
            K = -Q_uu\Q_ux;
            u_ff = -Q_uu\Q_u';
            V_x_prev = Q_x + Q_u*K;
            V_xx_prev = Q_xx + Q_ux'*K;
        end

        % --- MODIFIED FUNCTION ---
        function [X_new, U_new, cost_new] = forward_pass(obj, alpha)
            % This function now takes an 'alpha' and returns a
            % CANDIDATE trajectory, without modifying obj.X or obj.U

            X_new = zeros(obj.n_x, obj.N+1);
            X_new(:, 1) = obj.x_0;
            U_new = zeros(obj.n_u, obj.N);
            cost_new = 0;

            for k = 1:obj.N
                xk_new = X_new(:, k);       % State from this candidate rollout
                xk_old = obj.X(:, k);       % State from the *previous* nominal trajectory

                delta_x = xk_new - xk_old;

                K = obj.K_cell_array{k};
                uk_old = obj.U(:, k);       % Control from *previous* nominal trajectory
                uk_ff = obj.U_ff(:, k);   % Feedforward *correction*

                % --- THIS IS THE KEY LINE SEARCH EQUATION ---
                % u_new = u_nom + alpha * k + K * (x_new - x_nom)
                uk_new = uk_old + alpha * uk_ff + K * delta_x;

                % Store the new control
                U_new(:, k) = uk_new;

                % Simulate true nonlinear dynamics
                xkPlusOne = obj.f_fcn(xk_new, uk_new);
                X_new(:, k+1) = xkPlusOne;

                % Calculate stage cost
                cost_new = cost_new + obj.l_fcn(xk_new, uk_new);
            end

            % Add terminal cost
            cost_new = cost_new + obj.l_f_fcn(X_new(:, end));

            % --- NOTE: We DO NOT do obj.X = X_new here. ---
            % The optimize_trajectory function will do that if cost_new is good.
        end
        function backward_pass(obj)
            x = obj.X(:, end);
            [l_f, l_f_x, l_f_xx] = obj.terminal_cost_fcn(x);
            V_x = l_f_x;
            V_xx = l_f_xx;
            for k = obj.N:-1:1
                x = obj.X(:, k);
                u = obj.U(:, k);
                [u_ff, K, V_x_prev, V_xx_prev] = u_opt_fcn(obj, x, u, V_x, V_xx);
                obj.U_ff(:, k) = u_ff;
                obj.K_cell_array{k} = K;
                V_x = V_x_prev;
                V_xx = V_xx_prev;
            end

        end

        % --- MODIFIED FUNCTION ---
        function [X_star, U_star, cost] = optimize_trajectory(obj)

            % 1. Run the initial "rollout" to get the cost of the initial guess.
            % We call forward_pass with alpha=0. This simulates the initial
            % obj.U with K=0 and U_ff=0, giving the initial cost.
            [obj.X, obj.U, cost] = obj.forward_pass(0);

            fprintf('Initial cost: %.4f\n', cost);

            cost_prev = cost;

            % 2. Run the optimization loop
            for i = 1:obj.maxiter

                % 3. Check for convergence
                if abs(cost - cost_prev) <= obj.tol && i > 1
                    fprintf('Converged at iteration %d\n', i);
                    break;
                end

                % 4. Store the current cost
                cost_prev = cost;

                % 5. Compute new gains based on the current (obj.X, obj.U)
                obj.backward_pass(); % This populates obj.U_ff and obj.K_cell_array

                % 6. === LINE SEARCH ===
                % Try to find a step size alpha that reduces the cost
                alpha = 1.0; % Start with a full step
                is_step_accepted = false;

                for j = 1:10 % Max line search attempts
                    [X_new, U_new, cost_new] = obj.forward_pass(alpha);

                    % Check for cost reduction
                    if cost_new < cost_prev
                        % SUCCESS: Accept the new trajectory
                        obj.X = X_new;
                        obj.U = U_new;
                        cost = cost_new;
                        is_step_accepted = true;
                        fprintf('  Iter %d (alpha=%.2e): Cost improved to %.4f\n', i, alpha, cost);
                        break; % Exit line search loop
                    else
                        % FAILURE: Reduce step size and try again
                        alpha = alpha * obj.alpha_factor;
                        if alpha < obj.min_alpha
                            break; % Alpha is too small
                        end
                    end
                end

                if ~is_step_accepted
                    fprintf('Warning: Line search failed at iteration %d. Cost did not improve.\n', i);
                    break; % Exit main optimization loop
                end
                % === END LINE SEARCH ===
            end

            if i == obj.maxiter
                fprintf('Warning: Reached max iterations (%d) without converging.\n', obj.maxiter);
            end

            X_star = obj.X;
            U_star = obj.U;
        end

    end
end