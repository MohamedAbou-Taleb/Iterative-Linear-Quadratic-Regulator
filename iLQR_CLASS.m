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
        
        function [X, U, cost] = forward_pass(obj)
            X = zeros(obj.n_x, obj.N+1);
            X(:, 1) = obj.x_0;
            cost = 0;
            for k = 1:obj.N
                xk = X(:, k);
                xk_old = obj.X(:, k);
                delta_x = xk - xk_old;
                K = obj.K_cell_array{k};
                uk = obj.U(:, k) + obj.U_ff(:, k) + K*delta_x;
                xkPlusOne = obj.f_fcn(xk, uk);
                X(:, k+1) = xkPlusOne;
                obj.U(:, k) = uk;
                cost = cost + obj.l_fcn(xk, uk);
            end
            cost = cost + obj.l_f_fcn(X(:, end));
            obj.X = X;
            U = obj.U;
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

        function [X, U, cost] = optimize_trajectory(obj)

            % 1. Run the initial forward pass to get the nominal trajectory
            [X, U, cost] = obj.forward_pass();

            cost_prev = inf;

            % 2. Run the optimization loop
            for i = 1:obj.maxiter

                % 3. Check for convergence
                   % This compares the new cost (cost) from the *previous*
                   % iteration with the one before it (cost_prev).
                if abs(cost - cost_prev) <= obj.tol
                    fprintf('Converged at iteration %d\n', i);
                    break;
                end

                % 4. Store the current cost *before* you calculate the next one
                cost_prev = cost;

                % 5. Compute new gains based on the current (X, U)
                obj.backward_pass();

                % 6. Apply new gains (with line search) to get a new,
                   % lower-cost trajectory (X, U) and its new 'cost'.
                [X, U, cost] = obj.forward_pass();
            end

            if i == obj.maxiter
                fprintf('Warning: Reached max iterations (%d) without converging.\n', obj.maxiter);
            end

            X_star = X;
            U_star = U;
        end

    end
end