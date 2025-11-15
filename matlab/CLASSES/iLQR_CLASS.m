classdef iLQR_CLASS < handle
    properties
        % --- System and Trajectory ---
        system     % Handle to the System_CLASS object
        T          % Total time (from constructor)
        tspan
        N
        X          % Nominal state trajectory
        U          % Nominal control trajectory
        x_0        % Initial state
        U_init
        
        % ... (Other properties: K_cell_array, U_ff, tol, etc.) ...
        K_cell_array 
        U_ff         
        tol = 1e-6;
        maxiter = 100;
        alpha_factor = 0.5;
        min_alpha = 1e-8;
    end
    
    methods
        function obj = iLQR_CLASS(args)
            arguments
                args.system System_CLASS % Pass the system object
                args.T double             % Required: Total time
                
                % Optional: Override system defaults
                args.x_0 double = []
                
                % Optional: Solver settings
                args.tol
                args.maxiter
                args.U_init
            end
            % Assign validated values to object properties

            props = fieldnames(args);

            for k = 1:numel(props)

                obj.(props{k}) = args.(props{k});

            end
            
            obj.tspan = 0:obj.system.dt:obj.T;
            obj.N = length(obj.tspan) - 1;

            % 4. Set initial state (use system's as default)
            obj.x_0 = args.x_0;

            % 5. Initialize trajectories
            obj.X = zeros(obj.system.n_x, obj.N+1);
            obj.K_cell_array = cell(1, obj.N);
            obj.U_ff = zeros(obj.system.n_u, obj.N);
            
            
            obj.U = obj.U_init;
            
            % 7. Initialize K (as zeros)
            for i = 1:obj.N
                obj.K_cell_array{i} = zeros(obj.system.n_u, obj.system.n_x);
            end

        end
        
        function [xkPlusOne, f_x, f_u] = dynamics_fcn(obj, x, u)
            xkPlusOne = obj.system.f_fcn(x, u);
            f_x = obj.system.f_x_fcn(x, u);
            f_u = obj.system.f_u_fcn(x, u);
        end
        
        function [l, l_x, l_u, l_xx, l_ux, l_uu] = stage_cost_fcn(obj, x, u)
            l    = obj.system.l_fcn(x, u);
            l_x  = obj.system.l_x_fcn(x, u);
            l_u  = obj.system.l_u_fcn(x, u);
            l_xx = obj.system.l_xx_fcn(x, u);
            l_ux = obj.system.l_ux_fcn(x, u);
            l_uu = obj.system.l_uu_fcn(x, u);
        end
        
        function [l_f, l_f_x, l_f_xx] = terminal_cost_fcn(obj, x)
            l_f    = obj.system.l_f_fcn(x);
            l_f_x  = obj.system.l_f_x_fcn(x);
            l_f_xx = obj.system.l_f_xx_fcn(x);
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
        
        function [X_new, U_new, cost_new] = forward_pass(obj, alpha)
            X_new = zeros(obj.system.n_x, obj.N+1);
            X_new(:, 1) = obj.x_0;
            U_new = zeros(obj.system.n_u, obj.N);
            cost_new = 0;
            for k = 1:obj.N
                xk_new = X_new(:, k);
                xk_old = obj.X(:, k);
                delta_x = xk_new - xk_old;
                K = obj.K_cell_array{k};
                uk_old = obj.U(:, k);
                uk_ff = obj.U_ff(:, k);
                
                uk_new = uk_old + alpha * uk_ff + K * delta_x;
                
                U_new(:, k) = uk_new;
                
                xkPlusOne = obj.system.f_fcn(xk_new, uk_new);
                X_new(:, k+1) = xkPlusOne;
                cost_new = cost_new + obj.system.l_fcn(xk_new, uk_new);
            end
            cost_new = cost_new + obj.system.l_f_fcn(X_new(:, end));
        end
        
        function backward_pass(obj)
            x = obj.X(:, end);
            [~, V_x, V_xx] = obj.terminal_cost_fcn(x);
            
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
        
        function [X_star, U_star, cost] = optimize_trajectory(obj)
            [obj.X, obj.U, cost] = obj.forward_pass(0);
            fprintf('Initial cost: %.4f\n', cost);
            cost_prev = cost;
            for i = 1:obj.maxiter
                if abs(cost - cost_prev) <= obj.tol && i > 1
                    fprintf('Converged at iteration %d\n', i);
                    break;
                end
                cost_prev = cost;
                
                obj.backward_pass();
                
                alpha = 1.0;
                is_step_accepted = false;
                for j = 1:10
                    [X_new, U_new, cost_new] = obj.forward_pass(alpha);
                    if cost_new < cost_prev
                        obj.X = X_new;
                        obj.U = U_new;
                        cost = cost_new;
                        is_step_accepted = true;
                        fprintf('  Iter %d (alpha=%.2e): Cost improved to %.4f\n', i, alpha, cost);
                        break;
                    else
                        alpha = alpha * obj.alpha_factor;
                        if alpha < obj.min_alpha
                            break;
                        end
                    end
                end
                if ~is_step_accepted
                    fprintf('Warning: Line search failed at iteration %d. Cost did not improve.\n', i);
                    break;
                end
            end
            if i == obj.maxiter
                fprintf('Warning: Reached max iterations (%d) without converging.\n', obj.maxiter);
            end
            X_star = obj.X;
            U_star = obj.U;
        end
    end
end