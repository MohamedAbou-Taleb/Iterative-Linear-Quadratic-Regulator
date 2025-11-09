classdef optimized_iLQR_CLASS < handle
    properties
        % User-provided dynamics and cost functions
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
        
        % Problem dimensions and timing
        dt
        T
        tspan
        N
        n_x
        n_u
        
        % State and Control Trajectories
        X       % Nominal state trajectory [n_x, N+1]
        U       % Nominal control trajectory [n_u, N]
        x_0     % Initial state
        
        % iLQR Gains
        K_cell_array  % Feedback gains K [n_u, n_x] (cell array of N)
        U_ff          % Feedforward gains k [n_u, N]
        
        % Optimization Parameters
        tol = 1e-6;     % Convergence tolerance
        maxiter = 100;  % Max iterations
        
        % Regularization (Levenberg-Marquardt)
        lambda = 1.0;
        lambda_factor = 10.0;
        min_lambda = 1e-6;
        max_lambda = 1e10;
        
        % Line Search
        alpha_factor = 0.5; % Step size reduction factor (e.g., 0.5)
        max_linesearch_iter = 10;
    end
    
    methods
        function obj = optimized_iLQR_CLASS(args)
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
                args.U_ff % *** This is now treated as the INITIAL GUESS for U ***
                args.dt
                args.T
                args.tol = 1e-6;
                args.maxiter = 100;
            end
            
            % Assign validated values to object properties
            props = fieldnames(args);
            for k = 1:numel(props)
                % Special handling for U_ff, which we use as the initial U
                if strcmp(props{k}, 'U_ff')
                    obj.U = args.U_ff;
                else
                    obj.(props{k}) = args.(props{k});
                end
            end
            
            % Set dimensions
            obj.n_x = length(obj.x_0);
            [obj.n_u, ~] = size(obj.U);
            
            % Set timing
            obj.tspan = 0:obj.dt:obj.T;
            obj.N = length(obj.tspan) - 1;
            
            % Check dimensions
            if size(obj.U, 2) ~= obj.N
                error('Initial control guess U_ff must have N columns.');
            end
            
            % Preallocate memory
            obj.X = zeros(obj.n_x, obj.N+1);
            obj.U_ff = zeros(obj.n_u, obj.N); % Feedforward correction k
            
            % Use repmat for cleaner (though not significantly faster) preallocation
            obj.K_cell_array = repmat({zeros(obj.n_u, obj.n_x)}, 1, obj.N);
        end
        
        % --- Core iLQR Functions ---
        
        function [X_new, cost] = initial_rollout(obj, U_init)
            % Simulates the system forward with an initial control sequence
            % to get the initial state trajectory and cost.
            X_new = zeros(obj.n_x, obj.N+1);
            X_new(:, 1) = obj.x_0;
            cost = 0;
            
            for k = 1:obj.N
                xk = X_new(:, k);
                uk = U_init(:, k);
                
                xkPlusOne = obj.f_fcn(xk, uk);
                X_new(:, k+1) = xkPlusOne;
                cost = cost + obj.l_fcn(xk, uk);
            end
            cost = cost + obj.l_f_fcn(X_new(:, end));
        end
        
        function [X_new, U_new, cost] = forward_pass(obj, alpha)
            % Performs a forward simulation (line search) using the *current*
            % nominal trajectory (obj.X, obj.U) and the *new* gains
            % (obj.K_cell_array, obj.U_ff) with a step size 'alpha'.
            
            X_new = zeros(obj.n_x, obj.N+1);
            U_new = zeros(obj.n_u, obj.N);
            X_new(:, 1) = obj.x_0;
            cost = 0;
            
            for k = 1:obj.N
                xk_new = X_new(:, k);
                xk_nom = obj.X(:, k); % Nominal state from previous iter
                uk_nom = obj.U(:, k); % Nominal control from previous iter
                
                delta_x = xk_new - xk_nom;
                K = obj.K_cell_array{k};
                k_ff = obj.U_ff(:, k);
                
                % New control law: u = u_nom + alpha*k + K*(x - x_nom)
                uk_new = uk_nom + alpha * k_ff + K * delta_x;
                
                % Simulate one step
                xkPlusOne = obj.f_fcn(xk_new, uk_new);
                
                % Store and accumulate cost
                X_new(:, k+1) = xkPlusOne;
                U_new(:, k) = uk_new;
                cost = cost + obj.l_fcn(xk_new, uk_new);
            end
            
            % Add terminal cost
            cost = cost + obj.l_f_fcn(X_new(:, end));
        end
        
        function backward_pass(obj)
            % Computes the optimal control policy (K, k_ff) by propagating
            % the Value function (V_x, V_xx) backwards.
            
            % Get terminal cost derivatives
            [~, V_x, V_xx] = obj.terminal_cost_fcn(obj.X(:, end));
            
            for k = obj.N:-1:1
                x = obj.X(:, k);
                u = obj.U(:, k);
                
                % Get Q-function derivatives and compute optimal gains
                [u_ff, K, V_x_prev, V_xx_prev] = obj.u_opt_fcn(x, u, V_x, V_xx, obj.lambda);
                
                % Store gains
                obj.U_ff(:, k) = u_ff;
                obj.K_cell_array{k} = K;
                
                % Update value function for next (previous) timestep
                V_x = V_x_prev;
                V_xx = V_xx_prev;
            end
        end
        
        function [X_star, U_star, cost] = optimize_trajectory(obj)
            % 1. Run the initial forward pass to get the nominal trajectory
            fprintf('Starting iLQR... Initializing with rollout.\n');
            [obj.X, cost] = obj.initial_rollout(obj.U);
            cost_prev = cost;
            fprintf('Initial Cost: %.4f\n', cost);
            
            cost_history = [cost];
            
            % 2. Run the optimization loop
            for i = 1:obj.maxiter
                % 3. Compute new gains based on the current (X, U)
                obj.backward_pass();
                
                % 4. Apply new gains (with line search)
                is_linesearch_success = false;
                alpha = 1.0;
                
                for j = 1:obj.max_linesearch_iter
                    [X_new, U_new, cost_new] = obj.forward_pass(alpha);
                    
                    % Check if new cost is better
                    if cost_new < cost_prev
                        % 5. Accept new trajectory
                        obj.X = X_new;
                        obj.U = U_new;
                        cost = cost_new;
                        
                        cost_history = [cost_history, cost]; %#ok<AGROW>
                        
                        % Decrease regularization (trust this step more)
                        obj.decrease_lambda();
                        is_linesearch_success = true;
                        
                        fprintf('Iter: %3d | Cost: %9.4f | Alpha: %.3f | Lambda: %.2e\n', i, cost, alpha, obj.lambda);
                        break; % Exit line search
                    else
                        % 6. Reject step and try smaller alpha
                        alpha = alpha * obj.alpha_factor;
                    end
                end
                
                % 7. If line search failed, increase regularization
                if ~is_linesearch_success
                    obj.increase_lambda();
                    fprintf('Iter: %3d | Line search failed. Cost: %9.4f | Increasing Lambda to %.2e\n', i, cost_prev, obj.lambda);
                    
                    % Check for failure
                    if obj.lambda > obj.max_lambda
                        fprintf('Error: Lambda has exceeded max_lambda. Stopping.\n');
                        break;
                    end
                end
                
                % 8. Check for convergence
                cost_delta = abs(cost - cost_prev);
                if cost_delta <= obj.tol && is_linesearch_success
                    fprintf('Converged at iteration %d with cost %.4f\n', i, cost);
                    break;
                end
                
                % Update previous cost for next iteration's check
                cost_prev = cost;
            end
            
            if i == obj.maxiter
                fprintf('Warning: Reached max iterations (%d) without converging.\n', obj.maxiter);
            end
            
            X_star = obj.X;
            U_star = obj.U;
        end
        
        % --- Helper Methods ---
        
        function [l, l_x, l_u, l_xx, l_ux, l_uu] = stage_cost_fcn(obj, x, u)
            l    = obj.l_fcn(x, u);
            l_x  = obj.l_x_fcn(x, u);
            l_u  = obj.l_u_fcn(x, u);
            l_xx = obj.l_xx_fcn(x, u);
            l_ux = obj.l_ux_fcn(x, u);
            l_uu = obj.l_uu_fcn(x, u);
        end
        
        function [l_f, l_f_x, l_f_xx] = terminal_cost_fcn(obj, x)
            l_f    = obj.l_f_fcn(x);
            l_f_x  = obj.l_f_x_fcn(x);
            l_f_xx = obj.l_f_xx_fcn(x);
        end
        
        function [xkPlusOne, f_x, f_u] = dynamics_fcn(obj, x, u)
            xkPlusOne = obj.f_fcn(x, u);
            f_x = obj.f_x_fcn(x, u);
            f_u = obj.f_u_fcn(x, u);
        end
        
        function [Q_x, Q_u, Q_xx, Q_ux, Q_uu] = Q_fcn(obj, x, u, V_x, V_xx)
            [l, l_x, l_u, l_xx, l_ux, l_uu] = obj.stage_cost_fcn(x, u);
            [~, f_x, f_u] = obj.dynamics_fcn(x, u);
            
            Q_x = l_x + V_x*f_x;
            Q_u = l_u + V_x*f_u;
            
            % *** OPTIMIZATION: Cache f_u' * V_xx ***
            f_u_T_V_xx = f_u' * V_xx;
            
            Q_xx = l_xx + f_x' * V_xx * f_x;
            Q_ux = l_ux + f_u_T_V_xx * f_x;
            Q_uu = l_uu + f_u_T_V_xx * f_u;
        end
        
        function [u_ff, K, V_x_prev, V_xx_prev] = u_opt_fcn(obj, x, u, V_x, V_xx, lambda)
            [Q_x, Q_u, Q_xx, Q_ux, Q_uu] = obj.Q_fcn(x, u, V_x, V_xx);
            
            % *** ROBUSTNESS: Add regularization ***
            Q_uu_reg = Q_uu + lambda * eye(obj.n_u);
            
            % *** OPTIMIZATION: Combined solve ***
            % Solves for K = -inv(Q_uu_reg) * Q_ux
            % and   u_ff = -inv(Q_uu_reg) * Q_u'
            
            % Check for positive definiteness (optional but good)
            [~, p] = chol(Q_uu_reg);
            if p > 0
                % Not positive definite, something is wrong or needs more lambda
                % For this implementation, we rely on the line search and
                % lambda updates to handle this.
            end
            
            gains = - (Q_uu_reg \ [Q_ux, Q_u']);
            
            K    = gains(:, 1:obj.n_x);
            u_ff = gains(:, (obj.n_x + 1):end);
            
            % Calculate new Value function terms
            V_x_prev = Q_x + Q_u * K;
            V_xx_prev = Q_xx + Q_ux' * K;
        end
        
        function increase_lambda(obj)
            obj.lambda = max(obj.min_lambda, obj.lambda * obj.lambda_factor);
            if obj.lambda > obj.max_lambda
                obj.lambda = obj.max_lambda;
            end
        end
        
        function decrease_lambda(obj)
            obj.lambda = min(obj.max_lambda, obj.lambda / obj.lambda_factor);
            if obj.lambda < obj.min_lambda
                obj.lambda = obj.min_lambda;
            end
        end
        
    end
end