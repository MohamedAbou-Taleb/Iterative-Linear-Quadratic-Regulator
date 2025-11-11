classdef Pendulum_System_CLASS < System_CLASS
    properties
        g = 9.81; l = 1.0; d = 0.1;
        x_target; Q; R; Q_f;
    end

    methods
        function obj = Pendulum_System_CLASS(args)
            arguments
                args.g = 9.81;
                args.l = 1;
                args.d = 0.01;
                args.dt
                args.Q
                args.R
                args.Q_f
                args.x_target
            end
            % Assign validated values to object properties

            props = fieldnames(args);

            for k = 1:numel(props)

                obj.(props{k}) = args.(props{k});

            end
            % 1. --- Define system properties ---
            obj.n_x = 2; % [theta, theta_dot]
            obj.n_u = 1; % [tau]

        end

        % ... (All dynamics and cost functions remain identical) ...
        % In Pendulum_System_CLASS

        function val = f_fcn(obj, x, u)
            dt = obj.dt;
            x1 = x(1);
            x2 = x(2);
            u1 = u(1);

            % Continuous-time dynamics
            % x_dot = [x2; u - d*x2 - (g/l)*sin(x1)]
            % (Using g/l, not g*l as in my previous answer, to be more standard)
            x_dot = [
                x2;
                u1 - obj.d * x2 - (obj.g / obj.l) * sin(x1)
                ];

            % Euler integration
            val = x + x_dot * dt;
        end

        function val = f_x_fcn(obj, x, u)
            dt = obj.dt;
            x1 = x(1);

            % Jacobian of continuous-time dynamics (A_c)
            A_c = [
                0,                                 1;
                -(obj.g / obj.l) * cos(x1),      -obj.d
                ];

            % Discrete-time Jacobian (F_x = I + A_c*dt)
            val = eye(obj.n_x) + A_c * dt;
        end

        function val = f_u_fcn(obj, x, u)
            % Jacobian of continuous-time dynamics (B_c)
            B_c = [
                0;
                1
                ];

            % Discrete-time Jacobian (F_u = B_c*dt)
            val = B_c * obj.dt;
        end
        function val = l_fcn(obj, x, u)
            % stage cost
            dx = x - obj.x_target;
            val = (0.5 * dx' * obj.Q * dx + 0.5 * u' * obj.R * u) * obj.dt;
        end
        function val = l_x_fcn(obj, x, u)
            dx = x - obj.x_target;
            val = dx'*obj.Q*obj.dt;
        end
        function val = l_u_fcn(obj, x, u)
            val = u' * obj.R * obj.dt;
        end
        function val = l_xx_fcn(obj, x, u)
            val = obj.Q * obj.dt;
        end
        function val = l_ux_fcn(obj, x, u)
            val = zeros(obj.n_u, obj.n_x);
        end
        function val = l_uu_fcn(obj, x, u)
            val =  obj.R * obj.dt;
        end
        function val = l_f_fcn(obj, x)
            % terminal cost
            dx = x - obj.x_target;
            val = 0.5 * dx' * obj.Q_f * dx;
        end
        function val = l_f_x_fcn(obj, x)
            dx = x - obj.x_target;
            val = dx'*obj.Q_f ;
        end
        function val = l_f_xx_fcn(obj, x)
            val = obj.Q_f;
        end
    end
end