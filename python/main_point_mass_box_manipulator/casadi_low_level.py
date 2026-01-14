import casadi as ca
import numpy as np

class CasadiLowLevelController:
    def __init__(self, manipulator, box_MPC, Q_box_acc, R_box_force, R_tau, C, epsilon):
        """
        Initializes the CasADi optimizer.
        """
        # self.opti = ca.Opti('conic') # Use the Opti stack for easy constraint definition
        self.opti = ca.Opti() # Use the Opti stack for easy constraint definition
        # --- 1. Decision Variables ---
        # Acceleration (6), Torque (4), Contact Forces (4)
        self.ddq = self.opti.variable(6)
        self.tau = self.opti.variable(4)
        self.lam = self.opti.variable(4)

        # --- 2. Parameters (Values that change every step) ---
        self.u_box_ref = self.opti.parameter(2)
        self.ddq_box_ref = self.opti.parameter(2)
        
        # System matrices parameters (M, W, S, h)
        # We make these parameters so you can update dynamics based on state q if desired
        self.P_A = self.opti.parameter(10, 14) # Combined dynamics matrix [M -S -W; W.T 0 0]
        self.P_b = self.opti.parameter(10)     # Combined vector [h; 0]

        # --- 3. Cost Function Formulation ---
        # Note: We need to replicate box_MPC.u_box_of_lambda(_lambda). 
        # Assuming it is a linear map. You might need to adjust the matrix below 
        # based on your specific box_MPC logic. 
        # Here I infer the grasp matrix D based on your provided D variable or logic.
        # If u_box_of_lambda is just D @ lambda:
        D_matrix = np.array([[0.0, 1.0, 0.0, -1.0], [1.0, 0.0, 1.0, 0.0]]) # Example from your code
        u_box_pred = ca.mtimes(D_matrix, self.lam) 

        d_u_box = u_box_pred - self.u_box_ref
        
        # d_ddqdt_box = C @ ddqdt - ddqdt_box_ref
        # C must be converted to numpy/list for CasADi if it is a JAX array
        C_np = np.array(C) 
        d_ddq_box = ca.mtimes(C_np, self.ddq) - self.ddq_box_ref

        # Convert JAX/Numpy weights to CasADi-friendly format
        Q_acc_np = np.array(Q_box_acc)
        R_force_np = np.array(R_box_force)
        R_tau_np = np.array(R_tau)

        # Build Objective
        J = 0.5 * (
            ca.mtimes([d_u_box.T, R_force_np, d_u_box]) +
            ca.mtimes([d_ddq_box.T, Q_acc_np, d_ddq_box]) 
            # ca.mtimes([self.tau.T, R_tau_np, self.tau]) +
            # epsilon * ca.mtimes(self.lam.T, self.lam)
        )
        self.opti.minimize(J)

        # --- 4. Equality Constraints (Dynamics) ---
        # A * [ddq; tau; lam] = b
        vars_stack = ca.vertcat(self.ddq, self.tau, self.lam)
        self.opti.subject_to(ca.mtimes(self.P_A, vars_stack) == self.P_b)

        # --- 5. Inequality Constraints (The main benefit of CasADi) ---
        
        # Example A: Torque limits
        tau_limit = 10.0
        self.opti.subject_to(self.tau <= tau_limit)
        self.opti.subject_to(self.tau >= -tau_limit)

        # Example B: Unilateral contact (Forces can only push)
        # self.opti.subject_to(self.lam >= 0) 

        self.opti.subject_to(self.lam[1] >= 0)  # Normal force at contact 1
        self.opti.subject_to(self.lam[3] >= 0)  # Normal force at contact 2

        # Example C: Friction Cone (Simplified for 2D)
        # assuming lam indices are [n1, t1, n2, t2] or similar. 

        self.opti.subject_to(-manipulator.mu[0] * self.lam[1] <= self.lam[0])
        self.opti.subject_to(manipulator.mu[0] * self.lam[1] >= self.lam[0])
        self.opti.subject_to(-manipulator.mu[1] *self.lam[3] <=  self.lam[2])
        self.opti.subject_to(manipulator.mu[1] * self.lam[3] >= self.lam[2])

        # --- 6. Solver Settings ---
        # 'ipopt' is a robust NLP solver. 'qpoases' is faster for strict QPs.

        opts = {
            'ipopt.print_level': 0, 
            'print_time': 0, 
            'ipopt.sb': 'yes'
        }
        self.opti.solver('ipopt', opts)

        # opts = {'print_time': False, 'osqp': {'verbose': False}}
        # self.opti.solver('osqp', opts)

    def solve(self, u_box_ref_val, ddq_box_ref_val, A_val, b_val):
        # Set parameter values
        self.opti.set_value(self.u_box_ref, u_box_ref_val)
        self.opti.set_value(self.ddq_box_ref, ddq_box_ref_val)
        self.opti.set_value(self.P_A, A_val)
        self.opti.set_value(self.P_b, b_val)

        # Initial guess (optional, but good for convergence)
        # self.opti.set_initial(self.tau, np.zeros(4))

        try:
            sol = self.opti.solve()
            # print objective value
            print("Optimal cost:", sol.value(self.opti.f))
            return sol.value(self.ddq), sol.value(self.tau), sol.value(self.lam)
        except RuntimeError:
            # Handle infeasibility (return zeros or last valid control)
            print("Solver failed! Returning zeros.")
            return np.zeros(6), np.zeros(4), np.zeros(4)