import casadi as ca
import numpy as np

class CasadiLowLevelController:
    def __init__(self, manipulator, box_3DoF_MPC, Q_box_acc, R_box_force, R_tau, C, epsilon):
        """
        Initializes the CasADi optimizer with Slacked Dynamics.
        """
        self.opti = ca.Opti() 

        # --- 1. Decision Variables ---
        # Acceleration (9), Torque (6), Contact Forces (8)
        self.ddq = self.opti.variable(9)
        self.tau = self.opti.variable(6)
        self.lam = self.opti.variable(8)

        # [NEW] Dynamics Slack Variable (Physical Defect)
        # Size 17 matches the rows of your dynamics system [M -S -W; W.T 0 0]
        self.defect = self.opti.variable(17)

        # --- 2. Parameters (Values that change every step) ---
        self.u_box_ref = self.opti.parameter(3)
        self.ddq_box_ref = self.opti.parameter(3)
        
        # System matrices parameters (M, W, S, h)
        self.P_A = self.opti.parameter(17, 23) 
        self.P_b = self.opti.parameter(17)     

        # --- 3. Cost Function Formulation ---
        self.D_matrix = self.opti.parameter(3, 8) 
        u_box_pred = ca.mtimes(self.D_matrix, self.lam) 

        d_u_box = u_box_pred - self.u_box_ref
        
        C_np = np.array(C) 
        d_ddq_box = ca.mtimes(C_np, self.ddq) - self.ddq_box_ref

        # Convert weights
        Q_acc_np = np.array(Q_box_acc)
        R_force_np = np.array(R_box_force)
        R_tau_np = np.array(R_tau)

        # [NEW] Weight for dynamics violation
        # A high weight ensures we only "cheat" physics if absolutely necessary to avoid a crash.
        # 1e6 is a standard starting point for stiff mechanical systems.
        W_defect = 1e6 
        d_ddq_1 = self.ddq[0:3] - self.ddq_box_ref
        d_ddq_2 = self.ddq[3:6] - self.ddq_box_ref
        # Build Objective
        J = 0.5 * (
            ca.mtimes([d_u_box.T, R_force_np, d_u_box]) +
            ca.mtimes([d_ddq_box.T, Q_acc_np, d_ddq_box]) +
            ca.mtimes([self.tau.T, R_tau_np, self.tau]) +
            epsilon * ca.mtimes(self.lam.T, self.lam) +
            # [NEW] Penalize the defect
            W_defect * ca.mtimes(self.defect.T, self.defect)
        )
        # J += 0.5 * ( ca.mtimes([d_ddq_1.T, Q_acc_np, d_ddq_1]) +
        #              ca.mtimes([d_ddq_2.T, Q_acc_np, d_ddq_2]) ) 
        self.opti.minimize(J)

        # --- 4. Equality Constraints (Dynamics) ---
        vars_stack = ca.vertcat(self.ddq, self.tau, self.lam)
        
        # [MODIFIED] Relaxed Dynamics Constraint
        # Old: A * x == b  (Strict Physics)
        # New: A * x == b + defect (Soft Physics)
        # This allows a feasible solution even if friction constraints are limiting.
        self.opti.subject_to(ca.mtimes(self.P_A, vars_stack) == self.P_b + self.defect)


        # --- 5. Inequality Constraints (Strict) ---
        # We keep these strict because we cannot command invalid forces to the real robot.
        
        # Unilateral contact (Normal forces >= 0)
        self.opti.subject_to(self.lam[1] >= 0)  
        self.opti.subject_to(self.lam[3] >= 0)  
        self.opti.subject_to(self.lam[5] >= 0)  
        self.opti.subject_to(self.lam[7] >= 0) 

        # Friction Cones (Simplified for 2D)
        self.opti.subject_to(-manipulator.mu[0] * self.lam[1] <= self.lam[0])
        self.opti.subject_to(manipulator.mu[0] * self.lam[1] >= self.lam[0])
        self.opti.subject_to(-manipulator.mu[1] * self.lam[3] <= self.lam[2])
        self.opti.subject_to(manipulator.mu[1] * self.lam[3] >= self.lam[2])
        self.opti.subject_to(-manipulator.mu[2] * self.lam[5] <= self.lam[4])
        self.opti.subject_to(manipulator.mu[2] * self.lam[5] >= self.lam[4])
        self.opti.subject_to(-manipulator.mu[3] * self.lam[7] <= self.lam[6])
        self.opti.subject_to(manipulator.mu[3] * self.lam[7] >= self.lam[6])

        # --- 6. Solver Settings ---
        opts = {
            'ipopt.print_level': 0, 
            'print_time': 0, 
            'ipopt.sb': 'yes'
        }
        self.opti.solver('ipopt', opts)

    def solve(self, u_box_ref_val, ddq_box_ref_val, A_val, b_val, v=np.zeros(9)):
        # Set parameter values
        W = -A_val[0:9, 15:23]
        baumgarte_gain = 100.0
        gamma = W.T @ v
        b_val[9:17] += -baumgarte_gain * gamma
        self.opti.set_value(self.u_box_ref, u_box_ref_val)
        self.opti.set_value(self.ddq_box_ref, ddq_box_ref_val)
        self.opti.set_value(self.P_A, A_val)
        self.opti.set_value(self.P_b, b_val)

        self.opti.set_value(self.D_matrix, W[6:, :])

        # [OPTIONAL] Warm start logic could go here
        # self.opti.set_initial(self.ddq, prev_ddq)

        # try:
        sol = self.opti.solve()
        
        # [Optional Debug] Check if we are violating physics
        # defect_val = sol.value(self.defect)
        # if np.linalg.norm(defect_val) > 1e-3:
        #    print(f"Warning: Physics violation norm: {np.linalg.norm(defect_val):.4f}")
        # print objective value
        print("Optimal cost:", sol.value(self.opti.f))
        return sol.value(self.ddq), sol.value(self.tau), sol.value(self.lam)
            
        # except RuntimeError:
        #     print("Solver failed! Returning zeros.")
        #     return np.zeros(9), np.zeros(6), np.zeros(8)