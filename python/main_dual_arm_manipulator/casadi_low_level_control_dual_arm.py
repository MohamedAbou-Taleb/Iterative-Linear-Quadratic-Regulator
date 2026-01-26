import casadi as ca
import numpy as np

class CasadiLowLevelControllerDualArm:
    def __init__(self, manipulator, box_3DoF_MPC, Q_box_acc, R_box_force, R_tau, C, epsilon, tau_max=500.0):
        """
        Initializes the CasADi optimizer with Slacked Dynamics for the Dual Arm System.
        
        Args:
            tau_max: Maximum joint torque (Nm).
        """
        self.opti = ca.Opti() 

        # --- 1. Decision Variables ---
        # Acceleration (9): [q_L(3), q_R(3), q_Box(3)]
        self.ddq = self.opti.variable(9)
        # Torque (6): [tau_L(3), tau_R(3)]
        self.tau = self.opti.variable(6)
        # Contact Forces (8): 4 contacts * 2 (Normal/Tangent)
        # Order: [Up1_T, Up1_N, Low1_T, Low1_N, Up2_T, Up2_N, Low2_T, Low2_N]
        self.lam = self.opti.variable(8)

        # Dynamics Slack Variable (Physical Defect)
        # Size 17 matches rows of system: [M -S -W; W.T 0 0] * [v; u; lam] = [b]
        self.defect = self.opti.variable(17)

        # --- 2. Parameters ---
        # References from MPC (High Level)
        self.u_box_ref = self.opti.parameter(3)   # Desired Wrench on Box
        self.ddq_box_ref = self.opti.parameter(3) # Desired Box Acceleration
        
        # System matrices parameters (M, W, S, h)
        self.P_A = self.opti.parameter(17, 23) 
        self.P_b = self.opti.parameter(17)     

        self.u_prev = self.opti.parameter(6)

        # --- 3. Cost Function Formulation ---
        # D_matrix maps contact forces (lam) to Box Wrench
        # Used to ensure the local contact forces actually achieve the high-level MPC wrench
        self.D_matrix = self.opti.parameter(3, 8) 
        u_box_pred = ca.mtimes(self.D_matrix, self.lam) 

        d_u_box = u_box_pred - self.u_box_ref
        
        # C matrix extracts Box Acceleration from full ddq
        C_np = np.array(C) 
        d_ddq_box = ca.mtimes(C_np, self.ddq) - self.ddq_box_ref

        # Convert weights
        Q_acc_np = np.array(Q_box_acc)
        R_force_np = np.array(R_box_force)
        R_tau_np = np.array(R_tau)

        # Weight for dynamics violation (Soft Physics)
        W_defect = 1e6 

        w_smooth = 0.0 # Smoothing weight for joint torques
        
        # Build Objective
        J = 0.5 * (
            ca.mtimes([d_u_box.T, R_force_np, d_u_box]) +        # Track Box Wrench
            ca.mtimes([d_ddq_box.T, Q_acc_np, d_ddq_box]) +      # Track Box Accel
            ca.mtimes([self.tau.T, R_tau_np, self.tau]) +        # Minimize Effort
            epsilon * ca.mtimes(self.lam.T, self.lam) +          # Regularize Forces
            W_defect * ca.mtimes(self.defect.T, self.defect) +   # Penalize Physics Violation
            w_smooth * ca.mtimes([(self.tau - self.u_prev).T, (self.tau - self.u_prev)]) # Smooth Control
        )
        self.opti.minimize(J)

        # --- 4. Equality Constraints (Dynamics) ---
        # Stack: [Acceleration; Torques; Lambda]
        vars_stack = ca.vertcat(self.ddq, self.tau, self.lam)
        # box wrench as constraint
        # self.opti.subject_to(d_ddq_box == 0)
        # self.opti.subject_to(d_u_box == 0)
        # Relaxed Dynamics: A * x == b + defect
        self.opti.subject_to(ca.mtimes(self.P_A, vars_stack) == self.P_b + self.defect)
        # self.opti.subject_to(ca.mtimes(self.P_A, vars_stack) == self.P_b)
        # --- 5. Inequality Constraints ---
        
        # Unilateral contact (Normal forces >= 0)
        # Odd indices are Normal forces in this convention
        F_N = 10.0  # Large upper bound for normal forces
        self.opti.subject_to(self.lam[1] >= F_N)  
        self.opti.subject_to(self.lam[3] >= F_N)  
        self.opti.subject_to(self.lam[5] >= F_N)  
        self.opti.subject_to(self.lam[7] >= F_N) 

        # Friction Cones (Coulomb Friction)
        # |Tangent| <= mu * Normal
        # Note: manipulator.mu[i] corresponds to contact i
        # Contacts: 0:Up1, 1:Low1, 2:Up2, 3:Low2
        
        # Contact 0 (Upper Left)
        self.opti.subject_to(-manipulator.mu[0] * self.lam[1] <= self.lam[0])
        self.opti.subject_to(manipulator.mu[0] * self.lam[1] >= self.lam[0])
        
        # Contact 1 (Lower Left)
        self.opti.subject_to(-manipulator.mu[1] * self.lam[3] <= self.lam[2])
        self.opti.subject_to(manipulator.mu[1] * self.lam[3] >= self.lam[2])
        
        # Contact 2 (Upper Right)
        self.opti.subject_to(-manipulator.mu[2] * self.lam[5] <= self.lam[4])
        self.opti.subject_to(manipulator.mu[2] * self.lam[5] >= self.lam[4])
        
        # Contact 3 (Lower Right)
        self.opti.subject_to(-manipulator.mu[3] * self.lam[7] <= self.lam[6])
        self.opti.subject_to(manipulator.mu[3] * self.lam[7] >= self.lam[6])

        # Torque limits
        self.opti.subject_to(self.tau <= tau_max)
        self.opti.subject_to(self.tau >= -tau_max)

        # --- 6. Solver Settings ---
        opts = {
            'ipopt.print_level': 0, 
            'print_time': 0, 
            'ipopt.sb': 'yes'
        }
        self.opti.solver('ipopt', opts)

    def solve(self, u_box_ref_val, ddq_box_ref_val, A_val, b_val, v=np.zeros(9), u_prev_val=np.zeros(6)):
        # Calculate Baungarte Stabilization for the constraint manifold
        # W corresponds to the top-right block of A related to Lambda
        # A structure: [M -S -W; W.T 0 0]
        # W is located at A[0:9, 15:23] (Indices based on variable stack size)
        W = -A_val[0:9, 15:23]
        
        baumgarte_gain = 100.0
        # gamma = J * v (Constraint velocity)
        gamma = W.T @ v
        # Stabilize b term for constraint rows (rows 9 to 17)
        b_val[9:17] += -baumgarte_gain * gamma

        # Set Parameters
        self.opti.set_value(self.u_box_ref, u_box_ref_val)
        self.opti.set_value(self.ddq_box_ref, ddq_box_ref_val)
        self.opti.set_value(self.P_A, A_val)
        self.opti.set_value(self.P_b, b_val)
        
        # D_matrix extracts the wrench from contact Jacobian W
        # The rows of W corresponding to the Box DOFs (6,7,8) relate forces to Box Wrench
        self.opti.set_value(self.D_matrix, W[6:, :])

        self.opti.set_value(self.u_prev, u_prev_val)
        
        try:
            sol = self.opti.solve()
            print("Optimal cost:", sol.value(self.opti.f))
            return sol.value(self.ddq), sol.value(self.tau), sol.value(self.lam)
        except RuntimeError:
            print("Solver failed! Returning previous control/zeros.")
            return np.zeros(9), u_prev_val, np.zeros(8)