import jax
import jax.numpy as jnp
import numpy as np
from typing import Union
import time
from jax import jit, lax
import matplotlib.pyplot as plt

# Robust import for the System base class
try:
    from .system_base import System
except ImportError:
    try:
        from system_base import System
    except ImportError:
        print("Warning: 'system_base' not found. Ensure it is in the python path.")
        # Minimal Mock for standalone testing if base is missing
        class System:
            def __init__(self, n_q, n_v, n_u, n_c, dt, **kwargs):
                self.n_q = n_q
                self.n_v = n_v
                self.n_u = n_u
                self.n_c = n_c
                self.dt = dt
            def f_fcn(self, x, u): pass

# Import the auto-generated dynamics library
import class_files.systems.dynamics.dual_arm_box_dynamics_lib as sys_lib


class MyDualArmManipulator(System):

    def __init__(
        self,
        dt: float,
        box_target_state: Union[np.ndarray, jnp.ndarray],
        R: jnp.ndarray,
        Q_box: jnp.ndarray,
        RN_list: list, # List of weights for normal gaps
        Q_f: jnp.ndarray,
        RN_f_list: list, # List of terminal weights for normal gaps
        # --- Physical parameters ---
        g: float = 9.81,
        m_EE: float = 0.5,
        theta_EE: float = 0.05,
        m_box: float = 1.0,
        theta_box: float = 0.1,
        w_box: float = 0.4,
        h_box: float = 0.4,
        w_EE: float = 0.05,
        h_EE: float = 0.2,
        # Arm Parameters
        l1: float = 0.5,
        l2: float = 0.5,
        lc1: float = 0.25,
        lc2: float = 0.25,
        m1: float = 0.5,
        m2: float = 0.5,
        theta1: float = 0.05,
        theta2: float = 0.05,
        # Base Positions
        x_base_L: float = -0.6,
        y_base_L: float = 0.0,
        x_base_R: float = 0.6,
        y_base_R: float = 0.0,
        # --- System settings ---
        use_jit: bool = True,
        integrator: str = "contact_euler",
        mu: jnp.ndarray = jnp.zeros(6), # 6 contacts
        smooth_epsilon: float = 1.0,
        e_restitution=jnp.zeros(6),
        **kwargs,
    ):
        """
        State (n_q = 9, n_v = 9):
            q[0:3]: Left Arm (Shoulder, Elbow, Wrist)
            q[3:6]: Right Arm (Shoulder, Elbow, Wrist)
            q[6:9]: Box (x, y, alpha)
            
        Control (n_u = 6):
            tau[0:3]: Left Arm Torques
            tau[3:6]: Right Arm Torques
            
        Contacts (n_c = 6):
            0: Upper1 (Left EE)
            1: Lower1 (Left EE)
            2: Upper2 (Right EE)
            3: Lower2 (Right EE)
            4: Ground Left
            5: Ground Right
        """
        n_q = 9
        n_v = 9
        n_u = 6 
        n_c = 6

        # Physics Parameters
        self.g = g
        # Arm
        self.l1 = l1
        self.l2 = l2
        self.lc1 = lc1
        self.lc2 = lc2
        self.m1 = m1
        self.m2 = m2
        self.m_EE = m_EE
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta_EE = theta_EE
        # Box
        self.m_box = m_box
        self.theta_box = theta_box
        self.w_box = w_box
        self.h_box = h_box
        # EE Geometry
        self.w_EE = w_EE
        self.h_EE = h_EE
        # Bases
        self.x_base_L = x_base_L
        self.y_base_L = y_base_L
        self.x_base_R = x_base_R
        self.y_base_R = y_base_R

        # Cost / Weights
        self.box_target_state = box_target_state
        self.Q_box = Q_box
        self.R = R
        self.RN_list = jnp.array(RN_list)
        self.Q_f = Q_f
        self.RN_f_list = jnp.array(RN_f_list)

        super().__init__(
            n_q,
            n_v,
            n_u,
            n_c,
            dt,
            integrator=integrator,
            mu=mu,
            smooth_epsilon=smooth_epsilon,
            e_restitution=e_restitution,
            **kwargs,
        )

    def _get_dynamics_args(self, q, v, u=None):
        # Signature: q, dq, tau, l1, l2, lc1, lc2, m1, m2, m_EE, theta1, theta2, theta_EE, 
        #            w_box, h_box, m_box, theta_box, w_EE, h_EE, g, x_base_L, y_base_L, x_base_R, y_base_R
        
        # If u is not provided (e.g., for mass matrix calculation), assume zeros
        if u is None:
            u = jnp.zeros(self.n_u)

        return (q, v, u,
                self.l1, self.l2, self.lc1, self.lc2,
                self.m1, self.m2, self.m_EE,
                self.theta1, self.theta2, self.theta_EE,
                self.w_box, self.h_box, self.m_box, self.theta_box,
                self.w_EE, self.h_EE,
                self.g,
                self.x_base_L, self.y_base_L, self.x_base_R, self.y_base_R)

    def _mass_matrix(self, q):
        # Pass dummy v and u
        args = self._get_dynamics_args(q, jnp.zeros(self.n_v))
        M = sys_lib.get_M(*args)
        return M

    def _generalized_forces(self, q, v, u):
        # 1. Calculate Full RHS (Gravity + Coriolis + Actuation)
        args = self._get_dynamics_args(q, v, u)
        # Note: In the dual-arm derivation, get_gen_force includes B*tau.
        # So we pass 'u' directly into the library function.
        f_total = sys_lib.get_gen_force(*args)
        args_static = self._get_dynamics_args(q, jnp.zeros_like(v), jnp.zeros_like(u))
        f_static = sys_lib.get_gen_force(*args_static) 
        
        # 2. Subtract gravity from the total forces for the Arms (Indices 0:6)
        # f_total - f_static = (tau - C - G) - (-G) = tau - C
        # This removes gravity from the arm dynamics physically.
        # f_total = f_total.at[0:6].set(f_total[0:6] - f_static[0:6])

        
        # 2. Add PD Controller (Optional Damping/Stabilization)
        # Note: If PD outputs torques, we should technically add B * u_PD.
        # For simplicity here, we assume u_PD is already in torque space and project it manually
        # or add it to 'u' before passing to library. Here we add to generalized force space.
        
        # Simple joint damping to stabilize the arms
        damping_diag = jnp.array([0.5, 0.5, 0.1, 0.5, 0.5, 0.1, 0.0, 0.0, 0.0])*0.0 # Damping on arms only
        f_damping = -damping_diag * v
        
        # If you want a Task-Space PD like the previous example, it requires inverse kinematics
        # or Jacobian transpose control. For now, we apply joint damping.
        f_total = f_total
        h = f_total + f_damping
        return h

    def _contact_jacobian(self, q):
        args = self._get_dynamics_args(q, jnp.zeros(self.n_v))
        W = sys_lib.get_W(*args)
        return W
    
    def _contact_jacobian_dot_transpose_dqdt(self, q, v):
        args = self._get_dynamics_args(q, v)
        W_dot_T_dqdt = sys_lib.get_W_dot_transpose_dqdt(*args)
        return W_dot_T_dqdt

    def _gap_function(self, q):
        args = self._get_dynamics_args(q, jnp.zeros(self.n_v))
        g_N = sys_lib.get_g_N(*args)
        return g_N

    def _contact_velocity_function(self, q, v):
        args = self._get_dynamics_args(q, v)
        gamma_T = sys_lib.get_gamma_T(*args)
        return gamma_T
    
    def _end_effector_jacobians(self, q):
        args = self._get_dynamics_args(q, jnp.zeros(self.n_v))
        J_EE1 = sys_lib.get_J_EE1(*args)
        J_EE2 = sys_lib.get_J_EE2(*args)
        return J_EE1, J_EE2
    
    def _PD_controller(self, q, v):
        # Simple PD to keep End Effectors near a "ready" pose relative to the box
        # or to dampen velocities.
        args = self._get_dynamics_args(q, v)
        # Extract End Effector Positions
        I_r_OP1 = sys_lib.get_pos_EE1(*args)
        I_r_OP2 = sys_lib.get_pos_EE2(*args)
        J_EE1, J_EE2 = self._end_effector_jacobians(q)
        I_v_OP1 = J_EE1 @ v
        I_v_OP2 = J_EE2 @ v
        phi_box = q[8]
        phi_EE1 = q[0] + q[1] + q[2]
        phi_EE2 = q[3] + q[4] + q[5]

        A_IB_EE1 = jnp.array([[jnp.cos(phi_EE1), -jnp.sin(phi_EE1)],
                              [jnp.sin(phi_EE1),  jnp.cos(phi_EE1)]])
        
        A_IB_EE2 = jnp.array([[jnp.cos(phi_EE2), -jnp.sin(phi_EE2)],
                              [jnp.sin(phi_EE2),  jnp.cos(phi_EE2)]])
        
        A_IB_box = jnp.array([[jnp.cos(phi_box), -jnp.sin(phi_box)],
                              [jnp.sin(phi_box),  jnp.cos(phi_box)]])
        
        I_r_Obox = q[6:8]

        # Kp_pos = 20.0
        # Kp_ang = 20.0
        # Kd_lin = 5.0
        # Kd_ang = 2.0

        # Kp_pos_x = 100.0
        # Kp_pos_y = 500.0
        
        # Kd_lin_x = 10.0
        # Kd_lin_y = 30.0
        # Kp_ang = 20.0
        # Kd_ang = 2.0

        Kp_pos_x = 20.0
        Kp_pos_y = 40.0
        Kd_lin_x = 5.0
        Kd_lin_y = 10.0
        Kp_ang = 5.0
        Kd_ang = 2.0    



        # Desired offsets relative to box center (approximate grasping)
        # EE1 (Left)
        pen_factor = 0.99
        box_pen_factor = 0.7
        des_x1 = q[6] - box_pen_factor*self.w_box/2 - self.w_EE/2 *pen_factor # Slightly to the left
        des_y1 = q[7]
        des_phi1 = q[8]
        
        # EE2 (Right)
        des_x2 = q[6] + box_pen_factor*self.w_box/2 + self.w_EE/2 *pen_factor # Slightly to the right
        des_y2 = q[7]
        # I_r_OP1_des = I_r_Obox + A_IB_box @ jnp.array([-self.w_box/2*0.95, 0.0])
        # I_r_OP2_des = I_r_Obox + A_IB_box @ jnp.array([ self.w_box/2*0.95, 0.0])
        I_r_OP1_des = I_r_Obox + A_IB_box @ jnp.array([-self.w_box/2*0.95, 0.0])
        I_r_OP2_des = I_r_Obox + A_IB_box @ jnp.array([ self.w_box/2*0.95, 0.0])
        des_phi2 = q[8]
        # Errors (EE1)
        e_1 = jnp.array([des_x1, des_y1, des_phi1]) - jnp.concatenate([I_r_OP1, jnp.array([phi_EE1])])
        de_1 = -jnp.concatenate([I_v_OP1, jnp.array([v[0] + v[1] + v[2]])])
        u_1 = jnp.array([Kp_pos_x, Kp_pos_y, Kp_ang]) * e_1 + jnp.array([Kd_lin_x, Kd_lin_y, Kd_ang]) * de_1
        # Errors (EE2)
        e_2 = jnp.array([des_x2, des_y2, des_phi2]) - jnp.concatenate([I_r_OP2, jnp.array([phi_EE2])])
        de_2 = -jnp.concatenate([I_v_OP2, jnp.array([v[3] + v[4] + v[5]])])
        # u_2 = jnp.array([Kp_pos, Kp_pos, Kp_ang]) * e_2 + jnp.array([Kd_lin, Kd_lin, Kd_ang]) * de_2 + jnp.array([-0.5, 0.0, 0.0])
        u_2 = jnp.array([Kp_pos_x, Kp_pos_y, Kp_ang]) * e_2 + jnp.array([Kd_lin_y, Kd_lin_y, Kd_ang]) * de_2
        # Combine into generalized forces (9x1) - applied to EEs only
        # u_PD = jnp.concatenate([u_1, u_2, jnp.zeros(3)])
        F_1 = u_1[:2]
        M_1 = u_1[2]
        F_2 = u_2[:2]
        M_2 = u_2[2]
        f_1 = J_EE1.T @ F_1
        f_2 = J_EE2.T @ F_2
        f_1 = f_1.at[2].add(M_1)
        f_2 = f_2.at[5].add(M_2)
        u_PD = f_1 + f_2
        # u_PD = jnp.concatenate([J_EE1.T @ F_1 + J_EE2.T @ F_2, jnp.zeros(3)])
        args_static = self._get_dynamics_args(q, jnp.zeros_like(v), jnp.zeros(self.n_u))
        f_static = sys_lib.get_gen_force(*args_static) 
        u_PD = u_PD - jnp.hstack([f_static[0:6], jnp.array([0.0,0.0,0.0])]) # Gravity compensation for arms in PD control
        return u_PD
    

    def _l_fcn(self, x, u):
        q = x[: self.n_q]
        v = x[self.n_q :self.n_q + self.n_v]
        
        # Box State: q[6:9], v[6:9]
        x_box = jnp.concatenate([q[6:9], v[6:9]])
        err_box = x_box - self.box_target_state
        
        g_N = self._gap_function(q)
        
        cost_u = u.T @ self.R @ u
        cost_gap = jnp.sum(self.RN_list * (g_N ** 2))
        cost_track = err_box.T @ self.Q_box @ err_box
        
        l = cost_u + cost_gap + cost_track
        return l

    def _l_f_fcn(self, x):
        q = x[: self.n_q]
        v = x[self.n_q :]
        
        x_box = jnp.concatenate([q[6:9], v[6:9]])
        err_box = x_box - self.box_target_state
        
        g_N = self._gap_function(q)
        
        cost_gap = jnp.sum(self.RN_f_list * (g_N ** 2))
        cost_track = err_box.T @ self.Q_f @ err_box
        
        return cost_gap + cost_track
    
    # --- Helper to get Forward Kinematics for Rendering ---
    def get_forward_kinematics(self, q):
        """Returns dictionary of key point positions for plotting."""
        args = self._get_dynamics_args(q, jnp.zeros(self.n_v))
        # Note: args tuple unpacking must match the library function signature exactly.
        # The library FK functions usually take the same args list.
        
        return {
            "base_L": jnp.array([self.x_base_L, self.y_base_L]),
            "joint_L2": sys_lib.get_pos_joint_L2(*args),
            "EE1": sys_lib.get_pos_EE1(*args),
            "base_R": jnp.array([self.x_base_R, self.y_base_R]),
            "joint_R2": sys_lib.get_pos_joint_R2(*args),
            "EE2": sys_lib.get_pos_EE2(*args),
            "box": sys_lib.get_pos_box(*args)
        }
    
    def inverse_kinematics_arms(self, target_pose_L, target_pose_R, q_guess, max_iter=50, tol=1e-4):
        """
        Solves Inverse Kinematics for both arms using Newton-Raphson method.
        
        Args:
            target_pose_L (jnp.ndarray): Target [x, y, theta] for Left EE.
            target_pose_R (jnp.ndarray): Target [x, y, theta] for Right EE.
            q_guess (jnp.ndarray): Initial guess for full state q (size 9).
            max_iter (int): Maximum iterations.
            tol (float): Tolerance for convergence.
            
        Returns:
            q_sol (jnp.ndarray): The solved joint configuration (size 9). 
                                 (Box state q[6:9] is preserved from guess).
            success (bool): True if converged.
        """
        q_curr = jnp.array(q_guess)
        
        # Helper to ensure orientation is between -pi and pi
        def wrap_angle(angle):
            return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi

        for i in range(max_iter):
            # 1. Compute Forward Kinematics for current q
            args = self._get_dynamics_args(q_curr, jnp.zeros(self.n_v))
            
            # Left Arm FK
            pos_L = sys_lib.get_pos_EE1(*args)
            phi_L = q_curr[0] + q_curr[1] + q_curr[2]
            current_pose_L = jnp.concatenate([pos_L, jnp.array([phi_L])])
            
            # Right Arm FK
            pos_R = sys_lib.get_pos_EE2(*args)
            phi_R = q_curr[3] + q_curr[4] + q_curr[5]
            current_pose_R = jnp.concatenate([pos_R, jnp.array([phi_R])])
            
            # 2. Compute Errors
            err_L = current_pose_L - target_pose_L
            err_R = current_pose_R - target_pose_R
            
            # Wrap angle errors
            err_L = err_L.at[2].set(wrap_angle(err_L[2]))
            err_R = err_R.at[2].set(wrap_angle(err_R[2]))
            
            error_norm = jnp.linalg.norm(err_L) + jnp.linalg.norm(err_R)
            if error_norm < tol:
                return q_curr, True

            # 3. Compute Jacobians
            # The library provides position Jacobians (2x3). We append the orientation row [1, 1, 1].
            J_pos_L, J_pos_R = self._end_effector_jacobians(q_curr)
            
            # Augmented Jacobian for Left Arm (3x3)
            # Row 3 is [1, 1, 1] because phi = q0 + q1 + q2
            J_L = jnp.vstack([J_pos_L[:, 0:3], jnp.array([1.0, 1.0, 1.0])])
            
            # Augmented Jacobian for Right Arm (3x3)
            J_R = jnp.vstack([J_pos_R[:, 3:6], jnp.array([1.0, 1.0, 1.0])])

            # 4. Update Steps (Newton-Raphson: q_new = q - J_inv * err)
            # Since these are 3x3 matrices, we can use simple solve or pinv
            # Adding small damping (1e-6) for numerical stability near singularities
            dq_L = jnp.linalg.pinv(J_L, rcond=1e-3) @ err_L
            dq_R = jnp.linalg.pinv(J_R, rcond=1e-3) @ err_R
            
            # Update the arm segments of q (indices 0:3 and 3:6)
            q_curr = q_curr.at[0:3].set(q_curr[0:3] - dq_L)
            q_curr = q_curr.at[3:6].set(q_curr[3:6] - dq_R)
            
        print(f"IK Warning: Did not converge within {max_iter} iterations. Error: {error_norm:.4f}")
        return q_curr, False


if __name__ == "__main__":
    # --- 1. Parameters & Setup ---
    dt = 0.001
    T_sim = 5.0
    
    # Dimensions
    w_box = 0.4
    h_box = 0.4
    
    # Target (Placeholder)
    x_box_target = jnp.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0])

    # Weights
    R = jnp.diag(1e-4 * jnp.ones(6)) 
    Q_box = jnp.diag(jnp.array([10.0, 10.0, 1.0, 1.0, 1.0, 1.0]))
    Q_f = Q_box * 10.0
    RN_list = [100.0] * 6       # Normal penetration penalty
    RN_f_list = [1000.0] * 6    # Terminal penalty
    mu = jnp.array([0.5]*6)     # Friction coefficient

    # Instantiate System
    robot = MyDualArmManipulator(
        dt=dt,
        box_target_state=x_box_target,
        R=R,
        Q_box=Q_box,
        RN_list=RN_list,
        Q_f=Q_f,
        RN_f_list=RN_f_list,
        integrator="moreau",
        w_box=w_box,
        h_box=h_box,
        mu=mu,
        # --- Specify Base Positions Here ---
        x_base_L = -0.8,  # Move Left Arm further left
        y_base_L = 0.7,   # Raise Left Arm base
        x_base_R = 0.8,   # Move Right Arm further right
        y_base_R = 0.7    # Raise Right Arm base
    )

    # --- 2. Initial State ---
    # Left Arm: Shoulder=45deg, Elbow=-90deg
    q_L = jnp.array([-90*jnp.pi/180, 90*jnp.pi/180, 0*jnp.pi/180]) 

    
    # Right Arm: Shoulder=135deg, Elbow=90deg
    q_R = jnp.array([( 90 - 180 )*jnp.pi/180, -90*jnp.pi/180, +180*jnp.pi/180]) 
    
    # Box: Starts slightly in the air
    q_box = jnp.array([0.0, 0.5*robot.h_box, 0.0]) 

    q_0 = jnp.concatenate([q_L, q_R, q_box])
    v_0 = jnp.zeros(9)
    x_0 = jnp.concatenate([q_0, v_0])

    print(f"Initial State q: {q_0}")

    # --- 3. Simulation Loop ---
    N_sim = int(T_sim / dt)
    tspan = jnp.arange(0, T_sim, dt)

    print(f"Simulating {T_sim}s ({N_sim} steps)...")

    X_hist = [x_0]
    x_curr = x_0
    
    # Passive simulation (Zero Torques)
    u_passive = jnp.zeros(6) 
    
    start_time = time.time()
    for _ in range(N_sim):
        x_curr = robot.f_fcn(x_curr, u_passive)
        X_hist.append(x_curr)
    
    print(f"Simulation complete in {time.time() - start_time:.4f}s")

    X_hist = np.array(X_hist)[:len(tspan)]

    # --- 4. Plotting ---
    pos_EE1_hist = []
    pos_EE2_hist = []
    g_N_hist = []
    for i in range(len(X_hist)):
        fk = robot.get_forward_kinematics(X_hist[i, :9])
        pos_EE1_hist.append(fk['EE1'])
        pos_EE2_hist.append(fk['EE2'])
        g_N_hist.append(robot._gap_function(X_hist[i, :9]))
        
    pos_EE1_hist = np.array(pos_EE1_hist)
    pos_EE2_hist = np.array(pos_EE2_hist)
    g_N_hist = np.array(g_N_hist)

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(tspan, X_hist[:, 7], label="Box Y", linewidth=2)
    plt.axhline(h_box/2, color='k', linestyle='--', alpha=0.5, label="Ground")
    plt.title("Box Vertical Motion")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(tspan, pos_EE1_hist[:, 0], label="Left EE X")
    plt.plot(tspan, pos_EE2_hist[:, 0], label="Right EE X")
    plt.plot(tspan, X_hist[:, 6], 'k--', label="Box X")
    plt.title("Horizontal Positions")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    for i in range(g_N_hist.shape[1]):
        plt.plot(tspan, g_N_hist[:, i], label=f"g_N[{i}]")
    plt.title("Normal Gap Functions")
    plt.legend()

    
    plt.tight_layout()
    plt.show()

    # --- 5. Animation ---
    try:
        # Corrected import filename
        from class_files.animations.animation_dual_arm_manipulator import AnimationDualArmBox
        
        print("\nStarting Animation...")
        print("A VTK window should appear. Press 'q' inside the window to exit.")
        
        anim = AnimationDualArmBox(robot, X_hist.T, tspan, dt)
        anim.animate(save_video=False, filename="dual_arm_test.mp4")
        
    except ImportError:
        print("\n[Error] Could not import 'AnimationDualArmBox'.")
        print("Check that 'class_files/animations/animation_dual_arm_manipulation.py' exists.")
    except Exception as e:
        print(f"\n[Error] Animation failed: {e}")