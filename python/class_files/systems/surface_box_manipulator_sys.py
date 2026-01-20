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
import class_files.systems.dynamics.surface_box_dynamics_lib as sys_lib


class MySurfaceBoxManipulator(System):

    def __init__(
        self,
        dt: float,
        box_target_state: Union[np.ndarray, jnp.ndarray],
        R: jnp.ndarray,
        Q_box: jnp.ndarray,
        RN_list: list, # List of weights for normal gaps
        Q_f: jnp.ndarray,
        RN_f_list: list, # List of terminal weights for normal gaps
        Q_track: float = 10.0,
        # --- Physical parameters ---
        g: float = 9.81,
        m_EE: float = 1.0,
        theta_EE: float = 0.1,
        m_box: float = 0.5,
        theta_box: float = 0.1,
        w_box: float = 0.5,
        h_box: float = 0.5,
        w_EE: float = 0.1,
        h_EE: float = 0.3,
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
            q[0:3]: EE1 (x, y, phi)
            q[3:6]: EE2 (x, y, phi)
            q[6:9]: Box (x, y, phi)
            
        Control (n_u = 6):
            u[0:3]: EE1 (Fx, Fy, Tau)
            u[3:6]: EE2 (Fx, Fy, Tau)
            
        Contacts (n_c = 6):
            0: Upper1
            1: Lower1
            2: Upper2
            3: Lower2
            4: Ground Left
            5: Ground Right
        """
        n_q = 9
        n_v = 9
        n_u = 6 
        n_c = 6

        # Physics
        self.g = g
        self.m_EE = m_EE
        self.theta_EE = theta_EE
        self.m_box = m_box
        self.theta_box = theta_box
        self.w_box = w_box
        self.h_box = h_box
        self.w_EE = w_EE
        self.h_EE = h_EE

        # Cost / Weights
        self.box_target_state = box_target_state
        self.Q_box = Q_box
        self.Q_track = Q_track
        self.R = R
        self.RN_list = jnp.array(RN_list) # [RN_upper, RN_lower, RN_ground...]
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

    def _get_dynamics_args(self, q, v):
        # Helper to pack arguments for the library functions
        # Signature: q, dq, w_box, h_box, w_EE, h_EE, m_box, m_EE, theta_box, theta_EE, g
        return (q, v, self.w_box, self.h_box, self.w_EE, self.h_EE, 
                self.m_box, self.m_EE, self.theta_box, self.theta_EE, self.g)

    def _mass_matrix(self, q):
        # We pass a dummy v (zeros) because M usually only depends on q
        args = self._get_dynamics_args(q, jnp.zeros_like(q))
        M = sys_lib.get_M(*args)
        return M

    def _generalized_forces(self, q, v, u):
        # 1. Passive forces (Gravity)
        args = self._get_dynamics_args(q, v)
        f_g = sys_lib.get_gen_force(*args)
        
        # 2. Actuation
        # u is size 6: [Fx1, Fy1, T1, Fx2, Fy2, T2]
        # Map to full generalized force vector (size 9)
        # q order: EE1, EE2, Box. Box is unactuated.
        f_tau = jnp.concatenate([u, jnp.zeros(3)]) 
        
        # 3. PD Controller (Stabilization/Tracking helper)
        u_PD = self._PD_controller(q, v)
        
        # Combine
        h = f_g + f_tau + u_PD
        return h

    def _PD_controller(self, q, v):
        # Simple PD to keep End Effectors near a "ready" pose relative to the box
        # or to dampen velocities.
        
        # Gains
        # Kp_pos = 10.0
        # Kp_ang = 5.0
        # Kd_lin = 2.0
        # Kd_ang = 0.5

        Kp_pos = 10.0
        Kp_ang = 6.0
        Kd_lin = 5.0
        Kd_ang = 2.0
        
        # Desired offsets relative to box center (approximate grasping)
        # EE1 (Left)
        des_x1 = q[6] - self.w_box/2 - self.w_EE/2 *0.99 # Slightly to the left
        des_y1 = q[7]
        # des_phi1 = 0.0
        des_phi1 = q[8]
        
        # EE2 (Right)
        des_x2 = q[6] + self.w_box/2 + self.w_EE/2 *0.99 # Slightly to the right
        des_y2 = q[7]
        # des_phi2 = 0.0
        des_phi2 = q[8]
        # Errors (EE1)
        e_1 = jnp.array([des_x1, des_y1, des_phi1]) - q[0:3]
        de_1 = -v[0:3]

        # u_1 = jnp.array([Kp_pos, Kp_pos, Kp_ang]) * e_1 + jnp.array([Kd_lin, Kd_lin, Kd_ang]) * de_1 + jnp.array([0.5, 0.0, 0.0])
        u_1 = jnp.array([Kp_pos, Kp_pos, Kp_ang]) * e_1 + jnp.array([Kd_lin, Kd_lin, Kd_ang]) * de_1
        # Errors (EE2)
        e_2 = jnp.array([des_x2, des_y2, des_phi2]) - q[3:6]
        de_2 = -v[3:6]
        # u_2 = jnp.array([Kp_pos, Kp_pos, Kp_ang]) * e_2 + jnp.array([Kd_lin, Kd_lin, Kd_ang]) * de_2 + jnp.array([-0.5, 0.0, 0.0])
        u_2 = jnp.array([Kp_pos, Kp_pos, Kp_ang]) * e_2 + jnp.array([Kd_lin, Kd_lin, Kd_ang]) * de_2
        # Combine into generalized forces (9x1) - applied to EEs only
        u_PD = jnp.concatenate([u_1, u_2, jnp.zeros(3)])
        return u_PD
    
    def _contact_jacobian(self, q):
        # The library returns W with interleaved Tangent/Normal columns
        # Shape: (9, 2*n_c) = (9, 12)
        args = self._get_dynamics_args(q, jnp.zeros_like(q))
        W = sys_lib.get_W(*args)
        return W
    
    def _contact_jacobian_dot_transpose_dqdt(self, q, v):
        args = self._get_dynamics_args(q, v)
        W_dot_T_dqdt = sys_lib.get_W_dot_transpose_dqdt(*args)
        return W_dot_T_dqdt

    def _gap_function(self, q):
        args = self._get_dynamics_args(q, jnp.zeros_like(q))
        g_N = sys_lib.get_g_N(*args)
        # g_N vector order from lib: [Upper1, Lower1, Upper2, Lower2, GroundL, GroundR]
        return g_N

    def _contact_velocity_function(self, q, v):
        args = self._get_dynamics_args(q, v)
        gamma_T = sys_lib.get_gamma_T(*args)
        return gamma_T

    def _l_fcn(self, x, u):
        q = x[: self.n_q]
        v = x[self.n_q :]
        
        # Box State: q[6:9], v[6:9] -> indices 6,7,8, 15,16,17
        x_box = jnp.concatenate([q[6:9], v[6:9]])
        err_box = x_box - self.box_target_state
        
        g_N = self._gap_function(q)
        
        # Control regularization
        cost_u = u.T @ self.R @ u
        
        # Gap potential (barrier/penalty)
        # Weighted sum of g_N^2
        cost_gap = jnp.sum(self.RN_list * (g_N ** 2))
        
        # Tracking cost
        cost_track = err_box.T @ self.Q_box @ err_box
        
        # Optional: Regularize EE distance to Box to encourage contact?
        # Handled implicitly by gap costs or explicit Q_track logic if needed.
        
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


if __name__ == "__main__":
    # --- Parameters ---
    dt = 0.001
    
    # Dimensions
    w_box = 0.5
    h_box = 0.5
    w_EE = 0.1
    h_EE = 0.3
    
    # Target: Box lifted to y=1.0, upright (phi=0)
    # State: [x, y, phi, vx, vy, vphi]
    x_box_target = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    # Weights
    # u is size 6
    R = jnp.diag(1e-2 * jnp.ones(6)) 
    
    # Box tracking (x, y, phi, vx, vy, vphi)
    Q_box = jnp.diag(jnp.array([10.0, 10.0, 1.0, 1.0, 1.0, 1.0]))
    Q_f = Q_box * 10.0
    
    # Gap weights (6 contacts)
    RN_list = [10.0] * 6
    RN_f_list = [100.0] * 6
    
    # Friction (6 contacts) - high friction for grasp, low for ground slide?
    # Order: [U1, L1, U2, L2, GL, GR]
    mu = jnp.array([0.8, 0.8, 0.8, 0.8, 1.0, 1.0])*3

    # --- Instantiate System ---
    manipulator = MySurfaceBoxManipulator(
        dt=dt,
        box_target_state=x_box_target,
        R=R,
        Q_box=Q_box,
        RN_list=RN_list,
        Q_f=Q_f,
        RN_f_list=RN_f_list,
        integrator="contact_euler",
        w_box=w_box,
        h_box=h_box,
        w_EE=w_EE,
        h_EE=h_EE,
        m_box=1.0,
        m_EE=1.0,
        mu=mu,
    )

    # --- Initial State ---
    # EE1 (Left)
    q_ee1 = jnp.array([-0.6, 0.25, 1*30*jnp.pi/180])
    # EE2 (Right)
    q_ee2 = jnp.array([0.9, 0.25, 1*30*jnp.pi/180])
    # Box (Center, on ground)
    # h_box/2 is the y-center when on ground
    q_box = jnp.array([0.0, 1*h_box/2, 0.0]) 

    q_0 = jnp.concatenate([q_ee1, q_ee2, q_box])
    v_0 = jnp.zeros(9)
    x_0 = jnp.concatenate([q_0, v_0])

    print(f"Initial State: {x_0}")

    # --- Simulation ---
    T_sim = 2.0
    tspan = jnp.arange(0, T_sim, dt)
    N_sim = len(tspan)

    print(f"Simulating {T_sim}s ({N_sim} steps)...")

    X_hist = [x_0]
    x_curr = x_0
    
    # Apply a squeezing force for the simulation
    # EE1 pushes Right (+Fx), EE2 pushes Left (-Fx)
    # [Fx1, Fy1, T1, Fx2, Fy2, T2]
    force_squeeze = 0.0
    u_squeeze = jnp.array([force_squeeze, 0.0, 0.0, -force_squeeze*1.0, 0.0, 0.0])
    
    start_time = time.time()
    for _ in range(N_sim):
        # The PD controller inside `_generalized_forces` will also be active
        # trying to maintain relative positions.
        x_curr = manipulator.f_fcn(x_curr, u_squeeze)
        X_hist.append(x_curr)
    
    print(f"Simulation finished in {time.time() - start_time:.4f}s")

    X_hist = np.array(X_hist)
    X_hist = X_hist[: len(tspan)]

    # --- Plotting ---
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(tspan, X_hist[:, 7], label="Box Y") # q[7] is y_box
    plt.axhline(h_box/2, color='k', linestyle='--', alpha=0.5, label="Ground")
    plt.title("Box Y Position")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(tspan, X_hist[:, 0], label="EE1 X")
    plt.plot(tspan, X_hist[:, 3], label="EE2 X")
    plt.plot(tspan, X_hist[:, 6], 'k--', label="Box X")
    plt.title("X Positions")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    # --- Animation ---
    try:
        from class_files.animations.animation_surface_box import AnimationSurfaceBox

        print("\nStarting Animation...")
        # Animation expects X shape (N_states, N_timesteps)
        anim = AnimationSurfaceBox(manipulator, X_hist.T, tspan, dt)
        anim.animate(save_video=False, filename="surface_box_test.mp4", fullscreen=True)

    except ImportError:
        print("\n[Warning] 'animation_surface_box.py' not found. Animation skipped.")