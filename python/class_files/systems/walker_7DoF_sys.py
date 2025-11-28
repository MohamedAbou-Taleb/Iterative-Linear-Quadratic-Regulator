import jax
import jax.numpy as jnp
import numpy as np
from typing import Union
# Ensure this import matches your generated file name
from class_files.systems.dynamics.walking_7DoF_dynamics_lib import *

# Robust import for the System base class
try:
    from .system_base import System
except ImportError:
    try:
        from system_base import System
    except ImportError:
        print("Warning: system_base not found. Define System class or adjust path.")
        class System: pass

class Walking7DoF(System):

    def __init__(self,
                 dt: float,
                 target_state: Union[np.ndarray, jnp.ndarray],
                 Q: jnp.ndarray,
                 R: jnp.ndarray, 
                 Q_f: jnp.ndarray, 
                 # --- Physical parameters ---
                 g: float = 9.81, 
                 m_B: float = 5.0,
                 theta_B: float = 0.5, # Base Inertia (Required for 7-DoF)
                 m_upper: float = 2.0,
                 m_lower: float = 1.0,
                 theta_upper: float = 0.1,
                 theta_lower: float = 0.05,
                 l_upper: float = 0.5,
                 l_lower: float = 0.5,
                 # --- System settings ---
                 integrator: str = 'contact_euler',
                 mu: jnp.ndarray = jnp.array([0.6, 0.6]),
                 smooth_epsilon: float = 1.0,
                 e_restitution: jnp.ndarray = jnp.array([0.0, 0.0]),
                 **kwargs):
        
        # Dimensions for 7-DoF Model
        # q: [x_MB, y_MB, theta_B, q_hip1, q_knee1, q_hip2, q_knee2]
        n_q = 7
        n_v = 7
        n_u = 4  # Hip1, Knee1, Hip2, Knee2 (act on relative coords)
        n_c = 2  # 2 Contacts

        self.g = g
        self.m_B = m_B
        self.theta_B = theta_B
        self.m_upper = m_upper
        self.m_lower = m_lower
        self.theta_upper = theta_upper
        self.theta_lower = theta_lower
        self.l_upper = l_upper
        self.l_lower = l_lower

        self.target_state = target_state
        self.Q = Q
        self.Q_f = Q_f
        self.R = R

        super().__init__(n_q, n_v, n_u, n_c, dt,
                         integrator=integrator,
                         mu=mu,
                         smooth_epsilon=smooth_epsilon,
                         e_restitution=e_restitution,
                         **kwargs)

    def _get_params(self):
        """
        Unpack parameters matching the generated library signature.
        Order: m_B, theta_B, m_upper, m_lower, theta_upper, theta_lower, l_upper, l_lower, g
        """
        return (self.m_B, self.theta_B, 
                self.m_upper, self.m_lower, 
                self.theta_upper, self.theta_lower, 
                self.l_upper, self.l_lower, self.g)

    def _mass_matrix(self, q):
        # M(q)
        v_dummy = jnp.zeros(self.n_v)
        args = (q, v_dummy) + self._get_params()
        M = get_M(*args)
        return M
    
    def _generalized_forces(self, q, v, u):
        args = (q, v) + self._get_params()
        
        # Nonlinear effects (Coriolis + Gravity)
        f_cg = get_f_cg(*args)
        
        # Actuation Matrix
        B = get_B(*args)
        
        # Total generalized forces
        f_tau = B @ u
        h = f_cg + f_tau
        return h
    
    def _contact_jacobian(self, q):
        # W(q)
        v_dummy = jnp.zeros(self.n_v)
        args = (q, v_dummy) + self._get_params()
        W = get_W(*args)
        return W
    
    def _gap_function(self, q):
        # g_N(q)
        v_dummy = jnp.zeros(self.n_v)
        args = (q, v_dummy) + self._get_params()
        g_N = get_g_N(*args)
        return g_N
    
    def _contact_velocity_function(self, q, v):
        # Tangential velocities: gamma_T = W_T.T @ v
        # W = [w_T1, w_N1, w_T2, w_N2]
        # Tangential columns are indices 0 and 2
        W = self._contact_jacobian(q)
        W_T = W[:, [0, 2]] 
        gamma_T = W_T.T @ v
        return gamma_T

    def _l_fcn(self, x, u):
        """Running Cost"""
        err = x - self.target_state
        l = err.T @ self.Q @ err + u.T @ self.R @ u
        return l
    
    def _l_f_fcn(self, x):
        """Terminal Cost"""
        err = x - self.target_state
        l_f = err.T @ self.Q_f @ err
        return l_f
    
if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    from class_files.animations.animation_walker_7DoF import AnimationWalking7DoF

    # --- Parameters ---
    dt = 0.005 # Smaller dt for better stability with stiff PD
    
    # Target: Base at 0.8m height, upright, joints at 0
    # 7-DoF: [x, y, theta_B, hip1, knee1, hip2, knee2]
    q_target = jnp.array([0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0])
    v_target = jnp.zeros(7)
    x_target = jnp.hstack([q_target, v_target])
    
    # Costs (Updated for 7 states)
    Q_diag = jnp.concatenate([
        # q: x, y, theta, h1, k1, h2, k2
        jnp.array([10.0, 100.0, 10.0, 1.0, 1.0, 1.0, 1.0]), 
        # v:
        jnp.array([1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1])     
    ])
    Q = jnp.diag(Q_diag)
    Q_f = Q * 10.0
    R = jnp.eye(4) * 0.01

    # --- Instantiate System ---
    # Using g=0 to test momentum conservation (no rocketing)
    robot = Walking7DoF(dt=dt, 
                        target_state=x_target,
                        Q=Q, R=R, Q_f=Q_f,
                        integrator='moreau', # Smooth integrator for free-floating test
                        mu=jnp.array([1.0, 1.0]),
                        e_restitution=jnp.array([0.0, 0.0]),
                        g=9.81,         # Zero Gravity
                        theta_B=1.0)   # Significant Base Inertia
    
    # --- Initial State ---
    # Start high up (2.0m) with legs slightly bent
    # q: [x, y, theta_B, h1, k1, h2, k2]
    q_0 = jnp.array([0.0, 2.0, 0.0, -0.2, 0.4, 0.2, 0.4]) 
    v_0 = jnp.zeros(7)
    x_0 = jnp.hstack([q_0, v_0])
    
    print(f"Initial State: {x_0}")
    
    # Check Dynamics matrices
    print("M(q0):\n", robot._mass_matrix(q_0))
    print("Gap(q0):\n", robot._gap_function(q_0))
    
    # --- Simulation ---
    T_sim = 2.0
    tspan = jnp.arange(0, T_sim, dt)
    N_sim = len(tspan)
    
    print(f"Simulating {T_sim}s ({N_sim} steps)...")
    
    X_hist = [x_0]
    x_curr = x_0
    
    # Simple PD Controller to move legs
    kp = 150.0
    kd = 5.0
    
    # Reference: Kick legs (Change hips and knees)
    # q indices: 0:x, 1:y, 2:theta, 3:h1, 4:k1, 5:h2, 6:k2
    # We add offsets to the relative joints
    target_offset = jnp.array([0.0, 0.0, 0.0, 0.5, -0.5, -0.5, 0.5])*0
    q_ref = q_0 + target_offset

    start_time = time.time()
    for _ in range(N_sim):
        X_hist.append(x_curr)
        q_curr = x_curr[:7]
        v_curr = x_curr[7:]
        
        # Calculate error only for actuated joints (indices 3,4,5,6)
        # 3:Hip1, 4:Knee1, 5:Hip2, 6:Knee2
        q_err = q_ref[3:7] - q_curr[3:7]
        v_err = 0.0 - v_curr[3:7]
        
        u_control = kp * q_err + kd * v_err
        
        # Apply Control
        x_curr = robot.f_fcn(x_curr, u_control)
    
    print(f"Simulation finished in {time.time() - start_time:.4f}s")
    
    X_hist = np.array(X_hist)
    X_hist = X_hist[:len(tspan)]

    # --- Plotting ---
    plt.figure(figsize=(10, 8))
    
    # 1. Base Height (Should stay roughly constant if momentum is conserved linear)
    plt.subplot(3,1,1)
    plt.plot(tspan, X_hist[:, 1], label="Base Y")
    plt.plot(tspan, X_hist[:, 0], label="Base X")
    plt.title("Base Position (Should be constant/drifting slowly, NO ROCKETING)")
    plt.legend()
    plt.grid(True)
    
    # 2. Base Pitch (Should rotate opposite to legs)
    plt.subplot(3,1,2)
    plt.plot(tspan, X_hist[:, 2], color='r', label="Base Pitch (Theta_B)")
    plt.ylabel("Radians")
    plt.legend()
    plt.grid(True)
    
    # 3. Joints
    plt.subplot(3,1,3)
    plt.plot(tspan, X_hist[:, 3], label="Hip 1")
    plt.plot(tspan, X_hist[:, 4], label="Knee 1")
    plt.ylabel("Joint Angles [rad]")
    plt.xlabel("Time [s]")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    print("\nStarting Animation...")
    anim = AnimationWalking7DoF(robot, X_hist, tspan, dt)
    anim.animate(save_video=False, filename="walker_7DoF_floating.mp4", fullscreen=True)