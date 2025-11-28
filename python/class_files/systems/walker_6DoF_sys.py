import jax
import jax.numpy as jnp
import numpy as np
from typing import Union
import time 
import matplotlib.pyplot as plt 
from class_files.systems.dynamics.walking_6DoF_dynamics_lib import *


# Robust import for the System base class
try:
    from .system_base import System
except ImportError:
    try:
        from system_base import System
    except ImportError:
        print("Warning: system_base not found. Define System class or adjust path.")
        class System: pass # Placeholder for linting

class Walking6DoF(System):

    def __init__(self,
                 dt: float,
                 target_state: Union[np.ndarray, jnp.ndarray],
                 Q: jnp.ndarray,
                 R: jnp.ndarray, 
                 Q_f: jnp.ndarray, 
                 # --- Physical parameters ---
                 g: float = 9.81, 
                 m_B: float = 5.0, 
                 m_upper: float = 2.0,
                 m_lower: float = 1.0,
                 theta_upper: float = 0.1,
                 theta_lower: float = 0.05,
                 l_upper: float = 0.5,
                 l_lower: float = 0.5,
                 # --- System settings ---
                 integrator: str = 'contact_euler',
                 mu: jnp.ndarray = jnp.array([0.6, 0.6]), # Friction for 2 feet
                 smooth_epsilon: float = 1.0,
                 e_restitution: jnp.ndarray = jnp.array([0.0, 0.0]),
                 **kwargs):
        
        # Dimensions
        n_q = 6 # x_MB, y_MB, q_u1, q_l1, q_u2, q_l2
        n_v = 6
        n_u = 4 # Hip1, Knee1, Hip2, Knee2
        n_c = 2 # 2 contacts

        self.g = g
        self.m_B = m_B
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
        """Helper to unpack parameters for the auto-generated functions."""
        return (self.m_B, self.m_upper, self.m_lower, 
                self.theta_upper, self.theta_lower, 
                self.l_upper, self.l_lower, self.g)

    def _mass_matrix(self, q):
        # M(q)
        # Note: The generated get_M signature includes dq, but M is usually independent of dq for rigid bodies.
        # We pass zeros for dq to be safe and satisfy the signature.
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
        # Compute tangential velocities: gamma_T = W_T.T @ v
        # W is constructed as [w_T1, w_N1, w_T2, w_N2]
        # Tangential columns are indices 0 and 2
        W = self._contact_jacobian(q)
        W_T = W[:, [0, 2]] 
        gamma_T = W_T.T @ v
        return gamma_T

    def _l_fcn(self, x, u):
        """Running Cost"""
        # Simple quadratic regulation to target state
        err = x - self.target_state
        l = err.T @ self.Q @ err + u.T @ self.R @ u
        return l
    
    def _l_f_fcn(self, x):
        """Terminal Cost"""
        err = x - self.target_state
        l_f = err.T @ self.Q_f @ err
        return l_f

if __name__ == "__main__":
    # --- Parameters ---
    dt = 0.01
    
    # Target: Base at 0.8m height, joints at 0
    q_target = jnp.array([0.0, 0.8, 0.0, 0.0, 0.0, 0.0])
    v_target = jnp.zeros(6)
    x_target = jnp.hstack([q_target, v_target])
    
    # Costs
    Q_diag = jnp.concatenate([
        jnp.array([10.0, 100.0, 1.0, 1.0, 1.0, 1.0]), # q weights
        jnp.array([1.0, 1.0, 0.1, 0.1, 0.1, 0.1])     # v weights
    ])
    Q = jnp.diag(Q_diag)
    Q_f = Q * 10.0
    R = jnp.eye(4) * 0.01

    # --- Instantiate System ---
    robot = Walking6DoF(dt=dt, 
                        target_state=x_target,
                        Q=Q, R=R, Q_f=Q_f,
                        integrator='rk4',
                        mu=jnp.array([1.0, 1.0]), # High friction
                        e_restitution=jnp.array([0.0, 0.0]),
                        g=0)
    
    # --- Initial State ---
    # Start slightly above ground with bent knees
    # q: [x, y, u1, l1, u2, l2]
    # Legs bent: upper +0.5 rad, lower -1.0 rad
    q_0 = jnp.array([0.0, 2.1, -jnp.pi/8, +jnp.pi/8, jnp.pi/8, -jnp.pi/8]) 
    v_0 = jnp.zeros(6)
    x_0 = jnp.hstack([q_0, v_0])
    
    print(f"Initial State: {x_0}")
    
    # Check Dynamics matrices
    print("M(q0):\n", robot._mass_matrix(q_0))
    print("Gap(q0):\n", robot._gap_function(q_0))
    
    # --- Simulation ---
    T_sim = 3.0
    tspan = jnp.arange(0, T_sim, dt)
    N_sim = len(tspan)
    
    print(f"Simulating {T_sim}s ({N_sim} steps)...")
    
    X_hist = [x_0]
    x_curr = x_0
    u_zero = jnp.zeros(4) 
    # Simple PD Controller to hold initial pose
    kp = 200.0
    kd = 5.0
    q_ref = x_0[:6] + jnp.array([0.0,0.0, jnp.pi/4, -jnp.pi/4, -jnp.pi/4, jnp.pi/4]) # Try to stay at initial configuration
    start_time = time.time()
    for _ in range(N_sim):
        
        X_hist.append(x_curr)
        q_curr = x_curr[:6]
        v_curr = x_curr[6:]
        
        # Calculate error only for actuated joints (indices 2,3,4,5)
        q_err = q_ref[2:6] - q_curr[2:6]
        v_err = 0.0 - v_curr[2:6]
        
        u_control = kp * q_err + kd * v_err
        x_curr = robot.f_fcn(x_curr, u_control)
    
    print(f"Simulation finished in {time.time() - start_time:.4f}s")
    
    X_hist = np.array(X_hist)
    X_hist = X_hist[:len(tspan)]

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.subplot(2,1,1)
    plt.plot(tspan, X_hist[:, 1], label="Base Height (y)")
    plt.axhline(0.0, color='k', linestyle='--', label="Ground")
    plt.ylabel("Position [m]")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2,1,2)
    plt.plot(tspan, X_hist[:, 2], label="Upper 1")
    plt.plot(tspan, X_hist[:, 3], label="Lower 1")
    plt.ylabel("Joint Angles [rad]")
    plt.xlabel("Time [s]")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    from class_files.animations.animation_walker_6DoF import AnimationWalking6DoF
    print("\nStarting Animation...")
        # Note: X_hist is (N, 12), Animation expects (12, N)
        # anim = AnimationPointMassBox(manipulator, X_hist.T, tspan, dt)
        
        # anim.animate(save_video=False, 
        #              filename="box_drop.mp4", 
        #              fullscreen=False)
        
    anim = AnimationWalking6DoF(robot, X_hist, tspan, dt)

    # Run Live Preview
    anim.animate(save_video=False, filename="walker_6DoF.mp4", fullscreen=True)