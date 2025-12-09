import jax
import jax.numpy as jnp
import numpy as np
from typing import Union
import time 
from jax import jit, lax 
import matplotlib.pyplot as plt 
from class_files.systems.dynamics.point_box_7DoF_dynamics_lib import *

# Robust import
try:
    from .system_base import System
except ImportError:
    from system_base import System

class MyPointMassBoxManipulator7DoF(System):

    def __init__(self,
                 dt: float,
                 box_target_state: Union[np.ndarray, jnp.ndarray],
                 R: jnp.ndarray, 
                 Q_box: jnp.ndarray,
                 RN1: float,
                 RN2: float,
                 Q_f: jnp.ndarray, 
                 RN1_f: float,
                 RN2_f: float,
                 Q_box_ball: float = 100.0,
                 Q_vel: float = 0.1,
                 # --- Physical parameters ---
                 g: float = 9.81, 
                 m_ball: float = 1.0, 
                 m_box: float = 1.0,
                 box_width: float = 0.5,
                 box_height: float = 0.3,
                 ball_radius: float = 0.05,
                 # --- System settings ---
                 use_jit: bool = True,
                 integrator: str = 'contact_euler',
                 mu: jnp.ndarray = jnp.array([0.3, 0.3, 0.0, 0.0]),
                 smooth_epsilon: float = 1.0,
                 e_restitution=jnp.array([0.0, 0.0, 0.0, 0.0]),
                 **kwargs):
        n_q = 7
        n_v = 7
        n_u = 4
        n_c = 4

        self.g = g
        self.m_box = m_box
        self.m_ball = m_ball
        self.box_width = box_width
        self.box_height = box_height
        self.theta_box = self.m_box*(box_width**2 + box_height**2)
        self.ball_radius = ball_radius

        self.box_target_state = box_target_state
        self.Q_box = Q_box
        self.Q_box_ball = Q_box_ball
        self.Q_vel = Q_vel
        self.R = R
        self.RN1 = RN1
        self.RN2 = RN2
        self.Q_f = Q_f
        self.RN1_f = RN1_f
        self.RN2_f = RN2_f

        super().__init__(n_q, n_v, n_u, n_c, dt,
                         integrator=integrator,
                         mu=mu,
                         smooth_epsilon=smooth_epsilon,
                         e_restitution=e_restitution,
                         **kwargs)


    def _mass_matrix(self, q):
        M = get_M(q, jnp.zeros(self.n_v), self.box_width, self.box_height, self.m_box, self.m_ball, self.ball_radius, self.theta_box, self.g)
        return M
    
    def _generalized_forces(self, q, v, u):
        f_g = get_gen_force(q, v, self.box_width, self.box_height, self.m_box, self.m_ball, self.ball_radius, self.theta_box, self.g)
        
        
        u_PD = self._PD_controller(q, v)
        u += u_PD

        f_tau = jnp.eye(7,4) @ u

        h = f_g + f_tau
        return h
    
    def _PD_controller(self, q, v):
        K_p = jnp.diag(jnp.array([10.0, 2.0]))*5
        K_d = jnp.diag(jnp.array([10.0, 100.0]))
        # I_r_Oball_ref = q[4:6] + jnp.array([0.0, 1.0])
        I_r_Oball_ref = q[4:6] + self.box_target_state[0:2]
        I_r_Oball_1 = q[0:2]
        I_r_Oball_2 = q[2:4]
        I_v_ball_1 = v[0:2]
        I_v_ball_2 = v[2:4]
        err1 = I_r_Oball_ref - I_r_Oball_1
        err2 = I_r_Oball_ref - I_r_Oball_2
        derr1 = -I_v_ball_1
        derr2 = -I_v_ball_2
        u_PD = jnp.hstack([K_p @ err1 + K_d @ derr1, K_p @ err2 + K_d @ derr2]) + jnp.array([5.0, 0.0, -5.0, 0.0])
        return u_PD
    
    def _contact_jacobian(self, q):
        W = get_W(q, jnp.zeros(self.n_v), self.box_width, self.box_height, self.m_box, self.m_ball, self.ball_radius, self.theta_box, self.g)
        return W
    
    def _gap_function(self, q):
        g_N = get_g_N(q, jnp.zeros(self.n_v), self.box_width, self.box_height, self.m_box, self.m_ball, self.ball_radius, self.theta_box, self.g)
        return g_N
    
    def _contact_velocity_function(self, q, v):
        # FIX: Correct slicing with a list of indices
        WT = self._contact_jacobian(q)[:, [0, 2, 4]]
        gamma_T = WT.T @ v
        return gamma_T

    def _l_fcn(self, x, u):
        q = x[:self.n_q] # Using self.n_q from base
        v = x[self.n_q:]
        # FIX: Correct array indexing with jnp.array or list
        x_box = x[jnp.array([4, 5, 6, 11, 12, 13])]
        err_box = x_box - self.box_target_state
        g_N = self._gap_function(q) # Pass q to gap function
        B_r_P1ball1 = get_B_r_P1ball1(q, jnp.zeros(self.n_v), self.box_width, self.box_height, self.m_box, self.m_ball, self.ball_radius, self.theta_box, self.g)
        B_r_P2ball2 = get_B_r_P2ball2(q, jnp.zeros(self.n_v), self.box_width, self.box_height, self.m_box, self.m_ball, self.ball_radius, self.theta_box, self.g)
        # dx_B = jnp.array([B_r_P1ball1[0], B_r_P2ball2[0]])
        dx_B = g_N[0:2]
        dy_B = jnp.array([B_r_P1ball1[1], B_r_P2ball2[1]])
        RN = jnp.diag(jnp.array([self.RN1, self.RN2]))
        # u_ref = jnp.array([5.0, 0.0, -5.0, 0.0])
        # du = u-u_ref
        # K_p = jnp.diag(jnp.array([10.0, 2.0]))*2
        # K_d = jnp.diag(jnp.array([10.0, 100.0]))    
        # I_r_Oball_ref = x[4:6] + jnp.array([0.0, 1.0])
        # I_r_Oball_1 = x[0:2]
        # I_r_Oball_2 = x[2:4]
        # I_v_ball_1 = x[7:9]
        # I_v_ball_2 = x[9:11]
        # err1 = I_r_Oball_ref - I_r_Oball_1
        # err2 = I_r_Oball_ref - I_r_Oball_2
        # derr1 = -I_v_ball_1
        # derr2 = -I_v_ball_2
        # u_ref = jnp.hstack([K_p @ err1 + K_d @ derr1, K_p @ err2 + K_d @ derr2]) + jnp.array([5.0, 0.0, -5.0, 0.0])
        # du = u_ref - u

        du = u
        l = du.T @ self.R @ du + dx_B.T @ RN @ dx_B + self.Q_box_ball*dy_B.T@dy_B + err_box.T @ self.Q_box @ err_box + self.Q_vel*v.T@v
        # l = u.T @ self.R @ u + dx_B.T @ RN @ dx_B + self.Q_box_ball*dy_B.T@dy_B + err_box.T @ self.Q_box @ err_box + self.Q_vel*v.T@v
        return l
    
    def _l_f_fcn(self, x):
        q = x[:self.n_q]
        # FIX: Correct array indexing
        x_box = x[jnp.array([4, 5, 6, 11, 12, 13])]
        err_box = x_box - self.box_target_state
        g_N = self._gap_function(q)
        l_f = err_box.T @ self.Q_f @ err_box
        return l_f

if __name__ == "__main__":
    # --- Parameters ---
    dt = 0.001
    box_width = 0.5
    box_height = 0.3
    ball_radius = 0.05
    x_box_target = jnp.array([0.0, box_height/2, 0.0,
                               0.0, 0.0, 0.0])
    
    R = jnp.diag(jnp.array([1.0, 1.0, 1.0, 1.0]))
    Q_box = jnp.diag(jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    Q_f = jnp.diag(jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    RN1 = 1.0; RN2 = 1.0; RN1_f = 1.0; RN2_f = 1.0
    m_box = 0.5
    m_ball = 1
    reg_friction = jnp.array([1e-2, 1e-2, 1e-2])
    e_restitution = jnp.array([0.0, 0.0, 0.0, 0.0])
    mu = jnp.array([0.5, 0.5, 0.1, 0.1])
    # --- Instantiate System ---
    manipulator = MyPointMassBoxManipulator7DoF(dt=dt, 
                                            box_target_state=x_box_target, 
                                            R=R, Q_box=Q_box, RN1=RN1, RN2=RN2,
                                            Q_f=Q_f, RN1_f=RN1_f, RN2_f=RN2_f,
                                            integrator='moreau',
                                            box_height=box_height,
                                            box_width=box_width,
                                            ball_radius=ball_radius,
                                            m_box=m_box,
                                            m_ball=m_ball,
                                            mu=mu,
                                            e_restitution=e_restitution,
                                            reg_friction=reg_friction) # mu=0.0 for box-floor to slide
    
    print(manipulator._mass_matrix(jnp.zeros(manipulator.n_q)))
    print(manipulator._generalized_forces(jnp.zeros(manipulator.n_q,), jnp.zeros(manipulator.n_v,), jnp.zeros(manipulator.n_u,)))
    # --- Initial State ---
    # q = [x_b1, y_b1, x_b2, y_b2, x_box, y_box]
    q_0 = jnp.array([-(box_width/2 + ball_radius) -0.4, 0.1,
                      box_width/2 + ball_radius + 0.4, 0.1, 
                      0.0, 1*box_height/2, jnp.pi/4*0.0]) # Box starts high (0.5)
    v_0 = jnp.zeros(7,)
    x_0 = jnp.hstack([q_0, v_0])
    
    print(f"Initial State: {x_0}")
    
    # --- Simulation ---
    T_sim = 10.0
    tspan = jnp.arange(0, T_sim, dt)
    N_sim = len(tspan)
    
    print(f"Simulating {T_sim}s ({N_sim} steps)...")

    print(manipulator._l_fcn(x_0, jnp.zeros(4,)))
    
    X_hist = [x_0]
    gN_hist = [manipulator._gap_function(q_0)]
    x_curr = x_0
    u_zero = jnp.zeros(4,) # No actuation on balls
    # u_zero = jnp.array([100.0, 0.2, -100.0, 0.2])
    K_p = jnp.diag(jnp.array([10.0, 2.0]))*2
    K_d = jnp.diag(jnp.array([10.0, 100.0]))
    start_time = time.time()
    for _ in range(N_sim):
        I_r_Oball_ref = x_curr[4:6] + jnp.array([0.0, 1.0])
        I_r_Oball_1 = x_curr[0:2]
        I_r_Oball_2 = x_curr[2:4]
        I_v_ball_1 = x_curr[7:9]
        I_v_ball_2 = x_curr[9:11]
        err1 = I_r_Oball_ref - I_r_Oball_1
        err2 = I_r_Oball_ref - I_r_Oball_2
        derr1 = -I_v_ball_1
        derr2 = -I_v_ball_2
        u = jnp.hstack([K_p @ err1 + K_d @ derr1, K_p @ err2 + K_d @ derr2]) + jnp.array([5.0, 0.0, -5.0, 0.0])
        x_curr = manipulator.f_fcn(x_curr, u)
        X_hist.append(x_curr)
        gN_curr = manipulator._gap_function(x_curr[:manipulator.n_q])
        gN_hist.append(gN_curr)
        

    print(f"Simulation finished in {time.time() - start_time:.4f}s")
    
    X_hist = np.array(X_hist)
    X_hist = X_hist[:len(tspan)]
    gN_hist = np.array(gN_hist)
    gN_hist = gN_hist[:len(tspan)]
    # --- Plotting (Optional) ---
    plt.figure(figsize=(8, 5))
    plt.plot(tspan, X_hist[:, 5], label="Box Y") # Plot Box Height
    plt.title("box y position")
    plt.xlabel("Time [s]")
    plt.ylabel("Y Position")
    plt.grid(True)
    plt.legend()
    plt.figure(figsize=(8, 5))
    plt.plot(tspan, gN_hist[:, 2], label="left corner") # Plot Box Height
    plt.plot(tspan, gN_hist[:, 3], label="right corner") # Plot Box Height
    plt.title("box gap functions")
    plt.xlabel("Time [s]")
    plt.ylabel("Y Position")
    plt.grid(True)
    plt.legend()
    plt.show()


    # --- Animation ---
    try:
        from class_files.animations.animation_point_mass_box_7DoF import AnimationPointMassBox7DoF
        print("\nStarting Animation...")
        # Note: X_hist is (N, 12), Animation expects (12, N)
        # anim = AnimationPointMassBox(manipulator, X_hist.T, tspan, dt)
        
        # anim.animate(save_video=False, 
        #              filename="box_drop.mp4", 
        #              fullscreen=False)
        
        anim = AnimationPointMassBox7DoF(manipulator, X_hist, tspan, dt)

    # Run Live Preview
        anim.animate(save_video=False, filename="box_7dof_test.mp4", fullscreen=True)
                     
    except ImportError:
        print("\n[Warning] 'animation_point_mass_box.py' not found. Animation skipped.")