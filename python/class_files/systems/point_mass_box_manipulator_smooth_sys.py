import jax
import jax.numpy as jnp
import numpy as np
from typing import Union
import time
from jax import jit, lax
import matplotlib.pyplot as plt

# Robust import
try:
    from .system_base import System
except ImportError:
    from system_base import System


class MyPointMassBoxManipulatorSmooth(System):

    def __init__(
        self,
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
        # --- Physical parameters ---
        g: float = 9.81,
        m_ball: float = 1.0,
        m_box: float = 1.0,
        box_width: float = 0.5,
        box_height: float = 0.3,
        ball_radius: float = 0.05,
        # --- System settings ---
        use_jit: bool = True,
        integrator: str = "rk4",
        mu: jnp.ndarray = jnp.array([0.3, 0.3, 0.0]),
        smooth_epsilon: float = 1.0,
        e_restitution=jnp.array([0.0, 0.0, 0.0]),
        sigma: jnp.ndarray = jnp.array([1.0, 1.0, 1.0]),
        k_spring: jnp.ndarray = jnp.array([1.0, 1.0, 1.0]),
        v_d: float = 0.1,
        v_s: jnp.ndarray = jnp.array([1.0, 1.0, 1.0]),
        **kwargs,
    ):
        n_q = 6
        n_v = 6
        n_u = 4
        n_c = 3

        self.g = g
        self.m_box = m_box
        self.m_ball = m_ball
        self.box_width = box_width
        self.box_height = box_height
        self.ball_radius = ball_radius

        self.box_target_state = box_target_state
        self.Q_box = Q_box
        self.Q_box_ball = Q_box_ball
        self.R = R
        self.RN1 = RN1
        self.RN2 = RN2
        self.Q_f = Q_f
        self.RN1_f = RN1_f
        self.RN2_f = RN2_f

        self.k_spring = k_spring
        self.sigma = sigma
        self.v_d = v_d
        self.v_s = v_s

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

    def _mass_matrix(self, q):
        M = jnp.block(
            [
                [self.m_ball * jnp.eye(4), jnp.zeros((4, 2))],
                [jnp.zeros((2, 4)), self.m_box * jnp.eye(2)],
            ]
        )
        return M

    def _generalized_forces(self, q, v, u):
        f_g = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, -self.m_box * self.g])
        f_tau = jnp.eye(6, 4) @ u
        W = self._contact_jacobian(q)
        g_N = self._gap_function(q)
        gamma = W.T @ v
        gamma_N = gamma[jnp.array([1,3,5])]
        gamma_T = gamma[jnp.array([0, 2, 4])]
        lam = self.compute_lam(g_N, gamma_N, gamma_T)
        h = f_g + f_tau + W @ lam
        return h

    def _contact_jacobian(self, q):
        w_T1 = jnp.array([0.0, -1.0, 0.0, 0.0, 0.0, 1.0])
        w_N1 = jnp.array([-1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        w_T2 = jnp.array([0.0, 0.0, 0.0, -1.0, 0.0, 1.0])
        w_N2 = jnp.array([0.0, 0.0, 1.0, 0.0, -1.0, 0.0])
        w_T3 = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        w_N3 = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

        W = jnp.vstack([w_T1.T, w_N1.T, w_T2.T, w_N2.T, w_T3.T, w_N3.T]).T
        return W

    def _gap_function(self, q):
        q1, q2, q3, q4, q5, q6 = q
        g_N1 = (q5 - self.box_width / 2) - (q1 + self.ball_radius)
        g_N2 = (q3 - self.ball_radius) - (q5 + self.box_width / 2)
        g_N3 = q6 - self.box_height / 2

        g_N = jnp.array([g_N1, g_N2, g_N3])
        return g_N

    def _contact_velocity_function(self, q, v):
        # FIX: Correct slicing with a list of indices
        WT = self._contact_jacobian(q)[:, [0, 2, 4]]
        gamma_T = WT.T @ v
        return gamma_T

    def _l_fcn(self, x, u):
        q = x[: self.n_q]  # Using self.n_q from base
        # FIX: Correct array indexing with jnp.array or list
        x_box = x[jnp.array([4, 5, 10, 11])]
        err_box = x_box - self.box_target_state
        g_N = self._gap_function(q)  # Pass q to gap function
        l = (
            u.T @ self.R @ u
            + self.RN1 * g_N[0] ** 2
            + self.RN2 * g_N[1] ** 2
            + err_box.T @ self.Q_box @ err_box
            + self.Q_box_ball * ((q[1] - q[5]) ** 2 + (q[3] - q[5]) ** 2)
        )
        return l

    def _l_f_fcn(self, x):
        q = x[: self.n_q]
        # FIX: Correct array indexing
        x_box = x[jnp.array([4, 5, 10, 11])]
        err_box = x_box - self.box_target_state
        g_N = self._gap_function(q)
        l_f = (
            self.RN1_f * g_N[0] ** 2
            + self.RN2_f * g_N[1] ** 2
            + err_box.T @ self.Q_f @ err_box
        )
        return l_f
    
    # @jit
    def smooth_compliance(self, g_N, sigma, k):
        return sigma*k*jnp.log(1+jnp.exp(-g_N/sigma))
    
    # @jit
    def dissipation(self, gamma_N):
        val = gamma_N/self.v_d
        return jnp.where( val < 0, 1-val, jnp.where( val>=2, 0, ((val-2)**2)/4) )
    
    # @jit
    def smooth_normal_force(self, g_N, gamma_N, sigma, k):
        c = self.smooth_compliance(g_N, sigma, k)
        d = self.dissipation(gamma_N)
        lam_N = c*d
        return lam_N

    # @jit
    def smooth_friction_force(self, lam_N, gamma_T, mu, v_s):
        lam_T = -mu*lam_N*(gamma_T)/(jnp.sqrt(v_s**2 + gamma_T**2))
        return lam_T
    
    # @jit
    def compute_lam(self, g_N, gamma_N, gamma_T):
        lam_N1 = self.smooth_normal_force(g_N[0], gamma_N[0], self.sigma[0], self.k_spring[0])
        lam_T1 = self.smooth_friction_force(lam_N1, gamma_T[0], self.mu[0], self.v_s[0])
        lam_N2 = self.smooth_normal_force(g_N[1], gamma_N[1], self.sigma[1], self.k_spring[1])
        lam_T2 = self.smooth_friction_force(lam_N1, gamma_T[1], self.mu[1], self.v_s[1])
        lam_N3 = self.smooth_normal_force(g_N[2], gamma_N[2], self.sigma[2], self.k_spring[2])
        lam_T3 = self.smooth_friction_force(lam_N1, gamma_T[2], self.mu[2], self.v_s[2])

        lam = jnp.array([lam_T1, lam_N1, lam_T2, lam_N2, lam_T3, lam_N3])
        return lam

    


if __name__ == "__main__":
    # --- Parameters ---
    dt = 0.01
    box_width = 0.5
    box_height = 0.3
    ball_radius = 0.05
    x_box_target = jnp.array([0.0, box_height / 2, 0.0, 0.0])

    R = jnp.diag(jnp.array([1.0, 1.0, 1.0, 1.0]))
    Q_box = jnp.diag(jnp.array([1.0, 1.0, 1.0, 1.0]))
    Q_f = jnp.diag(jnp.array([1.0, 1.0, 1.0, 1.0]))
    RN1 = 1.0
    RN2 = 1.0
    RN1_f = 1.0
    RN2_f = 1.0
    m_box = 0.5
    m_ball = 1
    reg_friction = jnp.array([1e-2, 1e-2, 1e-2])

    sigma = jnp.array([1.0, 1.0, 1.0])*1e-2
    k_spring = jnp.array([1.0, 1.0, 1.0])*500
    v_d = 0.1
    v_s = jnp.array([1.0, 1.0, 1.0])
    mu = jnp.array([0.5, 0.5, 0.1])
    # --- Instantiate System ---
    manipulator = MyPointMassBoxManipulatorSmooth(
        dt=dt,
        box_target_state=x_box_target,
        R=R,
        Q_box=Q_box,
        RN1=RN1,
        RN2=RN2,
        Q_f=Q_f,
        RN1_f=RN1_f,
        RN2_f=RN2_f,
        integrator="contact_euler",
        box_height=box_height,
        box_width=box_width,
        ball_radius=ball_radius,
        m_box=m_box,
        m_ball=m_ball,
        mu=mu,
        reg_friction=reg_friction,
        sigma = sigma,
        k_spring=k_spring,
        v_d=v_d,
        v_s=v_s
    )  # mu=0.0 for box-floor to slide

    # --- Initial State ---
    # q = [x_b1, y_b1, x_b2, y_b2, x_box, y_box]
    q_0 = jnp.array(
        [
            -(box_width / 2 + ball_radius) - 0.1,
            0.1,
            box_width / 2 + ball_radius + 0.1,
            0.1,
            0.0,
            2 * box_height / 2,
        ]
    )  # Box starts high (0.5)
    v_0 = jnp.zeros(
        6,
    )
    x_0 = jnp.hstack([q_0, v_0])

    print(f"Initial State: {x_0}")

    # --- Simulation ---
    T_sim = 2.0
    tspan = jnp.arange(0, T_sim, dt)
    N_sim = len(tspan)

    print(f"Simulating {T_sim}s ({N_sim} steps)...")

    X_hist = [x_0]
    x_curr = x_0
    u_zero = jnp.zeros(
        4,
    )  # No actuation on balls
    u_zero = jnp.array([0.0, 0.2, -10.0, 0.2])
    start_time = time.time()
    for _ in range(N_sim):
        x_curr = manipulator.f_fcn(x_curr, u_zero)
        X_hist.append(x_curr)
        # q = x_curr[:manipulator.n_q]
        # v = x_curr[manipulator.n_q:]
        # W = manipulator._contact_jacobian(q)
        # g_N = manipulator._gap_function(q)
        # gamma = W.T @ v
        # gamma_N = gamma[jnp.array([1,3,5])]
        # gamma_T = gamma[jnp.array([0, 2, 4])]
        # lam = manipulator.compute_lam(g_N, gamma_N, gamma_T)
        # print(lam[5])
        # print(W @ lam)
    print(f"Simulation finished in {time.time() - start_time:.4f}s")

    X_hist = np.array(X_hist)
    X_hist = X_hist[: len(tspan)]

    # --- Plotting (Optional) ---
    plt.figure(figsize=(8, 5))
    plt.plot(tspan, X_hist[:, 5], label="Box Y")  # Plot Box Height
    plt.title("Box Height over Time (Gravity Drop)")
    plt.xlabel("Time [s]")
    plt.ylabel("Y Position")
    plt.grid(True)
    plt.legend()
    plt.show()

    # --- Animation ---
    try:
        from class_files.animations.animation_point_mass_box import (
            AnimationPointMassBox,
        )

        print("\nStarting Animation...")
        # Note: X_hist is (N, 12), Animation expects (12, N)
        anim = AnimationPointMassBox(manipulator, X_hist.T, tspan, dt)

        anim.animate(save_video=False, filename="box_drop.mp4", fullscreen=False)

    except ImportError:
        print("\n[Warning] 'animation_point_mass_box.py' not found. Animation skipped.")
