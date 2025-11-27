import jax
import jax.numpy as jnp
import numpy as np
from typing import Union
import time
from jax import jit, lax
import matplotlib.pyplot as plt
from class_files.animations.animation_double_pendulum import AnimationDoublePendulum

# FIX: Robust import to handle both script execution and module import
try:
    # Try relative import (for when imported as a module in run_iLQR...)
    from .system_base import System
except ImportError:
    # Fallback to absolute import (for when running this script directly)
    from system_base import System


class MyDoublePendulum(System):
    """
    JAX-based Double Pendulum system adapted for the Contact-Implicit framework.
    """

    def __init__(
        self,
        dt: float,
        x_target: Union[np.ndarray, jnp.ndarray],
        Q: jnp.ndarray,
        R: jnp.ndarray,
        Q_f: jnp.ndarray,
        # --- Physical parameters ---
        g: float = 9.81,
        m1: float = 1.0,
        m2: float = 1.0,
        l1: float = 1.0,
        l2: float = 1.0,
        d1: float = 0.01,
        d2: float = 0.01,
        theta1: float = 0.0,  # Inertia term
        theta2: float = 0.0,  # Inertia term
        # --- System settings ---
        use_jit: bool = True,
        integrator: str = "rk4",
        mu: float = jnp.array([0.0, 0.0]),
        smooth_epsilon: float = 1.0,
        d_wall: float = 0.1,
        e_restitution=jnp.array([0.0, 0.0]),
        **kwargs,
    ):

        # 1. --- Define system properties ---
        n_q = 2  # [q1, q2]
        n_v = 2  # [q1_dot, q2_dot]
        n_u = 2  # [tau1, tau2]
        n_c = 2

        # Store physical parameters
        self.g = g
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.d1 = d1
        self.d2 = d2
        self.theta1 = theta1
        self.theta2 = theta2

        # 2. --- Store cost parameters as JAX arrays ---
        self.x_target = jnp.asarray(x_target)
        self.Q = jnp.asarray(Q)
        self.R = jnp.asarray(R)
        self.Q_f = jnp.asarray(Q_f)

        # 3. --- Call the base class constructor ---
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

        self.d_wall = d_wall

    # --- Physics Implementation (M, h, W) ---

    def _mass_matrix(self, q: jnp.ndarray) -> jnp.ndarray:
        """Returns M(q) of shape (n_v, n_v)."""
        q1, q2 = q

        # Derived from your provided equations
        m11 = (
            (self.m1 * self.l1**2) / 4
            + self.m2 * self.l1**2
            + (self.m2 * self.l2**2) / 4
            + self.m2 * self.l1 * self.l2 * jnp.cos(q2)
            + self.theta1
            + self.theta2
        )

        m12 = (
            (self.m2 * self.l2**2) / 4
            + (self.m2 * self.l1 * self.l2 * jnp.cos(q2)) / 2
            + self.theta2
        )

        m21 = m12
        m22 = (self.m2 * self.l2**2) / 4 + self.theta2

        M = jnp.array([[m11, m12], [m21, m22]])
        return M

    def _generalized_forces(
        self, q: jnp.ndarray, v: jnp.ndarray, u: jnp.ndarray
    ) -> jnp.ndarray:
        """Returns h(q, v, u) of shape (n_v,)."""
        # Map inputs to your variables
        q1, q2 = q
        q1d, q2d = v
        tau1, tau2 = u

        s1 = jnp.sin(q1)
        s2 = jnp.sin(q2)
        s12 = jnp.sin(q1 + q2)

        # Coriolis / Centripetal Terms (rhs)
        f_c1 = (self.m2 * self.l1 * self.l2 * s2 * (2 * q1d * q2d + q2d**2)) / 2
        f_c2 = -(self.m2 * self.l1 * self.l2 * s2 * (q1d**2)) / 2
        f_c = jnp.array([f_c1, f_c2])

        # Gravity Terms (rhs forces)
        f_g1 = (
            -self.m2 * self.g * (self.l2 * s12 / 2 + self.l1 * s1)
            - (self.m1 * self.g * self.l1 * s1) / 2
        )
        f_g2 = -self.m2 * self.g * (self.l2 * s12 / 2)
        f_g = jnp.array([f_g1, f_g2])

        # Damping Terms
        f_d1 = -self.d1 * q1d
        f_d2 = -self.d2 * q2d
        f_d = jnp.array([f_d1, f_d2])

        # Actuation
        f_act = jnp.array([tau1, tau2])

        # Total generalized forces h where M*v_dot = h
        h = f_act + f_c + f_g + f_d

        return h

    def _contact_jacobian(self, q: jnp.ndarray) -> jnp.ndarray:
        """Returns W(q) of shape (n_v, 2*n_c)."""
        w_T1 = jnp.array([self.l1 * jnp.sin(q[0]), 0])
        w_N1 = jnp.array([-self.l1 * jnp.cos(q[0]), 0])
        w_T2 = jnp.array(
            [
                self.l2 * jnp.sin(q[0] + q[1]) + self.l1 * jnp.sin(q[0]),
                self.l2 + jnp.sin(q[0] + q[1]),
            ]
        )
        w_N2 = jnp.array(
            [
                -self.l2 * jnp.cos(q[0] + q[1]) - self.l1 * jnp.cos(q[0]),
                -self.l2 * jnp.cos(q[0] + q[1]),
            ]
        )
        W = jnp.vstack([w_T1.T, w_N1.T, w_T2.T, w_N2.T]).T
        return W

    def _gap_function(self, q: jnp.ndarray) -> jnp.ndarray:
        """Returns gap vector g(q) of shape (n_c,)."""
        g_N1 = self.d_wall - self.l1 * jnp.sin(q[0])
        g_N2 = self.d_wall - self.l2 * jnp.sin(q[0] + q[1]) - self.l1 * jnp.sin(q[0])
        g_N = jnp.array([g_N1, g_N2])
        return g_N

    def _contact_velocity_function(self, q: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Returns tangential contact velocity gamma(q, v) of shape (n_c,)."""
        gamma_T1 = v[0] * self.l1 * jnp.sin(q[0])
        gamma_T2 = -v[0] * (
            self.l2 * jnp.cos(q[0] + q[1]) + self.l1 * jnp.cos(q[0])
        ) - v[1] * self.l2 * jnp.cos(q[0] + q[1])
        gamma_T = jnp.array([gamma_T1, gamma_T2])
        return gamma_T

    # --- Cost Implementation ---

    def _l_fcn(self, x: jnp.ndarray, u: jnp.ndarray) -> float:
        """Stage cost."""
        dx = x - self.x_target
        cost_x = 0.5 * dx.T @ self.Q @ dx
        cost_u = 0.5 * u.T @ self.R @ u

        val = (cost_x + cost_u) * self.dt
        return val

    def _l_f_fcn(self, x: jnp.ndarray) -> float:
        """Terminal cost."""
        dx = x - self.x_target
        val = 0.5 * dx.T @ self.Q_f @ dx
        return val


# --- Simulation Test ---
if __name__ == "__main__":
    # Parameters
    dt = 0.01
    # Target: Both links upright [pi, 0, 0, 0] if using relative angles where 0 is extended
    # Or whatever your coordinate convention is. Assuming q1=pi is up, q2=0 is straight relative.
    x_target = jnp.array([jnp.pi, 0.0, 0.0, 0.0])

    Q = jnp.diag(jnp.array([10.0, 10.0, 1.0, 1.0]))
    R = jnp.diag(jnp.array([0.1, 0.1]))
    Q_f = jnp.diag(jnp.array([100.0, 100.0, 10.0, 10.0]))

    x_0 = jnp.array([-jnp.pi / 8, -1.0, 2.0, 0.0])  # Start hanging down
    u_0 = jnp.array([0.0, 0.0])  # No torque

    print("--- Testing 'rk4' (Double Pendulum) ---")
    sys_dp = MyDoublePendulum(
        dt=dt,
        d1=0.5,
        d2=0.2,
        x_target=x_target,
        Q=Q,
        R=R,
        Q_f=Q_f,
        integrator="contact_euler",
        e_restitution=jnp.array([0.0, 0.0]),
        d_wall=0.3,
        mu=jnp.array([0.3]),
    )

    x_next = sys_dp.f_fcn(x_0, u_0)
    print(f"Current state: {x_0}")
    print(f"Next state (RK4, zero input): {x_next}")

    # Test derivatives
    print("\nCalculating derivatives (f_x)...")
    f_x = sys_dp.f_x_fcn(x_0, u_0)
    print(f"f_x shape: {f_x.shape}")

    # Simulation
    print("\nSimulating free fall...")
    T_sim = 2.0
    tspan = jnp.arange(0, T_sim, dt)
    N_sim = int(T_sim / dt)

    X_hist = [x_0]
    curr_x = x_0
    for _ in range(N_sim):
        curr_x = sys_dp.f_fcn(curr_x, u_0)
        X_hist.append(curr_x)

    X_hist = np.array(X_hist)

    plt.figure(figsize=(10, 6))
    plt.plot(X_hist[:, 0], label="q1")
    plt.plot(X_hist[:, 1], label="q2")
    plt.legend()
    plt.title("Double Pendulum Free Fall")
    plt.grid(True)
    plt.show()

    anim = AnimationDoublePendulum(sys_dp, X_hist.T, tspan, dt)
    anim.animate(
        save_video=False, filename="double_pendulum_swing_up.mp4", fullscreen=True
    )
