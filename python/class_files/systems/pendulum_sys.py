import jax
import jax.numpy as jnp
import numpy as np
from typing import Sequence, Union
import time
from jax import jit, lax
import matplotlib.pyplot as plt

# FIX: Robust import to handle both script execution and module import
try:
    # Try relative import (for when imported as a module in run_iLQR...)
    from .system_base import System
except ImportError:
    # Fallback to absolute import (for when running this script directly)
    from system_base import System


class MyPendulum(System):
    """
    JAX-based Pendulum system adapted for the Contact-Implicit framework.
    """

    def __init__(
        self,
        dt: float,
        x_target: Union[np.ndarray, jnp.ndarray],
        Q: jnp.ndarray,
        R: jnp.ndarray,
        Q_f: jnp.ndarray,
        g: float = 9.81,
        l: float = 1.0,
        d: float = 0.01,
        use_jit: bool = True,
        integrator: str = "rk4",
        mu: float = 0.0,
        smooth_epsilon: float = 1.0,
        **kwargs,
    ):

        # 1. --- Define system properties ---
        n_q = 1  # [theta]
        n_v = 1  # [theta_dot]
        n_u = 1  # [tau]
        n_c = 0  # No contacts

        self.g = g
        self.l = l
        self.d = d

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
            **kwargs,
        )

    # --- Physics Implementation (M, h, W) ---

    def _mass_matrix(self, q: jnp.ndarray) -> jnp.ndarray:
        return jnp.eye(1)

    def _generalized_forces(
        self, q: jnp.ndarray, v: jnp.ndarray, u: jnp.ndarray
    ) -> jnp.ndarray:
        theta = q[0]
        theta_dot = v[0]
        tau = u[0]
        force = tau - self.d * theta_dot - (self.g / self.l) * jnp.sin(theta)
        return jnp.array([force])

    def _contact_jacobian(self, q: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros((self.n_v, 0))

    def _gap_function(self, q: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros((0,))

    def _contact_velocity_function(self, q: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros((0,))

    # --- Cost Implementation ---

    def _l_fcn(self, x: jnp.ndarray, u: jnp.ndarray) -> float:
        dx = x - self.x_target
        cost_x = 0.5 * dx.T @ self.Q @ dx
        cost_u = 0.5 * u.T @ self.R @ u
        val = (cost_x + cost_u) * self.dt
        return val

    def _l_f_fcn(self, x: jnp.ndarray) -> float:
        dx = x - self.x_target
        val = 0.5 * dx.T @ self.Q_f @ dx
        return val


# --- Simulation Test ---
if __name__ == "__main__":
    # Test Parameters
    dt = 0.01
    x_target = jnp.array([jnp.pi, 0.0])
    Q = jnp.diag(jnp.array([10.0, 1.0]))
    R = jnp.array([[0.1]])
    Q_f = jnp.diag(jnp.array([100.0, 10.0]))
    x_0 = jnp.array([jnp.pi - 0.1, 0.1])
    u_0 = jnp.array([0.5])

    print("--- Testing 'rk4' (via System Base) ---")
    sys_rk4 = MyPendulum(dt=dt, x_target=x_target, Q=Q, R=R, Q_f=Q_f, integrator="rk4")
    print(f"Next state (RK4): {sys_rk4.f_fcn(x_0, u_0)}")

    print("\n--- Testing 'contact_euler' (n_c=0) ---")
    sys_contact = MyPendulum(
        dt=dt, x_target=x_target, Q=Q, R=R, Q_f=Q_f, integrator="contact_euler"
    )
    print(f"Next state (Contact Euler): {sys_contact.f_fcn(x_0, u_0)}")

    print("Calculating derivatives...")
    f_x = sys_contact.f_x_fcn(x_0, u_0)
    print(f"f_x:\n{f_x}")
