import jax
import jax.numpy as jnp
import numpy as np
from typing import Union
import time
from jax import jit, lax
import matplotlib.pyplot as plt
from class_files.animations.animation_double_pendulum import AnimationDoublePendulum
from rotations import Exp_SO3, Log_SO3, ax2skew

# FIX: Robust import to handle both script execution and module import
try:
    # Try relative import (for when imported as a module in run_iLQR...)
    from .system_base import System
except ImportError:
    # Fallback to absolute import (for when running this script directly)
    from system_base import System


class MyDiscreteRod(System):
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
        n_element: int = 2,
        # --- Physical parameters ---
        g: float = 9.81,
        m_rod: float = 1.0,
        theta_rod: float = 1.0,
        l_rod: float = 1.0,
        # --- System settings ---
        use_jit: bool = True,
        integrator: str = "rk4",
        mu: float = jnp.array([0.0, 0.0]),
        smooth_epsilon: float = 1.0,
        e_restitution=jnp.array([0.0, 0.0]),
        **kwargs,
    ):

        # 1. --- Define system properties ---
        self.n_element = n_element
        n_q = (n_element + 1) * 3  # [q1, q2]
        n_v = (n_element + 1) * 3  # [q1_dot, q2_dot]
        n_u = 1  # [tau1, tau2]
        n_c = 0

        # Store physical parameters
        self.g = g
        self.m_rod = m_rod
        self.theta_rod = theta_rod
        self.l_rod = l_rod

        # 2. --- Store cost parameters as JAX arrays ---
        self.x_target = jnp.asarray(x_target)
        self.Q = jnp.asarray(Q)
        self.R = jnp.asarray(R)
        self.Q_f = jnp.asarray(Q_f)

        # 3. --- Call the base class constructor ---
        super().__init__(
            n_q=n_q,
            n_v=n_v,
            n_u=n_u,
            n_c=n_c,
            dt=dt,
            integrator=integrator,
            mu=mu,
            smooth_epsilon=smooth_epsilon,
            e_restitution=e_restitution,
            **kwargs,
        )

    # --- Physics Implementation (M, h, W) ---

    def _mass_matrix(self, q: jnp.ndarray) -> jnp.ndarray:
        """Returns M(q) of shape (n_v, n_v)."""
        M1 = [self.m_rod, self.m_rod, self.theta_rod]
        M2 = [2 * self.m_rod, 2 * self.m_rod, 2 * self.theta_rod]
        M = jnp.diag(jnp.array(M1 + M2 * (self.n_element - 1) + M1))

        return M

    def _generalized_forces(
        self, q: jnp.ndarray, v: jnp.ndarray, u: jnp.ndarray
    ) -> jnp.ndarray:
        """Returns h(q, v, u) of shape (n_v,)."""
        # Map inputs to your variables
        # q1, q2 = q
        # v1, v2 = v
        # tau = u

        H = jnp.zeros((self.n_v, self.n_v))
        for i in range(self.n_element):
            h_i = jnp.zeros(
                self.n_v,
            )
            A_IB1 = Exp_SO3(jnp.array([0.0, 0.0, q[3 * i - 1]]))
            A_IB2 = Exp_SO3(jnp.array([0.0, 0.0, q[3 * i + 2]]))
            A_B1B2 = A_IB1.T @ A_IB2
            psi = Log_SO3(A_B1B2)
            A_IB = Exp_SO3(psi / 2)
            r_OC1 = jnp.array([*q[3 * i : 3 * i + 2], 0.0])
            r_OC2 = jnp.array([*q[3 * i + 3 : 3 * i + 5], 0.0])
            l_i = self.l_rod / self.n_element
            gamma = A_IB.T @ (r_OC2 - r_OC1) / l_i
            kappa = psi / l_i
            gamma_0 = jnp.array([1.2, 0.0, 0.0])
            n = jnp.diag(jnp.array([20.0, 1.0, 1.0])) @ (gamma - gamma_0)
            m = kappa
            W = jnp.block(
                [
                    [A_IB, jnp.zeros((3, 3))],
                    [
                        0.5 * ax2skew(gamma) * l_i,
                        jnp.eye(3) + 0.5 * ax2skew(kappa) * l_i,
                    ],
                    [-A_IB, jnp.zeros((3, 3))],
                    [
                        0.5 * ax2skew(gamma) * l_i,
                        -jnp.eye(3) + 0.5 * ax2skew(kappa) * l_i,
                    ],
                ]
            )
            W_lam = W @ jnp.array([*n, *m])
            planar_idx_rows = jnp.array([0, 1, 5, 6, 7, 11])
            W_lam = W_lam[planar_idx_rows]
            idx = jnp.array(
                [3 * i, 1 + 3 * i, 5 + 3 * i, 6 + 3 * i, 7 + 3 * i, 11 + 3 * i]
            )
            H = H.at[idx, i].set(W_lam)
        h_gyro = jnp.zeros(
            self.n_v,
        )
        h = jnp.sum(H, axis=1)

        h = h.at[0:3].set(0)
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

    print("--- Testing 'rk4' (Double Pendulum) ---")
    sys_dp = MyDiscreteRod(
        dt=dt,
        x_target=x_target,
        Q=Q,
        R=R,
        Q_f=Q_f,
        integrator="rk4",
        e_restitution=jnp.array([0.0, 0.0]),
        mu=jnp.array([0.3]),
    )

    q_x_0 = jnp.linspace(0, sys_dp.l_rod, sys_dp.n_element + 1)
    q_y_0 = jnp.zeros(sys_dp.n_element + 1)
    q_theta_0 = jnp.zeros(sys_dp.n_element + 1)
    q_0 = jnp.array([q_x_0, q_y_0, q_theta_0]).T.flatten()
    v_0 = jnp.zeros_like(q_0)
    x_0 = jnp.array([*q_0, *v_0])  # Start hanging down
    u_0 = jnp.array([0.0])  # No torque

    # print(sys_dp._generalized_forces(jnp.zeros(9,), jnp.zeros(9,), jnp.array([0.0])))

    x_next = sys_dp.f_fcn(x_0, u_0)
    print(f"Current state: {x_0}")
    print(f"Next state (RK4, zero input): {x_next}")

    # Test derivatives
    print("\nCalculating derivatives (f_x)...")
    f_x = sys_dp.f_x_fcn(x_0, u_0)
    print(f"f_x shape: {f_x.shape}")

    # Simulation
    print("\nSimulating free fall...")
    T_sim = 5.0
    tspan = jnp.arange(0, T_sim, dt)
    N_sim = len(tspan)

    X_hist = [x_0]
    curr_x = x_0
    for _ in range(N_sim):
        curr_x = sys_dp.f_fcn(curr_x, u_0)
        X_hist.append(curr_x)

    X_hist = np.array(X_hist)

    # plt.figure(figsize=(10, 6))
    # plt.plot(X_hist[0, 0::3], X_hist[0, 1::3], '-rx')
    # plt.plot(X_hist[N_sim//2, 0::3], X_hist[N_sim//2, 1::3], '-go')
    # plt.plot(X_hist[-1, 0::3], X_hist[-1, 1::3], '--b^')
    # plt.legend()
    # plt.title("Double Pendulum Free Fall")
    # plt.grid(True)
    # plt.show()

    import matplotlib.pyplot as plt
import numpy as np

# --- 1. Generate Mock Data (Replace this with your actual X_hist) ---
# Creating a dummy double pendulum motion for demonstration
t_vals = tspan

# --- 2. Setup the Animation ---

# Turn on interactive mode
plt.ion()

# Initialize the figure ONCE
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("Double Pendulum Free Fall")
ax.grid(True)

# Calculate fixed axis limits so the camera doesn't jump around
# We look at all time steps to find the min/max X and Y
all_x = X_hist[:, 0::3].flatten()
all_y = X_hist[:, 1::3].flatten()
ax.set_xlim(all_x.min() - 0.5, all_x.max() + 0.5)
ax.set_ylim(all_y.min() - 0.5, all_y.max() + 0.5)

# Initialize an empty line object.
# This returns a tuple, so we use the comma to unpack the first element.
(line,) = ax.plot([], [], "-rx", linewidth=2, markersize=8)

# --- 3. The Animation Loop ---
for t in range(len(X_hist)):
    # Extract data for the current time step 't'
    # 0::3 gets indices 0, 3, 6... (The X coordinates)
    # 1::3 gets indices 1, 4, 7... (The Y coordinates)
    current_x = X_hist[t, 0 : sys_dp.n_q : 3]
    current_y = X_hist[t, 1 : sys_dp.n_q : 3]
    print(X_hist.shape)

    # Update the data in the existing line object
    line.set_data(current_x, current_y)

    # Update title to show progress (optional)
    ax.set_title(f"Double Pendulum Free Fall (t = {t*dt})")

    # Pause briefly to allow the plot to render and the eye to register the frame
    plt.pause(0.05)

# Turn off interactive mode and keep the final window open
plt.ioff()
plt.show()

# anim = AnimationDoublePendulum(sys_dp, X_hist.T, tspan, dt)
# anim.animate(save_video=False,
#              filename="double_pendulum_swing_up.mp4",
#              fullscreen=True)
