import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time

# Import your custom classes from the other files
from class_files.systems.system_base import System
from class_files.systems.point_mass_box_7DoF_sys import (
    MyPointMassBoxManipulator7DoF,
)  # <-- Import MyDoublePendulum
from class_files.iLQR_class import iLQR
from class_files.animations.animation_point_mass_box_7DoF import (
    AnimationPointMassBox7DoF,
)


def main():
    print("Setting up double pendulum parameters...")
    dt = 0.01
    T = 6.0  # Longer horizon for the harder problem
    tspan = jnp.arange(0, T + dt, dt)
    N = len(tspan) - 1
    box_width = 0.5
    box_height = 0.3
    ball_radius = 0.05
    x_box_target = jnp.array([0.0, 3*box_height / 2, 0.0,
                              0.0, 0.0, 0.0])

    R = jnp.diag(jnp.array([1.0, 1.0, 1.0, 1.0]))*1e-1
    Q_box = jnp.diag(jnp.array([100.0, 100.0, 1.0,
                                0.0, 0.0, 0.0]))
    Q_f = jnp.diag(jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))*100
    Q_box_ball = 100.0
    RN1 = 50
    RN2 = 50
    RN1_f = 0.0
    RN2_f = 0.0
    m_box = 0.5
    m_ball = 1

    # --- Initial State ---
    # q = [x_b1, y_b1, x_b2, y_b2, x_box, y_box]
    q_box_x_0 = +0.0
    q_0 = jnp.array(
        [
            -(box_width / 2 + ball_radius) + q_box_x_0 - 0.0,
            0.1,
            box_width / 2 + ball_radius + q_box_x_0 + 0.0,
            0.1,
            q_box_x_0,
            box_height / 2,
            0.0
        ]
    )  # Box starts high (0.5)
    v_0 = jnp.zeros(
        7,
    )
    x_0 = jnp.hstack([q_0, v_0])
    n_x = 7
    n_u = 4

    key = jax.random.key(1)
    U_init = (2*jax.random.uniform(key, shape=(n_u, N)) - 1) * 1
    # U_init = jnp.zeros((n_u, N))
    # U_init= jnp.vstack([10*jnp.ones((1, N)), jnp.zeros((3, N))])

    print(f"Initial State: {x_0}")
    # Solver settings
    tol = 1e-5
    maxiter = 700  # More iterations for the harder problem
    reg_friction = jnp.array([1e-2, 1e-2, 1e-2, 1e-2]) * 1e-4
    # mu = jnp.array([0.5, 0.5, 0.01])
    mu = jnp.array([0.3, 0.3, 0.0, 0.0])
    # --- Instantiate System ---
    manipulator = MyPointMassBoxManipulator7DoF(
        dt=dt,
        box_target_state=x_box_target,
        R=R,
        Q_box=Q_box,
        RN1=RN1,
        RN2=RN2,
        Q_f=Q_f,
        Q_box_ball=Q_box_ball,
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
    )  # mu=0.0 for box-floor to slide

    ilqr_solver = iLQR(
        system=manipulator,
        T=T,
        x_0=x_0,
        U_init=U_init,
        tol=tol,
        maxiter=maxiter,
        verbose=True,
    )

    # =========================================================================
    # --- 3. JIT Warm-up ---
    # =========================================================================
    print("Warming up JIT-compiled functions...")

    # 1. Warm up the backward pass
    X_warmup = jnp.zeros_like(ilqr_solver.X)
    U_warmup = jnp.zeros_like(ilqr_solver.U)
    ilqr_solver.backward_pass(X_warmup, U_warmup)[0].block_until_ready()

    # 2. Warm up the forward pass
    U_ff_warmup = jnp.zeros_like(ilqr_solver.U_ff)
    K_warmup = jnp.zeros_like(ilqr_solver.K)

    # Pass the initial state x_0 as an argument
    ilqr_solver.forward_pass(
        ilqr_solver.x_0, 0.0, X_warmup, U_warmup, U_ff_warmup, K_warmup
    )[0].block_until_ready()

    print("Warm-up complete.")

    # =========================================================================
    # --- 4. Run iLQR Solver (Timed) ---
    # =========================================================================
    print("Running iLQR solve for double pendulum swing-up...")

    start_time_ilqr = time.time()
    X_bar, U_bar, cost_ilqr = ilqr_solver.optimize_trajectory()
    elapsed_time_ilqr = time.time() - start_time_ilqr

    print(f"Time taken to execute iLQR: {elapsed_time_ilqr:.4f} seconds")

    # =========================================================================
    # --- 5. Plotting ---
    # =========================================================================
    print("Plotting results...")

    # Convert JAX arrays to Numpy for plotting
    X_plot = X_bar.T
    t_plot = tspan

    # Ensure lengths match (truncate X if it has one more step than t, or vice versa)
    if len(X_plot) > len(t_plot):
        X_plot = X_plot[: len(t_plot)]
    elif len(t_plot) > len(X_plot):
        t_plot = t_plot[: len(X_plot)]

    fig, axes = plt.subplots(3, 2, figsize=(10, 5), sharex=True)
    fig.suptitle(f"Positions over Time (T={T}s)", fontsize=16)

    # --- Ball 1 (Left) ---
    axes[0, 0].plot(t_plot, X_plot[:, 0], "b-", linewidth=2, label=r"$x_{b1}$")
    axes[0, 0].set_ylabel("Position [m]")
    axes[0, 0].set_title("Ball 1 X")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    axes[0, 1].plot(t_plot, X_plot[:, 1], "b--", linewidth=2, label=r"$y_{b1}$")
    axes[0, 1].set_ylabel("Position [m]")
    axes[0, 1].set_title("Ball 1 Y")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    # --- Ball 2 (Right) ---
    axes[1, 0].plot(t_plot, X_plot[:, 2], "r-", linewidth=2, label=r"$x_{b2}$")
    axes[1, 0].set_ylabel("Position [m]")
    axes[1, 0].set_title("Ball 2 X")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    axes[1, 1].plot(t_plot, X_plot[:, 3], "r--", linewidth=2, label=r"$y_{b2}$")
    axes[1, 1].set_ylabel("Position [m]")
    axes[1, 1].set_title("Ball 2 Y")
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    # --- Box ---
    # Target X
    axes[2, 0].axhline(
        y=x_box_target[0], color="k", linestyle=":", linewidth=2, label="Target"
    )
    axes[2, 0].plot(t_plot, X_plot[:, 4], "g-", linewidth=2, label=r"$x_{box}$")
    axes[2, 0].set_ylabel("Position [m]")
    axes[2, 0].set_xlabel("Time [s]")
    axes[2, 0].set_title("Box X")
    axes[2, 0].grid(True)
    axes[2, 0].legend()

    # Target Y
    axes[2, 1].axhline(
        y=x_box_target[1], color="k", linestyle=":", linewidth=2, label="Target"
    )
    axes[2, 1].plot(t_plot, X_plot[:, 5], "g--", linewidth=2, label=r"$y_{box}$")
    axes[2, 1].set_ylabel("Position [m]")
    axes[2, 1].set_xlabel("Time [s]")
    axes[2, 1].set_title("Box Y")
    axes[2, 1].grid(True)
    axes[2, 1].legend()

    # =========================================================================
    # --- 6. Plotting Controls ---
    # =========================================================================
    print("Plotting control inputs...")
    U_plot = U_bar.T

    # Create time array for controls (length N)
    t_u = t_plot[: U_plot.shape[0]]

    fig_u, axes_u = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig_u.suptitle(f"Control Inputs (Forces) over Time", fontsize=16)

    # Ball 1 Controls (u0, u1)
    axes_u[0, 0].plot(t_u, U_plot[:, 0], "b-", linewidth=2)
    axes_u[0, 0].set_title("u0: Ball 1 Force X")
    axes_u[0, 0].set_ylabel("Force [N]")
    axes_u[0, 0].grid(True)

    axes_u[0, 1].plot(t_u, U_plot[:, 1], "b--", linewidth=2)
    axes_u[0, 1].set_title("u1: Ball 1 Force Y")
    axes_u[0, 1].set_ylabel("Force [N]")
    axes_u[0, 1].grid(True)

    # Ball 2 Controls (u2, u3)
    axes_u[1, 0].plot(t_u, U_plot[:, 2], "r-", linewidth=2)
    axes_u[1, 0].set_title("u2: Ball 2 Force X")
    axes_u[1, 0].set_ylabel("Force [N]")
    axes_u[1, 0].grid(True)

    axes_u[1, 1].plot(t_u, U_plot[:, 3], "r--", linewidth=2)
    axes_u[1, 1].set_title("u3: Ball 2 Force Y")
    axes_u[1, 1].set_ylabel("Force [N]")
    axes_u[1, 1].grid(True)

    axes_u[1, 0].set_xlabel("Time [s]")
    axes_u[1, 1].set_xlabel("Time [s]")

    plt.tight_layout()
    plt.show()

    anim = AnimationPointMassBox7DoF(manipulator, X_bar.T, tspan, dt)
    anim.animate(fullscreen=True, save_video=False, filename="box_manipulation.mp4")


if __name__ == "__main__":
    main()
