import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time

# Import your custom classes from the other files
from class_files.systems.system_base import System
from class_files.systems.UA_double_pendulum_sys import MyUADoublePendulum # <-- Import MyDoublePendulum
from class_files.iLQR_class import iLQR
from class_files.animations.animation_double_pendulum import AnimationDoublePendulum

def main():
    # =========================================================================
    # --- 1. System Parameters (Double Pendulum) ---
    # =========================================================================
    print("Setting up double pendulum parameters...")
    dt = 0.01
    T = 8.0  # Longer horizon for the harder problem
    tspan = jnp.arange(0, T + dt, dt)
    N = len(tspan) - 1
    
    # System dimensions
    n_x = 4  # [q1, q2, q1_dot, q2_dot]
    n_u = 1  # [tau]
    
    # System dynamics parameters (using defaults from your class)
    g = 9.81
    m1 = 1.0
    m2 = 1.0
    l1 = 1.0
    l2 = 1.0
    d1 = 0.1 # Small damping
    d2 = 0.1 # Small damping
    
    # --- Calculate Moments of Inertia ---
    # Assuming slender rods/cuboids: I_cm = (1/12) * m * L^2
    theta1 = (1/12) * m1 * l1**2
    theta2 = (1/12) * m2 * l2**2
    
    # Cost parameters
    # Penalize position error and control effort
    Q = jnp.diag(jnp.array([1.0, 1.0, 0.1, 0.1]))
    R = jnp.diag(jnp.array([1.0]))
    Q_f = jnp.diag(jnp.array([1000.0, 1000.0, 100.0, 100.0])) # High terminal cost
    
    # Target: "up-up" position
    x_target = jnp.array([jnp.pi, 0.0, 0.0, 0.0])
    # Initial state: "down-down" position
    x_0 = jnp.array([0.0, 0.0, 0.0, 0.0])
    
    # Initial control guess (zero)
    U_init = jnp.zeros((n_u, N))
    
    # Solver settings
    tol = 1e-5
    maxiter = 700 # More iterations for the harder problem
    
    # =========================================================================
    # --- 2. Instantiate System and Solver ---
    # =========================================================================
    print("Instantiating double pendulum system...")
    
    double_pendulum_sys = MyUADoublePendulum(
        dt=dt,
        x_target=x_target,
        Q=Q, R=R, Q_f=Q_f,
        g=g, m1=m1, m2=m2, l1=l1, l2=l2, d1=d1, d2=d2,
        theta1=theta1, theta2=theta2, # <-- Pass the calculated MOI
        integrator='backward_euler', # Use RK4 for better accuracy
        use_jit=True
    )
    
    ilqr_solver = iLQR(
        system=double_pendulum_sys,
        T=T,
        x_0=x_0,
        U_init=U_init,
        tol=tol,
        maxiter=maxiter,
        verbose=True
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
    
    fig, axs = plt.subplots(3, 2, figsize=(12, 10), facecolor='w')
    fig.suptitle('iLQR Double Pendulum Swing-up', fontsize=16)

    # Plot q1
    axs[0, 0].plot(tspan, X_bar[0, :], 'b-', linewidth=2, label='q1')
    axs[0, 0].axhline(x_target[0], color='r', linestyle='--', label='q1_target')
    axs[0, 0].set_ylabel('q1 (rad)')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot q1_dot
    axs[0, 1].plot(tspan, X_bar[2, :], 'g-', linewidth=2, label='q1_dot')
    axs[0, 1].axhline(x_target[2], color='r', linestyle='--', label='q1_dot_target')
    axs[0, 1].set_ylabel('q1_dot (rad/s)')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot q2
    axs[1, 0].plot(tspan, X_bar[1, :], 'b-', linewidth=2, label='q2')
    axs[1, 0].axhline(x_target[1], color='r', linestyle='--', label='q2_target')
    axs[1, 0].set_ylabel('q2 (rad)')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Plot q2_dot
    axs[1, 1].plot(tspan, X_bar[3, :], 'g-', linewidth=2, label='q2_dot')
    axs[1, 1].axhline(x_target[3], color='r', linestyle='--', label='q2_dot_target')
    axs[1, 1].set_ylabel('q2_dot (rad/s)')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # Plot tau1
    axs[2, 0].plot(tspan[:-1], U_bar[0, :], 'k-', linewidth=2, label='tau1')
    axs[2, 0].set_xlabel('Time (s)')
    axs[2, 0].set_ylabel('Torque 1 (Nm)')
    axs[2, 0].legend()
    axs[2, 0].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

    anim = AnimationDoublePendulum(double_pendulum_sys, X_bar, tspan, dt)
    anim.animate(fullscreen=True, save_video=False, filename="double_pendulum_swing_up.mp4")

if __name__ == "__main__":
    main()