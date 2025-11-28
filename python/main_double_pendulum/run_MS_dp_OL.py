import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time

# Import your custom classes from the other files
from class_files.systems.system_base import System
from class_files.systems.double_pendulum_sys import MyDoublePendulum
# --- CHANGED: Import the Multiple Shooting solver ---
from class_files.MS_iLQR_class import MultipleShootingiLQR
from class_files.animations.animation_double_pendulum import AnimationDoublePendulum

def main():
    # =========================================================================
    # --- 1. System Parameters (Double Pendulum) ---
    # =========================================================================
    print("Setting up double pendulum parameters...")
    dt = 0.01
    T = 4  # Longer horizon for the harder problem
    tspan = jnp.arange(0, T + dt, dt)
    N = len(tspan) - 1
    
    # System dimensions
    n_x = 4  # [q1, q2, q1_dot, q2_dot]
    n_u = 2  # [tau1, tau2]
    
    # System dynamics parameters
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
    R = jnp.diag(jnp.array([0.1, 1.0]))
    Q_f = jnp.diag(jnp.array([1000.0, 1000.0, 100.0, 100.0])) # High terminal cost
    
    # Target: "up-up" position
    x_target = jnp.array([-jnp.pi, 0.0, 0.0, 0.0])
    # Initial state: "down-down" position
    x_0 = jnp.array([-jnp.pi/4, 0.0, 0.0, 0.0])
    
    # Initial control guess (zero)
    U_init = jnp.zeros((n_u, N))

    # --- OPTIONAL: Initial State Guess (Warm Start) ---
    # Multiple Shooting allows you to pass a state guess X_init.
    # If None, it performs a rollout of U_init to generate the initial trajectory.

    def get_smooth_guess(x_0, x_target, N):
        """
        Uses cosine interpolation for a smoother 'ease-in, ease-out' trajectory.
        """
        # Create time steps from 0 to 1
        t = jnp.linspace(0.0, 1.0, N + 1)
        
        # Cosine interpolation formula: 0.5 * (1 - cos(pi * t))
        # This creates an S-curve from 0 to 1
        alpha = 0.5 * (1.0 - jnp.cos(jnp.pi * t))
        
        X_init = x_0[:, None] + (x_target - x_0)[:, None] * alpha[None, :]
        
        return X_init
    X_init = None 
    X_init = get_smooth_guess(x_0, x_target, N)
    
    # Solver settings
    tol = 1e-5
    maxiter = 200 
    
    # =========================================================================
    # --- 2. Instantiate System and Solver ---
    # =========================================================================
    print("Instantiating double pendulum system...")
    
    double_pendulum_sys = MyDoublePendulum(
        dt=dt,
        x_target=x_target,
        Q=Q, R=R, Q_f=Q_f,
        g=g, m1=m1, m2=m2, l1=l1, l2=l2, d1=d1, d2=d2,
        theta1=theta1, theta2=theta2,
        d_wall = 2,
        e_restitution=jnp.array([1.0, 1.0]),
        integrator='elastic_contact_euler', 
        use_jit=True
    )
    
    print("Instantiating Multiple Shooting iLQR solver...")
    # --- CHANGED: Use MultipleShootingiLQR class ---
    ms_solver = MultipleShootingiLQR(
        system=double_pendulum_sys,
        T=T,
        x_0=x_0,
        U_init=U_init,
        X_init=X_init, # Pass the optional state guess
        tol=tol,
        maxiter=maxiter,
        verbose=True
    )

    # =========================================================================
    # --- 3. JIT Warm-up ---
    # =========================================================================
    print("Warming up JIT-compiled functions...")
    
    # 1. Warm up the backward pass
    # (Note: MS-iLQR backward pass logic is different but signature is the same)
    X_warmup = jnp.zeros_like(ms_solver.X)
    U_warmup = jnp.zeros_like(ms_solver.U)
    ms_solver.backward_pass(X_warmup, U_warmup)[0].block_until_ready()
    
    # 2. Warm up the forward pass
    U_ff_warmup = jnp.zeros_like(ms_solver.U_ff)
    K_warmup = jnp.zeros_like(ms_solver.K)
    
    # Pass the initial state x_0 as an argument
    ms_solver.forward_pass(
        ms_solver.x_0, 0.0, X_warmup, U_warmup, U_ff_warmup, K_warmup
    )[0].block_until_ready()

    print("Warm-up complete.")

    # =========================================================================
    # --- 4. Run MS-iLQR Solver (Timed) ---
    # =========================================================================
    print("Running MS-iLQR solve for double pendulum swing-up...")

    start_time_ilqr = time.time()
    # The interface is identical: optimize_trajectory returns (X, U, cost)
    X_bar, U_bar, cost_ilqr = ms_solver.optimize_trajectory()
    elapsed_time_ilqr = time.time() - start_time_ilqr
    
    print(f"Time taken to execute MS-iLQR: {elapsed_time_ilqr:.4f} seconds")

    # =========================================================================
    # --- 5. Plotting ---
    # =========================================================================
    print("Plotting results...")
    
    fig, axs = plt.subplots(3, 2, figsize=(12, 10), facecolor='w')
    fig.suptitle('Multiple Shooting iLQR Double Pendulum Swing-up', fontsize=16)

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

    # Plot tau2
    axs[2, 1].plot(tspan[:-1], U_bar[1, :], 'k-', linewidth=2, label='tau2')
    axs[2, 1].set_xlabel('Time (s)')
    axs[2, 1].set_ylabel('Torque 2 (Nm)')
    axs[2, 1].legend()
    axs[2, 1].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

    anim = AnimationDoublePendulum(double_pendulum_sys, X_bar, tspan, dt)
    anim.animate(fullscreen=True, save_video=False, filename="double_pendulum_ms_swing_up.mp4")

if __name__ == "__main__":
    main()