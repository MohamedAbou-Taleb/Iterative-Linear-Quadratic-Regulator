import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time

# Import your custom classes from the other files
from class_files.systems.system_base import System  # Base class
from class_files.systems.pendulum_sys import MyPendulum # Your pendulum implementation
from class_files.iLQR_class import iLQR          # Your iLQR solver

def main():
    # =========================================================================
    # --- 1. System Parameters ---
    # =========================================================================
    print("Setting up parameters...")
    dt = 0.01
    T = 4.0
    tspan = jnp.arange(0, T + dt, dt)
    N = len(tspan) - 1
    
    # System dimensions
    n = 2  # n_x
    m = 1  # n_u
    
    # System dynamics parameters
    g = 9.81
    l = 1.0
    d = 0.0 # Damping (set to 0 to match MATLAB)
    
    # Cost parameters
    Q = jnp.diag(jnp.array([1.0, 1.0]))
    R = jnp.diag(jnp.array([1.0])) # Equivalent to eye(m)
    Q_f = jnp.diag(jnp.array([0.0, 0.0])) # Equivalent to eye(n)*0
    
    x_target = jnp.array([jnp.pi, 0.0])
    x_0 = jnp.array([1.0, 0.0])
    
    # Initial control guess
    U_init = jnp.zeros((m, N))
    
    # Solver settings
    tol = 1e-5
    maxiter = 100 # Reduced from 1000 for faster demo
    
    # =========================================================================
    # --- 2. Run iLQR Solver ---
    # =========================================================================
    print("Running iLQR...")
    
    # Instantiate the pendulum system
    # NOTE: We use 'euler' to match your original MATLAB f_fcn
    pendulum_sys = MyPendulum(
        dt=dt,
        x_target=x_target,
        Q=Q, R=R, Q_f=Q_f,
        g=g, l=l, d=d,
        integrator='euler', # Match MATLAB's Euler integration
        use_jit=True
    )
    
    # Instantiate the iLQR solver
    ilqr_solver = iLQR(
        system=pendulum_sys,
        T=T,
        x_0=x_0,
        U_init=U_init,
        tol=tol,
        maxiter=maxiter
    )

    # =========================================================================
    # --- 2. JIT Warm-up ---
    # =========================================================================
    print("Warming up JIT-compiled functions...")
    
    # 1. Warm up the backward pass. This will compile the scan body
    #    and all its dependencies (all system derivatives).
    X_warmup = jnp.zeros_like(ilqr_solver.X)
    U_warmup = jnp.zeros_like(ilqr_solver.U)
    
    # We need to block until compilation is done.
    # We just grab the first output (U_ff) and block on it.
    ilqr_solver.backward_pass(X_warmup, U_warmup)[0].block_until_ready()
    
    # 2. Warm up the forward pass.
    U_ff_warmup = jnp.zeros_like(ilqr_solver.U_ff)
    K_warmup = jnp.zeros_like(ilqr_solver.K)
    
    # Grab the first output (X_new) and block on it.
    # Pass the x_0 argument
    ilqr_solver.forward_pass(
        ilqr_solver.x_0, 0.0, X_warmup, U_warmup, U_ff_warmup, K_warmup
    )[0].block_until_ready()

    print("Warm-up complete.")

    # =========================================================================
    # --- 3. Run iLQR Solver (Timed) ---
    # =========================================================================
    print("Running iLQR...")

    # Solve
    # Now this timer only measures execution, not compilation.
    start_time_ilqr = time.time()
    X_bar, U_bar, cost_ilqr = ilqr_solver.optimize_trajectory()
    elapsed_time_ilqr = time.time() - start_time_ilqr
    
    print(f"Time taken to execute iLQR: {elapsed_time_ilqr:.4f} seconds")

    # =========================================================================
    # --- 4. Plotting ---
    # =========================================================================
    print("Plotting results...")
    
    fig = plt.figure(figsize=(10, 8), facecolor='w')
    
    # Plot State 1 (theta)
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(tspan, X_bar[0, :], 'b-', linewidth=2, label='iLQR')
    # ax1.plot(tspan, X_bar_casadi_val[0, :], 'r--', linewidth=2, label='Collocation (CasADi)')
    ax1.set_title('Optimal State Trajectories')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Theta (rad)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot State 2 (theta_dot)
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(tspan, X_bar[1, :], 'b-', linewidth=2, label='iLQR')
    # ax2.plot(tspan, X_bar_casadi_val[1, :], 'r--', linewidth=2, label='Collocation (CasADi)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Theta_dot (rad/s)')
    ax2.grid(True)
    
    # Plot Control (tau)
    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(tspan[:-1], U_bar[0, :], 'b-', linewidth=2, label='iLQR')
    # ax3.plot(tspan[:-1], U_bar_casadi_val[0, :], 'r--', linewidth=2, label='Collocation (CasADi)')
    ax3.set_title('Optimal Control Input')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Control (torque)')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print final times
    print(f"\n--- Summary ---")
    print(f"iLQR solver time:      {elapsed_time_ilqr:.4f} seconds")

if __name__ == "__main__":
    main()