import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time

# Import your custom classes from the other files
# (Assuming they are in a package/folder named 'class_files')
from class_files.systems.system_base import System
from class_files.systems.pendulum_sys import MyPendulum
from class_files.iLQR_class import iLQR

def main():
    # =========================================================================
    # --- 1. System Parameters ---
    # =========================================================================
    print("Setting up MPC parameters...")
    dt = 0.01
    
    # --- MPC Horizon Settings ---
    T_horizon = 2.0  # Time horizon for each MPC solve
    tspan_horizon = jnp.arange(0, T_horizon + dt, dt)
    N_horizon = len(tspan_horizon) - 1
    
    # --- Simulation Settings ---
    T_sim = 4.0 # Total simulation time
    tspan_sim = jnp.arange(0, T_sim + dt, dt)
    N_sim = len(tspan_sim) - 1
    
    # System dimensions
    n_x = 2  # n_x
    n_u = 1  # n_u
    
    # System dynamics parameters
    g = 9.81
    l = 1.0
    d = 0.0 # Damping
    
    # Cost parameters (more aggressive for MPC)
    Q = jnp.diag(jnp.array([10.0, 1.0]))
    R = jnp.diag(jnp.array([1.0]))
    Q_f = jnp.diag(jnp.array([10.0, 10.0])) # More weight on terminal cost
    
    x_target = jnp.array([jnp.pi, 0.0])
    x_0 = jnp.array([0.0, 0.0]) # Start from 0
    
    # Initial control guess for the *first* solve
    U_init = jnp.zeros((n_u, N_horizon))
    
    # Solver settings
    tol = 1e-6
    maxiter = 10 # Low maxiter for MPC
    
    # =========================================================================
    # --- 2. Initialize System and Solver ---
    # =========================================================================
    
    # System for the iLQR solver (optimizer)
    pendulum_sys = MyPendulum(
        dt=dt,
        x_target=x_target,
        Q=Q, R=R, Q_f=Q_f,
        g=g, l=l, d=d,
        integrator='midpoint', # Use Euler for the optimizer
        use_jit=True
    )
    
    # "Real-world" simulation system (can use a better integrator)
    pendulum_sys_sim = MyPendulum(
        dt=dt,
        x_target=x_target,
        Q=Q, R=R, Q_f=Q_f,
        g=g, l=l, d=d,
        integrator='midpoint', # Use RK4 for the "real" plant
        use_jit=True
    )
    
    # Instantiate the iLQR solver ONCE.
    # We will just update its x_0 and U_guess in the loop.
    ilqr_solver = iLQR(
        system=pendulum_sys,
        T=T_horizon,
        x_0=x_0,
        U_init=U_init,
        tol=tol,
        maxiter=maxiter,
        verbose=False # <-- Set to False for silent operation in MPC loop
    )

    # =========================================================================
    # --- 3. JIT Warm-up (Optional but good) ---
    # =========================================================================
    print("Warming up JIT-compiled solver...")
    # Run one optimization to compile everything
    # We need to pass the x_0 argument
    ilqr_solver.optimize_trajectory()[0].block_until_ready()
    print("Warm-up complete.")

    # =========================================================================
    # --- 4. MPC Simulation Loop ---
    # =========================================================================
    print("Running MPC simulation...")
    
    # Storage for simulation results
    X_sim = jnp.zeros((n_x, N_sim + 1))
    U_sim = jnp.zeros((n_u, N_sim))
    
    # Initialize simulation
    current_x = x_0
    X_sim = X_sim.at[:, 0].set(current_x)
    
    # U_guess will be the "warm start" for the next iteration
    U_guess = U_init
    
    start_time_mpc = time.time()
    
    for k in range(N_sim):
        # 1. Update the solver's initial state
        ilqr_solver.x_0 = current_x
        
        # 2. Provide the warm-start control guess
        ilqr_solver.U = U_guess
        
        # 3. Solve the optimization problem
        X_bar, U_bar, cost = ilqr_solver.optimize_trajectory()
        
        # 4. Get the first control input
        uk = U_bar[:, 0]
        
        # 5. Simulate the "real" system one step forward
        xkPlusOne = pendulum_sys_sim.f_fcn(current_x, uk)
        
        # 6. Store results
        U_sim = U_sim.at[:, k].set(uk)
        X_sim = X_sim.at[:, k+1].set(xkPlusOne)
        
        # 7. Prepare warm start for next iteration (shift U_bar)
        U_guess = jnp.concatenate([U_bar[:, 1:], U_bar[:, -1:]], axis=1)
        
        # 8. Update the current state
        current_x = xkPlusOne
        
        if k % 100 == 0:
            print(f"MPC Step {k}/{N_sim}...")

    elapsed_time_mpc = time.time() - start_time_mpc
    print(f"MPC simulation finished.")
    print(f"Total MPC time: {elapsed_time_mpc:.4f} seconds")
    print(f"Average time per step: {elapsed_time_mpc / N_sim:.5f} seconds")

    # =========================================================================
    # --- 5. Plotting ---
    # =========================================================================
    print("Plotting results...")
    
    fig = plt.figure(figsize=(10, 8), facecolor='w')
    
    # Plot State 1 (theta)
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(tspan_sim, X_sim[0, :], 'b-', linewidth=2, label='Closed Loop')
    ax1.axhline(x_target[0], color='r', linestyle='--', linewidth=2, label='Target')
    ax1.set_title('Closed-Loop State Trajectories')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Theta (rad)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot State 2 (theta_dot)
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(tspan_sim, X_sim[1, :], 'b-', linewidth=2)
    ax2.axhline(x_target[1], color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Theta_dot (rad/s)')
    ax2.grid(True)
    
    # Plot Control (tau)
    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(tspan_sim[:-1], U_sim[0, :], 'k-', linewidth=2, label='Control Input')
    ax3.set_title('Optimal Control Input')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Control (torque)')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()