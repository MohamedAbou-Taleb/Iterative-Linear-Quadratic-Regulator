import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time

# Import your custom classes from the other files
from class_files.systems.system_base import System
from class_files.systems.UA_double_pendulum_sys import MyUADoublePendulum
from class_files.iLQR_class import iLQR
from class_files.animations.animation_double_pendulum import AnimationDoublePendulum

def main():
    # =========================================================================
    # --- 1. System Parameters ---
    # =========================================================================
    print("Setting up MPC parameters for double pendulum...")
    dt = 0.01
    # --- MPC Horizon Settings ---
    # T_horizon = 1 # Time horizon for each MPC solve
    T_horizon = 2  # Time horizon for each MPC solve
    tspan_horizon = jnp.arange(0, T_horizon + dt, dt)
    N_horizon = len(tspan_horizon) - 1
    
    # --- Simulation Settings ---
    T_sim = 5.0 # Total simulation time
    tspan_sim = jnp.arange(0, T_sim + dt, dt)
    N_sim = len(tspan_sim) - 1
    
    # System dimensions
    n_x = 4  # [q1, q2, q1_dot, q2_dot]
    n_u = 1  # [tau1]
    
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
    # Q = jnp.diag(jnp.array([1.0, 2.0, 0.1, 0.1]))
    # R = jnp.diag(jnp.array([0.2]))
    # Q_f = jnp.diag(jnp.array([10.0, 10.0, 10.0, 10.0]))

    Q = jnp.diag(jnp.array([5.0, 5.0, 0.1, 0.1]))
    R = jnp.diag(jnp.array([50]))
    Q_f = jnp.diag(jnp.array([1000.0, 1000.0, 10.0, 10.0]))
    
    # Target: "up-up" position
    x_target = jnp.array([jnp.pi, 0.0, 0.0, 0.0])
    # Initial state: "down-down" position
    # x_0 = jnp.array([0.0, 0.0, -5.0, 1.0])
    x_0 = jnp.array([0.0, 0.0, 0.0, 0.0])
    
    # Initial control guess for the *first* solve
    U_init = jnp.zeros((n_u, N_horizon))
    
    # Solver settings
    tol = 1e-5
    maxiter = 50 # Low maxiter for MPC speed
    
    # =========================================================================
    # --- 2. Initialize System and Solver ---
    # =========================================================================
    
    # System for the iLQR solver (optimizer)
    # Use a simpler integrator for speed inside the MPC loop
    pendulum_sys = MyUADoublePendulum(
        dt=dt,
        x_target=x_target,
        Q=Q, R=R, Q_f=Q_f,
        g=g, m1=m1, m2=m2, l1=l1, l2=l2, d1=d1, d2=d2,
        theta1=theta1, theta2=theta2,
        integrator='rk4', # <-- Use Euler for MPC speed
        use_jit=True
    )
    
    # "Real-world" simulation system
    # Use a high-fidelity integrator for the "real" plant
    pendulum_sys_sim = MyUADoublePendulum(
        dt=dt,
        x_target=x_target,
        Q=Q, R=R, Q_f=Q_f,
        g=g, m1=m1, m2=m2, l1=l1, l2=l2, d1=d1, d2=d2,
        theta1=theta1, theta2=theta2,
        integrator='backward_euler', # <-- Use RK4 for simulation accuracy
        use_jit=True
    )
    
    # Instantiate the iLQR solver ONCE.
    ilqr_solver = iLQR(
        system=pendulum_sys,
        T=T_horizon,
        x_0=x_0,
        U_init=U_init,
        tol=tol,
        maxiter=maxiter,
        verbose=True # <-- Set to False for silent operation
    )

    # =========================================================================
    # --- 3. JIT Warm-up ---
    # =========================================================================
    print("Warming up JIT-compiled solver...")
    
    # 1. Warm up the backward pass
    X_warmup = jnp.zeros((n_x, N_horizon + 1))
    U_warmup = jnp.zeros((n_u, N_horizon))
    ilqr_solver.backward_pass(X_warmup, U_warmup)[0].block_until_ready()
    
    # 2. Warm up the forward pass
    U_ff_warmup = jnp.zeros((n_u, N_horizon))
    K_warmup = jnp.zeros((N_horizon, n_u, n_x))
    
    ilqr_solver.forward_pass(
        ilqr_solver.x_0, 0.0, X_warmup, U_warmup, U_ff_warmup, K_warmup
    )[0].block_until_ready()

    print("Warm-up complete.")

    # =========================================================================
    # --- 4. MPC Simulation Loop ---
    # =========================================================================
    print("Running MPC simulation for double pendulum...")
    
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
        # uk = jnp.array([0.0, 0.0])  # No control for testing
        
        # 5. Simulate the "real" system one step forward
        xkPlusOne = pendulum_sys_sim.f_fcn(current_x, uk)
        
        # 6. Store results
        U_sim = U_sim.at[:, k].set(uk)
        X_sim = X_sim.at[:, k+1].set(xkPlusOne)
        
        # 7. Prepare warm start for next iteration (shift U_bar)
        U_guess = jnp.concatenate([U_bar[:, 1:], U_bar[:, -1:]], axis=1)
        
        # 8. Update the current state
        current_x = xkPlusOne
        
        if (k+1) % 100 == 0:
            print(f"MPC Step {k+1}/{N_sim}...")

    elapsed_time_mpc = time.time() - start_time_mpc
    print(f"MPC simulation finished.")
    print(f"Total MPC time: {elapsed_time_mpc:.4f} seconds")
    print(f"Average time per step: {elapsed_time_mpc / N_sim:.5f} seconds")


    # =========================================================================
    # --- 5. Plotting ---
    # =========================================================================
    print("Plotting results...")
    
    fig, axs = plt.subplots(3, 2, figsize=(12, 10), facecolor='w')
    fig.suptitle('MPC Closed-Loop: Double Pendulum', fontsize=16)

    # Plot q1
    axs[0, 0].plot(tspan_sim, X_sim[0, :], 'b-', linewidth=2, label='q1')
    axs[0, 0].axhline(x_target[0], color='r', linestyle='--', label='q1_target')
    axs[0, 0].set_ylabel('q1 (rad)')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot q1_dot
    axs[0, 1].plot(tspan_sim, X_sim[2, :], 'g-', linewidth=2, label='q1_dot')
    axs[0, 1].axhline(x_target[2], color='r', linestyle='--', label='q1_dot_target')
    axs[0, 1].set_ylabel('q1_dot (rad/s)')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot q2
    axs[1, 0].plot(tspan_sim, X_sim[1, :], 'b-', linewidth=2, label='q2')
    axs[1, 0].axhline(x_target[1], color='r', linestyle='--', label='q2_target')
    axs[1, 0].set_ylabel('q2 (rad)')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Plot q2_dot
    axs[1, 1].plot(tspan_sim, X_sim[3, :], 'g-', linewidth=2, label='q2_dot')
    axs[1, 1].axhline(x_target[3], color='r', linestyle='--', label='q2_dot_target')
    axs[1, 1].set_ylabel('q2_dot (rad/s)')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # Plot tau1
    axs[2, 0].plot(tspan_sim[:-1], U_sim[0, :], 'k-', linewidth=2, label='tau1')
    axs[2, 0].set_xlabel('Time (s)')
    axs[2, 0].set_ylabel('Torque 1 (Nm)')
    axs[2, 0].legend()
    axs[2, 0].grid(True)

    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

    anim = AnimationDoublePendulum(pendulum_sys_sim, X_sim, tspan_sim, dt)
    anim.animate(save_video=False, 
                 filename="double_pendulum_swing_up.mp4", 
                 fullscreen=True)

if __name__ == "__main__":
    main()