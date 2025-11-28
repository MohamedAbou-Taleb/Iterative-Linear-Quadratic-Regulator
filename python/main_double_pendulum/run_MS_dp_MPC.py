import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time

# Import your custom classes
from class_files.systems.system_base import System
from class_files.systems.double_pendulum_sys import MyDoublePendulum
from class_files.MS_iLQR_class import MultipleShootingiLQR  # <-- IMPORT NEW CLASS
from class_files.animations.animation_double_pendulum import AnimationDoublePendulum

def get_linear_guess(x_start, x_goal, N):
    """Helper to create a linear initialization for X."""
    alpha = jnp.linspace(0.0, 1.0, N + 1)
    return x_start[:, None] + (x_goal - x_start)[:, None] * alpha[None, :]

def main():
    # =========================================================================
    # --- 1. System Parameters ---
    # =========================================================================
    print("Setting up MS-iLQR MPC parameters for double pendulum...")
    dt = 0.01
    
    # --- MPC Horizon Settings ---
    T_horizon = 1.0  # Time horizon for each MPC solve
    tspan_horizon = jnp.arange(0, T_horizon + dt, dt)
    N_horizon = len(tspan_horizon) - 1
    
    # --- Simulation Settings ---
    T_sim = 3.0 # Total simulation time
    tspan_sim = jnp.arange(0, T_sim + dt, dt)
    N_sim = len(tspan_sim) - 1
    
    # System dimensions
    n_x = 4  # [q1, q2, q1_dot, q2_dot]
    n_u = 2  # [tau1, tau2]
    
    # System dynamics parameters
    g = 9.81
    m1 = 1.0
    m2 = 1.0
    l1 = 1.0
    l2 = 1.0
    d1 = 0.1 
    d2 = 0.1 
    
    theta1 = (1/12) * m1 * l1**2
    theta2 = (1/12) * m2 * l2**2
    
    # Cost parameters
    Q = jnp.diag(jnp.array([1.0, 2.0, 0.1, 0.1]))
    R = jnp.diag(jnp.array([0.1, 0.1]))
    Q_f = jnp.diag(jnp.array([10.0, 10.0, 10.0, 10.0]))
    
    # Target: "up-up" position
    x_target = jnp.array([jnp.pi, 0.0, 0.0, 0.0])
    # Initial state
    x_0 = jnp.array([0.0, 0.0, -10.0, 10.0])
    
    # Initial control guess 
    U_init = jnp.zeros((n_u, N_horizon))
    
    # Initial State Guess
    # Set to None to force the solver to rollout U_init first
    X_init = None
    
    # Solver settings
    tol = 1e-5
    maxiter = 50 
    
    # =========================================================================
    # --- 2. Initialize System and Solver ---
    # =========================================================================
    
    # Optimizer System (faster integrator)
    # Using 'euler' for speed in MPC, d_wall=2 ensures no unwanted contact checks in mid-air
    pendulum_sys = MyDoublePendulum(
        dt=dt,
        x_target=x_target,
        Q=Q, R=R, Q_f=Q_f,
        g=g, m1=m1, m2=m2, l1=l1, l2=l2, d1=d1, d2=d2,
        theta1=theta1, theta2=theta2,
        integrator='euler',
        d_wall=2, 
        use_jit=True
    )
    
    # Simulation System (high fidelity)
    # Using 'rk4' for accurate world simulation
    pendulum_sys_sim = MyDoublePendulum(
        dt=dt,
        x_target=x_target,
        Q=Q, R=R, Q_f=Q_f,
        g=g, m1=m1, m2=m2, l1=l1, l2=l2, d1=d1, d2=d2,
        theta1=theta1, theta2=theta2,
        integrator='rk4',
        d_wall=2, 
        use_jit=True
    )
    
    # Instantiate Multiple Shooting Solver
    ms_solver = MultipleShootingiLQR(
        system=pendulum_sys,
        T=T_horizon,
        x_0=x_0,
        U_init=U_init,
        X_init=X_init,  
        tol=tol,
        maxiter=maxiter,
        verbose=False # Keep it quiet for MPC
    )

    # =========================================================================
    # --- 3. JIT Warm-up ---
    # =========================================================================
    print("Warming up JIT-compiled solver...")
    
    # 1. Warm up backward pass
    X_warmup = jnp.zeros((n_x, N_horizon + 1))
    U_warmup = jnp.zeros((n_u, N_horizon))
    ms_solver.backward_pass(X_warmup, U_warmup)[0].block_until_ready()
    
    # 2. Warm up forward pass
    U_ff_warmup = jnp.zeros((n_u, N_horizon))
    K_warmup = jnp.zeros((N_horizon, n_u, n_x))
    
    ms_solver.forward_pass(
        ms_solver.x_0, 0.0, X_warmup, U_warmup, U_ff_warmup, K_warmup
    )[0].block_until_ready()

    print("Warm-up complete.")

    # =========================================================================
    # --- 4. MPC Simulation Loop ---
    # =========================================================================
    print("Running MPC simulation (Multiple Shooting)...")
    
    # Storage
    X_sim = jnp.zeros((n_x, N_sim + 1))
    U_sim = jnp.zeros((n_u, N_sim))
    
    # Initialize
    current_x = x_0
    X_sim = X_sim.at[:, 0].set(current_x)
    
    # Warm Start Variables
    U_guess = U_init
    
    # --- FIX START ---
    # If X_init is None, initialize X_guess to zeros. 
    # The solver requires an array (not None) to function, even if it's just a placeholder.
    if X_init is None:
        X_guess = jnp.zeros((n_x, N_horizon + 1))
    else:
        X_guess = X_init
    # --- FIX END ---
    
    start_time_mpc = time.time()
    
    for k in range(N_sim):
        # 1. Update Solver State
        ms_solver.x_0 = current_x
        
        # 2. Inject Warm Starts (Crucial for MS!)
        ms_solver.U = U_guess
        ms_solver.X = X_guess # We warm start the state trajectory too
        
        # 3. Solve
        X_bar, U_bar, cost = ms_solver.optimize_trajectory()
        
        # 4. Extract Control
        uk = U_bar[:, 0]
        
        # 5. Step Real System
        xkPlusOne = pendulum_sys_sim.f_fcn(current_x, uk)
        
        # 6. Store
        U_sim = U_sim.at[:, k].set(uk)
        X_sim = X_sim.at[:, k+1].set(xkPlusOne)
        
        # 7. Shift Trajectories for Next Warm Start
        # Shift Controls
        U_guess = jnp.concatenate([U_bar[:, 1:], U_bar[:, -1:]], axis=1)
        
        # Shift States (MS Specific)
        # We shift X left, and duplicate the last state (or assume steady state)
        X_guess = jnp.concatenate([X_bar[:, 1:], X_bar[:, -1:]], axis=1)
        
        # 8. Update Loop Variable
        current_x = xkPlusOne
        
        if (k+1) % 50 == 0:
            print(f"MPC Step {k+1}/{N_sim} | Cost: {cost:.2f}")

    elapsed_time_mpc = time.time() - start_time_mpc
    print(f"MPC simulation finished.")
    print(f"Total MPC time: {elapsed_time_mpc:.4f} seconds")
    print(f"Average time per step: {elapsed_time_mpc / N_sim:.5f} seconds")


    # =========================================================================
    # --- 5. Plotting ---
    # =========================================================================
    print("Plotting results...")
    
    fig, axs = plt.subplots(3, 2, figsize=(12, 10), facecolor='w')
    fig.suptitle('MS-iLQR MPC: Double Pendulum', fontsize=16)

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

    # Plot tau2
    axs[2, 1].plot(tspan_sim[:-1], U_sim[1, :], 'k-', linewidth=2, label='tau2')
    axs[2, 1].set_xlabel('Time (s)')
    axs[2, 1].set_ylabel('Torque 2 (Nm)')
    axs[2, 1].legend()
    axs[2, 1].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

    anim = AnimationDoublePendulum(pendulum_sys_sim, X_sim, tspan_sim, dt)
    anim.animate()

if __name__ == "__main__":
    main()