import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time

# Import your custom classes from the other files
from class_files.systems.system_base import System
from class_files.systems.point_mass_box_manipulator_sys import MyPointMassBoxManipulator # <-- Import MyDoublePendulum
from class_files.iLQR_class import iLQR
from class_files.animations.animation_point_mass_box import AnimationPointMassBox

def main():
    # =========================================================================
    # --- 1. System Parameters ---
    # =========================================================================
    print("Setting up MPC parameters for double pendulum...")
    
    dt = 0.0025
    # dt = 0.001
    # --- MPC Horizon Settings ---
    # T_horizon = 1.0 # Time horizon for each MPC solve
    # T_horizon = 0.1
    T_horizon = dt
    # T_horizon = 2  # Time horizon for each MPC solve
    tspan_horizon = jnp.arange(0, T_horizon + dt, dt)
    N_horizon = len(tspan_horizon) - 1
    
    # --- Simulation Settings ---
    T_sim = 3.0 # Total simulation time
    tspan_sim = jnp.arange(0, T_sim + dt, dt)
    N_sim = len(tspan_sim) - 1
    
    box_width = 0.5
    box_height = 0.3
    ball_radius = 0.05
    x_box_target = jnp.array([0.0, 3*box_height/2, 0.0, 0.0])
    
    # R = jnp.diag(jnp.array([1.0, 100.0, 1.0, 100.0]))*1e-1
    R = jnp.diag(jnp.array([1.0*1e-2, 1.0*1e-3, 1.0*1e-2, 1.0*1e-3]))
    Q_box = jnp.diag(jnp.array([100.0, 1000.0, 0.0, 0.0]))
    Q_f = jnp.diag(jnp.array([1.0, 1.0, 1.0, 1.0]))*100
    RN1 = 50; RN2 = 50; RN1_f = 0.0; RN2_f = 0.0
    m_box = 0.5
    m_ball = 1

    
    # --- Initial State ---
    # q = [x_b1, y_b1, x_b2, y_b2, x_box, y_box]
    q_box_x_0 = 0.4
    ball_1_to_box_distance = 0.5
    ball_2_to_box_distance = 0.3
    q_0 = jnp.array([-(box_width/2 + ball_radius) + q_box_x_0 - ball_1_to_box_distance, 0.1,
                      box_width/2 + ball_radius+ q_box_x_0 + ball_2_to_box_distance, 0.1,
                        q_box_x_0, box_height/2])
    v_0 = jnp.zeros(6,)
    x_0 = jnp.hstack([q_0, v_0])
    n_x = 12
    n_u = 4
    
    # Solver settings
    tol = 1e-5
    maxiter = 50 # Low maxiter for MPC speed
    
    # U_init = jnp.zeros((n_u, N_horizon))
    key = jax.random.key(1)
    U_init = jax.random.uniform(key, shape=(n_u, N_horizon))*100
    # U_init= jnp.vstack([10*jnp.ones((1, N)), jnp.zeros((3, N))])
    
    print(f"Initial State: {x_0}")
    # Solver settings
    tol = 1e-5
    maxiter = 700 # More iterations for the harder problem
    mu_ball = 0.5
    mu_ball_real = 0.5
    mu_floor = 0.1
    mu_floor_real = 0.1
    reg_friction = jnp.array([1e-2, 1e-2, 1e-2])*1e-4
        # --- Instantiate System ---
    manipulator = MyPointMassBoxManipulator(dt=dt, 
                                            box_target_state=x_box_target, 
                                            R=R, Q_box=Q_box, RN1=RN1, RN2=RN2,
                                            Q_f=Q_f, RN1_f=RN1_f, RN2_f=RN2_f,
                                            integrator='moreau',
                                            box_height=box_height,
                                            box_width=box_width,
                                            ball_radius=ball_radius,
                                            m_box=m_box,
                                            m_ball=m_ball,
                                            mu=jnp.array([mu_ball, mu_ball, mu_floor]),
                                            reg_friction=reg_friction) # mu=0.0 for box-floor to slide
    
    
    
    # "Real-world" simulation system
    # Use a high-fidelity integrator for the "real" plant
    manipulator_sim = MyPointMassBoxManipulator(dt=dt, 
                                            box_target_state=x_box_target, 
                                            R=R, Q_box=Q_box, RN1=RN1, RN2=RN2,
                                            Q_f=Q_f, RN1_f=RN1_f, RN2_f=RN2_f,
                                            integrator='contact_euler',
                                            box_height=box_height,
                                            box_width=box_width,
                                            ball_radius=ball_radius,
                                            m_box=m_box,
                                            m_ball=m_ball,
                                            mu=jnp.array([mu_ball_real, mu_ball_real, mu_floor_real])) # mu=0.0 for box-floor to slide
    
    ilqr_solver = iLQR(
        system=manipulator,
        T=T_horizon,
        x_0=x_0,
        U_init=U_init,
        tol=tol,
        maxiter=maxiter,
        verbose=True
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
    uk = jnp.zeros(manipulator_sim.n_u)
    U_bar = jnp.zeros([manipulator_sim.n_u, N_horizon])
    
    start_time_mpc = time.time()
    
    for k in range(N_sim):
        # 1. Update the solver's initial state
        ilqr_solver.x_0 = current_x
        
        # 2. Provide the warm-start control guess
        ilqr_solver.U = U_guess
        
        # 3. Solve the optimization problem
        g_N = manipulator_sim._gap_function(current_x[:manipulator_sim.n_q])
        if g_N[0] <= 0.0 and g_N[1] <= 0.0:
            X_bar, U_bar, cost = ilqr_solver.optimize_trajectory()
            uk = U_bar[:, 0]
            
        # else:
        #     U_bar = jnp.zeros([manipulator_sim.n_u, N_horizon])
        
        
        # 4. Get the first control input
        
        # uk = jnp.array([0.0, 0.0])  # No control for testing
        
        # 5. Simulate the "real" system one step forward
        xkPlusOne = manipulator_sim.f_fcn(current_x, uk)
        
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
    
    # Convert JAX arrays to Numpy for plotting
    X_plot = X_sim.T
    t_plot = tspan_sim
    
    # Ensure lengths match (truncate X if it has one more step than t, or vice versa)
    if len(X_plot) > len(t_plot):
        X_plot = X_plot[:len(t_plot)]
    elif len(t_plot) > len(X_plot):
        t_plot = t_plot[:len(X_plot)]

    fig, axes = plt.subplots(3, 2, figsize=(10, 5), sharex=True)
    fig.suptitle(f"Positions over Time (T={T_sim}s)", fontsize=16)

    # --- Ball 1 (Left) ---
    axes[0, 0].plot(t_plot, X_plot[:, 0], 'b-', linewidth=2, label=r'$x_{b1}$')
    axes[0, 0].set_ylabel('Position [m]')
    axes[0, 0].set_title('Ball 1 X')
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    axes[0, 1].plot(t_plot, X_plot[:, 1], 'b--', linewidth=2, label=r'$y_{b1}$')
    axes[0, 1].set_ylabel('Position [m]')
    axes[0, 1].set_title('Ball 1 Y')
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    # --- Ball 2 (Right) ---
    axes[1, 0].plot(t_plot, X_plot[:, 2], 'r-', linewidth=2, label=r'$x_{b2}$')
    axes[1, 0].set_ylabel('Position [m]')
    axes[1, 0].set_title('Ball 2 X')
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    axes[1, 1].plot(t_plot, X_plot[:, 3], 'r--', linewidth=2, label=r'$y_{b2}$')
    axes[1, 1].set_ylabel('Position [m]')
    axes[1, 1].set_title('Ball 2 Y')
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    # --- Box ---
    # Target X
    axes[2, 0].axhline(y=x_box_target[0], color='k', linestyle=':', linewidth=2, label='Target')
    axes[2, 0].plot(t_plot, X_plot[:, 4], 'g-', linewidth=2, label=r'$x_{box}$')
    axes[2, 0].set_ylabel('Position [m]')
    axes[2, 0].set_xlabel('Time [s]')
    axes[2, 0].set_title('Box X')
    axes[2, 0].grid(True)
    axes[2, 0].legend()

    # Target Y
    axes[2, 1].axhline(y=x_box_target[1], color='k', linestyle=':', linewidth=2, label='Target')
    axes[2, 1].plot(t_plot, X_plot[:, 5], 'g--', linewidth=2, label=r'$y_{box}$')
    axes[2, 1].set_ylabel('Position [m]')
    axes[2, 1].set_xlabel('Time [s]')
    axes[2, 1].set_title('Box Y')
    axes[2, 1].grid(True)
    axes[2, 1].legend()

        # =========================================================================
    # --- 6. Plotting Controls ---
    # =========================================================================
    print("Plotting control inputs...")
    U_plot = U_sim.T
    
    
    # Create time array for controls (length N)
    t_u = t_plot[:U_plot.shape[0]]
    
    fig_u, axes_u = plt.subplots(2, 2, figsize=(8, 5), sharex=True)
    fig_u.suptitle(f"Control Inputs (Forces) over Time", fontsize=16)
    
    # Ball 1 Controls (u0, u1)
    axes_u[0, 0].plot(t_u, U_plot[:, 0], 'b-', linewidth=2)
    axes_u[0, 0].set_title("u0: Ball 1 Force X")
    axes_u[0, 0].set_ylabel("Force [N]")
    axes_u[0, 0].grid(True)

    axes_u[0, 1].plot(t_u, U_plot[:, 1], 'b--', linewidth=2)
    axes_u[0, 1].set_title("u1: Ball 1 Force Y")
    axes_u[0, 1].set_ylabel("Force [N]")
    axes_u[0, 1].grid(True)
    
    # Ball 2 Controls (u2, u3)
    axes_u[1, 0].plot(t_u, U_plot[:, 2], 'r-', linewidth=2)
    axes_u[1, 0].set_title("u2: Ball 2 Force X")
    axes_u[1, 0].set_ylabel("Force [N]")
    axes_u[1, 0].grid(True)

    axes_u[1, 1].plot(t_u, U_plot[:, 3], 'r--', linewidth=2)
    axes_u[1, 1].set_title("u3: Ball 2 Force Y")
    axes_u[1, 1].set_ylabel("Force [N]")
    axes_u[1, 1].grid(True)
    
    axes_u[1, 0].set_xlabel("Time [s]")
    axes_u[1, 1].set_xlabel("Time [s]")
    
    plt.tight_layout()
    plt.show()

    anim = AnimationPointMassBox(manipulator, X_sim, tspan_sim, dt)
    anim.animate(fullscreen=True, save_video=False, filename='box_transport.mp4')


if __name__ == "__main__":
    main()