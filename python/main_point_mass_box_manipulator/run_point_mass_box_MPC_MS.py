import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time

# Import your custom classes from the other files
from class_files.systems.system_base import System
from class_files.systems.point_mass_box_manipulator_sys import MyPointMassBoxManipulator
# --- CHANGED: Import Multiple Shooting Solver ---
from class_files.MS_iLQR_class import MultipleShootingiLQR
from class_files.animations.animation_point_mass_box import AnimationPointMassBox

def main():
    # =========================================================================
    # --- 1. System Parameters ---
    # =========================================================================
    print("Setting up MS-iLQR MPC parameters for Point Mass Box Manipulator...")
    
    dt = 0.01
    # --- MPC Horizon Settings ---
    T_horizon = 1.0 # Time horizon for each MPC solve
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
    
    R = jnp.diag(jnp.array([1.0, 100.0, 1.0, 100.0]))*1e-1
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
    
    key = jax.random.key(1)
    U_init = jax.random.uniform(key, shape=(n_u, N_horizon))*100
    
    # --- X Initialization ---
    # For contact problems, linear interpolation can be dangerous (clipping objects).
    # We pass None so the solver rolls out U_init to generate the first feasible guess.
    X_init = None

    print(f"Initial State: {x_0}")
    
    # Physics parameters
    mu_ball = 0.5
    mu_ball_real = 0.5
    mu_floor = 0.1
    mu_floor_real = 0.1
    reg_friction = jnp.array([1e-2, 1e-2, 1e-2])*1e-4

    # =========================================================================
    # --- 2. Instantiate System and Solver ---
    # =========================================================================
    
    # --- Optimization System ---
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
                                            reg_friction=reg_friction)
    
    # --- Simulation System ---
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
                                                mu=jnp.array([mu_ball_real, mu_ball_real, mu_floor_real]))
    



    # =========================================================================
    # --- 4. MPC Simulation Loop ---
    # =========================================================================
    print("Running MPC simulation (Multiple Shooting)...")
    
    # Storage for simulation results
    X_sim = jnp.zeros((n_x, N_sim + 1))
    U_sim = jnp.zeros((n_u, N_sim))
    
    # Initialize simulation
    current_x = x_0
    X_sim = X_sim.at[:, 0].set(current_x)
    
    # Warm Start Variables
    U_guess = U_init
# ... inside main() ...

    # --- 1. Calculate Gravity Compensation Force ---
    # Each ball must support its own weight + half the box's weight
    total_mass_per_side = m_ball + 0.5 * m_box
    f_y_hold = total_mass_per_side * manipulator.g
    
    # Create U_init that fights gravity
    # U = [fx1, fy1, fx2, fy2]
    # We set x-forces to 0, y-forces to hold the weight
    u_static = jnp.array([0.0, f_y_hold, 0.0, f_y_hold])
    
    # Tile this for the whole horizon
    U_init = jnp.tile(u_static[:, None], (1, N_horizon))

    # --- 2. Use Rigid Transport for State (X) ---
    # We reuse the helper from before for the geometry
    def get_rigid_transport_guess(x_0, x_box_target, N):
        # [Same helper function as before]
        box_start_pos = x_0[4:6]
        box_end_pos = x_box_target[:2]
        delta_pos = box_end_pos - box_start_pos
        t = jnp.linspace(0.0, 1.0, N + 1)
        alpha = 0.5 * (1.0 - jnp.cos(jnp.pi * t))
        trajectory_shift = delta_pos[None, :] * alpha[:, None]
        X_guess = jnp.repeat(x_0[None, :], N + 1, axis=0)
        X_guess = X_guess.at[:, 0:2].add(trajectory_shift) # Ball 1
        X_guess = X_guess.at[:, 2:4].add(trajectory_shift) # Ball 2
        X_guess = X_guess.at[:, 4:6].add(trajectory_shift) # Box
        return X_guess

    X_init = get_rigid_transport_guess(x_0, x_box_target, N_horizon).T
    
    print("Initialized with Gravity Compensated Rigid Transport.")

    # --- 3. Instantiate Solver ---
    ms_solver = MultipleShootingiLQR(
        system=manipulator,
        T=T_horizon,
        x_0=x_0,
        U_init=U_init,   # <--- Now passing non-zero, gravity-aware controls
        X_init=X_init,   # <--- And the geometric path
        tol=tol,
        maxiter=maxiter, # MS might need slightly more iters to close gaps
        verbose=False 
    )

    X_init = get_rigid_transport_guess(x_0, x_box_target, N_horizon).T*0

    if X_init is None:
        X_guess = jnp.zeros((n_x, N_horizon + 1))
    else:
        X_guess = X_init

            # =========================================================================
    # --- 3. JIT Warm-up ---
    # =========================================================================
    print("Warming up JIT-compiled solver...")
    
    # 1. Warm up the backward pass
    X_warmup = jnp.zeros((n_x, N_horizon + 1))
    U_warmup = jnp.zeros((n_u, N_horizon))
    ms_solver.backward_pass(X_warmup, U_warmup)[0].block_until_ready()
    
    # 2. Warm up the forward pass
    U_ff_warmup = jnp.zeros((n_u, N_horizon))
    K_warmup = jnp.zeros((N_horizon, n_u, n_x))
    
    ms_solver.forward_pass(
        ms_solver.x_0, 0.0, X_warmup, U_warmup, U_ff_warmup, K_warmup
    )[0].block_until_ready()

    print("Warm-up complete.")

    start_time_mpc = time.time()
    
    for k in range(N_sim):
        # 1. Update the solver's initial state
        ms_solver.x_0 = current_x
        
        # 2. Inject Warm Starts
        ms_solver.U = U_guess
        ms_solver.X = X_guess # Warm start states
        
        # 3. Solve the optimization problem
        X_bar, U_bar, cost = ms_solver.optimize_trajectory()
        
        # 4. Get the first control input
        uk = U_bar[:, 0]
        
        # 5. Simulate the "real" system one step forward
        xkPlusOne = manipulator_sim.f_fcn(current_x, uk)
        
        # 6. Store results
        U_sim = U_sim.at[:, k].set(uk)
        X_sim = X_sim.at[:, k+1].set(xkPlusOne)
        
        # 7. Shift Trajectories for Next Warm Start
        U_guess = jnp.concatenate([U_bar[:, 1:], U_bar[:, -1:]], axis=1)
        X_guess = jnp.concatenate([X_bar[:, 1:], X_bar[:, -1:]], axis=1)
        
        # 8. Update the current state
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
    
    # Convert JAX arrays to Numpy for plotting
    X_plot = X_sim.T
    t_plot = tspan_sim
    
    # Ensure lengths match
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