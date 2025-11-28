import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time
import numpy as np

# =============================================================================
# IMPORTS
# =============================================================================
try:
    # Update these imports to match your file naming convention
    from class_files.systems.walker_7DoF_sys import Walking7DoF 
    from class_files.iLQR_class import iLQR
    from class_files.animations.animation_walker_7DoF import AnimationWalking7DoF
except ImportError as e:
    print("Error importing custom classes. Please check your directory structure and filenames.")
    print(e)
    exit()

def main():
    # =========================================================================
    # --- 1. System Parameters & Configuration ---
    # =========================================================================
    print("--- Setting up Walker 7DoF Parameters ---")
    dt = 0.01
    T = 3.0  
    tspan = jnp.arange(0, T + dt, dt)
    N = len(tspan) - 1
    
    # Physical Mass Parameters
    m_B = 1.5      # Trunk mass
    m_upper = 0.5  # Thigh mass
    m_lower = 0.2  # Shin mass

    # --- Targets ---
    # 7 Positions: [x, y, theta_trunk, hip1, knee1, hip2, knee2]
    # We generally want the trunk upright (theta=0) and at a specific height
    q_target = jnp.array([2.0, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0]) 
    
    # 7 Velocities
    v_target = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 
    
    x_target = jnp.hstack([q_target, v_target]) # Size 14
    
    # --- Costs (Q and R) ---
    # State weights (Size 14)
    # High cost on Base Height (idx 1) and Trunk Angle (idx 2)
    Q_diag_pos = jnp.array([1000.0, 1000.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    Q_diag_vel = jnp.array([10.0, 10.0, 100.0, 0.0, 0.0, 0.0, 0.0])
    
    Q_diag = jnp.concatenate([Q_diag_pos, Q_diag_vel])
    Q = jnp.diag(Q_diag)
    Q_f = Q * 100.0 
    
    # Control weights (Size 4 for Hip/Knee only, or 6 if ankles exist)
    # Assuming 4 actuated joints here.
    R = jnp.eye(4) * 0.01 
    reg_friction = jnp.array([1e-3, 1e-3])
    # --- Instantiate System ---
    robot = Walking7DoF(
        dt=dt, 
        target_state=x_target,
        Q=Q, R=R, Q_f=Q_f,
        integrator='contact_euler', 
        mu=jnp.array([0.9, 0.9]),           
        e_restitution=jnp.array([0.0, 0.0]),
        m_B=m_B,
        m_lower=m_lower,
        m_upper=m_upper,
        reg_friction=reg_friction,
        smooth_epsilon=1e-3
    )

    # --- Initial State ---
    # [x, y, theta, h1, k1, h2, k2]
    # Start slightly above ground, trunk upright
    q_0 = jnp.array([0.0, 1.05, 0.0, jnp.pi/4, jnp.pi/8*0, 0.0, -0.0]) 
    v_0 = jnp.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x_0 = jnp.hstack([q_0, v_0])

    # =========================================================================
    # --- 2. Phase 1: Generate Initial Guess via Walking Heuristic ---
    # =========================================================================
    print(f"Generating initial 'shuffling' guess ({N} steps)...")
    
    X_hist = [x_0]
    U_hist = []
    x_curr = x_0
    
    kp = 100.0 # Stiff joints for the guess
    kd = 10.0
    
    # Walking parameters for the guess
    step_freq = 3.0  # Radians per second
    step_amp = 0.5   # Step size (radians)
    
    for k in range(N):
        t = k * dt
        q_curr = x_curr[:7]
        v_curr = x_curr[7:]
        
        # Create a time-varying reference for the hips
        # Hips are usually indices 3 (left) and 5 (right)
        # Knees are indices 4 and 6
        
        # # Anti-phase sine waves for hips
        # hip1_ref = step_amp * jnp.sin(step_freq * t)
        # hip2_ref = step_amp * jnp.sin(step_freq * t + jnp.pi) # 180 deg out of phase
        
        # # Simple knee flexion based on hip (heuristic to clear ground)
        # knee1_ref = 0.0 if hip1_ref > 0 else -1.0 * jnp.abs(hip1_ref)
        # knee2_ref = 0.0 if hip2_ref > 0 else -1.0 * jnp.abs(hip2_ref)

        # # Build reference vector [x, y, th, h1, k1, h2, k2]
        # # We want base to move forward, so update x_ref based on time
        # expected_vel = 0.3
        # q_ref_t = jnp.array([
        #     q_0[0] + expected_vel * t,  # Moving x target
        #     q_0[1],                     # Constant height
        #     0.0,                        # Upright trunk
        #     hip1_ref,
        #     knee1_ref,
        #     hip2_ref,
        #     knee2_ref
        # ])

        # PD Control
        # Indices 3:7 are the actuated joints
        q_err = q_target[3:7] - q_curr[3:7]
        v_err = 0.0 - v_curr[3:7]
        
        u_control = kp * q_err + kd * v_err
        
        x_curr = robot.f_fcn(x_curr, u_control)
        
        X_hist.append(x_curr)
        U_hist.append(u_control)

    
    X_guess = jnp.array(X_hist) 
    U_guess = jnp.array(U_hist).T *0
    U_guess = jnp.zeros((4,N))
    key = jax.random.key(1)
    U_init = jax.random.uniform(key, shape=(robot.n_u, N))*10

    # =========================================================================
    # --- 3. Instantiate iLQR Solver ---
    # =========================================================================
    print("Instantiating iLQR Solver...")
    
    ilqr_solver = iLQR(
        system=robot,
        T=T,
        x_0=x_0,
        U_init=U_guess, 
        tol=1e-4,
        maxiter=100,
        verbose=True
    )

    # =========================================================================
    # --- 4. JIT Compilation Warm-up ---
    # =========================================================================
    print("Warming up JAX functions...")
    
    # 1. Warm up Backward Pass
    X_warmup = jnp.zeros_like(ilqr_solver.X)
    U_warmup = jnp.zeros_like(ilqr_solver.U)
    ilqr_solver.backward_pass(X_warmup, U_warmup)[0].block_until_ready()
    
    # 2. Warm up Forward Pass
    U_ff_warmup = jnp.zeros_like(ilqr_solver.U_ff)
    K_warmup = jnp.zeros_like(ilqr_solver.K)
    ilqr_solver.forward_pass(
        ilqr_solver.x_0, 0.0, X_warmup, U_warmup, U_ff_warmup, K_warmup
    )[0].block_until_ready()

    print("Warm-up complete.")

    # =========================================================================
    # --- 5. Run iLQR Optimization ---
    # =========================================================================
    print(f"Starting iLQR optimization for {T}s horizon...")
    
    start_time_ilqr = time.time()
    X_opt, U_opt, final_cost = ilqr_solver.optimize_trajectory()
    elapsed_time_ilqr = time.time() - start_time_ilqr
    
    print(f" Optimization finished in: {elapsed_time_ilqr:.4f} seconds")
    print(f" Final Cost: {final_cost:.4f}")

    # =========================================================================
    # --- 6. Visualization ---
    # =========================================================================
    print("Starting Animation...")

    # Ensure your AnimationWalking7DoF class accepts X_data.T (14, N)
    anim = AnimationWalking7DoF(robot, X_opt.T, tspan, dt)
    anim.animate(fullscreen=False, save_video=False, filename="walking_7dof.mp4")

if __name__ == "__main__":
    main()