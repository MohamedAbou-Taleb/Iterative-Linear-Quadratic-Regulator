import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time
import numpy as np

# =============================================================================
# IMPORTS
# =============================================================================
try:
    from class_files.systems.walker_7DoF_sys import Walking7DoF 
    from class_files.MS_iLQR_class import MultipleShootingiLQR
    from class_files.animations.animation_walker_7DoF import AnimationWalking7DoF
except ImportError as e:
    print("Error importing custom classes.")
    print(e)
    exit()

def main():
    # =========================================================================
    # --- 1. System Parameters & Configuration ---
    # =========================================================================
    print("--- Setting up Walker 7DoF Parameters ---")
    dt = 0.01
    T = 5.0  # Shorter horizon is often easier to stabilize first
    tspan = jnp.arange(0, T + dt, dt)
    N = len(tspan) - 1
    
    # Physical Mass Parameters
    m_B = 1.5      
    m_upper = 0.5  
    m_lower = 0.2  

    # --- Targets ---
    # [x, y, theta, hip1, knee1, hip2, knee2]
    # We want to move forward to x=1.5
    q_target = jnp.array([1.5, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0]) 
    v_target = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # Target velocity 1 m/s
    x_target = jnp.hstack([q_target, v_target]) 
    
    # --- Costs ---
    # Lower costs on positions initially to allow the gait to emerge
    Q_diag_pos = jnp.array([100.0, 500.0, 100.0, 0.0, 0.0, 0.0, 0.0])
    Q_diag_vel = jnp.array([10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0])
    
    Q = jnp.diag(jnp.concatenate([Q_diag_pos, Q_diag_vel]))
    Q_f = Q * 100.0 
    
    R = jnp.eye(4) * 0.1 
    reg_friction = jnp.array([1e-3, 1e-3])

    robot = Walking7DoF(
        dt=dt, 
        target_state=x_target,
        Q=Q, R=R, Q_f=Q_f,
        integrator='moreau', 
        mu=jnp.array([0.9, 0.9]),           
        e_restitution=jnp.array([0.0, 0.0]),
        m_B=m_B, m_lower=m_lower, m_upper=m_upper,
        reg_friction=reg_friction, smooth_epsilon=1e-3
    )

    # --- Initial State ---
    # Start standing
    q_0 = jnp.array([0.0, 1.0, 0.0, 0.3, -0.6, 0.3, -0.6]) # Slight crouch
    v_0 = jnp.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x_0 = jnp.hstack([q_0, v_0])

# =========================================================================
    # --- 2. Phase 1: Generate "Static Standing" Guess (Safe Mode) ---
    # =========================================================================
    print("Generating 'Static Standing' guess (Finite Cost Safety)...")
    
    # 1. Create a trajectory where the robot stays at x_0 for the whole time.
    #    (Or ideally, interpolates to x_target, but x_0 is safer to start).
    X_guess_array = jnp.tile(x_0[:, None], (1, N + 1))
    
    # 2. Gravity Compensation (Crucial for MS-iLQR stability)
    #    Calculate roughly what force is needed to hold the robot up.
    #    Total mass approx: m_B + 2*m_upper + 2*m_lower = 1.5 + 1.0 + 0.4 = 2.9 kg
    #    Force y = m * g = 2.9 * 9.81 ≈ 28.5 N
    #    Each leg takes half -> ~14 N vertical force.
    #    This is a rough guess for the 'knee' and 'hip' torques to hold it.
    #    Let's just apply a constant upward holding torque guess or zeros.
    
    #    Initialize controls to zero (Let the solver find the gravity compensation)
    #    This is safe because MS-iLQR handles the "falling" defect in the first iter.
    U_guess_array = jnp.zeros((robot.n_u, N))

    print("Static guess generated. Solver will fix the dynamics.")

    # =========================================================================
    # --- 3. Instantiate MS-iLQR Solver ---
    # =========================================================================
    print("Instantiating Multiple Shooting iLQR Solver...")
    
    ms_solver = MultipleShootingiLQR(
        system=robot,
        T=T,
        x_0=x_0,
        # Warm start BOTH U and X.
        # This gives the solver the "shape" of walking (X) and the forces to do it (U)
        U_init=U_guess_array,       
        X_init=X_guess_array,
        tol=1e-3,  # Looser tolerance for complex walking
        maxiter=150,
        verbose=True
    )

    # =========================================================================
    # --- 4. JIT Warm-up ---
    # =========================================================================
    print("Warming up JAX functions...")
    X_warmup = jnp.zeros_like(ms_solver.X)
    U_warmup = jnp.zeros_like(ms_solver.U)
    ms_solver.backward_pass(X_warmup, U_warmup)[0].block_until_ready()
    
    U_ff_warmup = jnp.zeros_like(ms_solver.U_ff)
    K_warmup = jnp.zeros_like(ms_solver.K)
    ms_solver.forward_pass(
        ms_solver.x_0, 0.0, X_warmup, U_warmup, U_ff_warmup, K_warmup
    )[0].block_until_ready()
    print("Warm-up complete.")

    # =========================================================================
    # --- 5. Run Optimization ---
    # =========================================================================
    print(f"Starting MS-iLQR optimization...")
    start_time = time.time()
    X_opt, U_opt, cost = ms_solver.optimize_trajectory()
    print(f"Done in {time.time() - start_time:.4f}s. Cost: {cost:.4f}")

    # =========================================================================
    # --- 6. Visualization ---
    # =========================================================================
    print("Starting Animation...")
    anim = AnimationWalking7DoF(robot, X_opt.T, tspan, dt)
    anim.animate(fullscreen=False, save_video=False, filename="walker_ms_ilqr.mp4")

if __name__ == "__main__":
    main()