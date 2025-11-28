import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time

# Import your custom classes from the other files
from class_files.systems.system_base import System
from class_files.systems.walker_6DoF_sys import Walking6DoF # <-- Import MyDoublePendulum
from class_files.iLQR_class import iLQR
from class_files.animations.animation_walker_6DoF import AnimationWalking6DoF

def main():
    # =========================================================================
    # --- 1. System Parameters (Double Pendulum) ---
    # =========================================================================
    print("Setting up walker parameters...")
    dt = 0.01
    T = 5  # Longer horizon for the harder problem
    tspan = jnp.arange(0, T + dt, dt)
    N = len(tspan) - 1
    
    
    # Target: Base at 0.8m height, joints at 0
    q_target = jnp.array([5.0, 0.96, +jnp.pi/4, -jnp.pi/8, +jnp.pi/4, -jnp.pi/4])
    v_target = jnp.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
    x_target = jnp.hstack([q_target, v_target])
    
    # Costs
    Q_diag = jnp.concatenate([
        jnp.array([1000.0, 10.0, 10.0, 10.0, 10.0, 10.0]), # q weights
        jnp.array([100.0, 100.0, 0.1, 0.1, 0.1, 0.1])     # v weights
    ])
    Q = jnp.diag(Q_diag)
    Q_f = Q * 100.0
    R = jnp.eye(4) * 0.1
    m_B = 1
    m_upper = 0.8
    m_lower = 0.3
    # --- Instantiate System ---
    robot = Walking6DoF(dt=dt, 
                        target_state=x_target,
                        Q=Q, R=R, Q_f=Q_f,
                        integrator='elastic_contact_euler',
                        mu=jnp.array([1.0, 1.0]), # High friction
                        e_restitution=jnp.array([0.0, 0.0]),
                        m_B=m_B,
                        m_lower=m_lower,
                        m_upper=m_upper)
    

    # Initial state: "down-down" position
    q_0 = jnp.array([0.0, 1.08, jnp.pi/4, -jnp.pi/4, jnp.pi/6, -jnp.pi/8]) 
    v_0 = jnp.array([2, 0.0, -0.2, 0.0, 0.0, 0.0])
    x_0 = jnp.hstack([q_0, v_0])


    # x_0 = x_target
    # x_0 = x_0.at[0].set(0.0)
    # Initial control guess (zero)
    # U_init = jnp.zeros((robot.n_u, N))
    key = jax.random.key(1)
    U_init = jax.random.uniform(key, shape=(robot.n_u, N))*1


    X_hist = [x_0]
    x_curr = x_0
    u_zero = jnp.zeros(4) 
    # Simple PD Controller to hold initial pose
    kp = 20.0
    kd = 5.0
    # q_ref = x_0[:6] # Try to stay at initial configuration
    q_ref = x_target[:6] # Try to stay at initial configuration
    start_time = time.time()
    U_hist = []
    for _ in range(N):
        
        X_hist.append(x_curr)
        q_curr = x_curr[:6]
        v_curr = x_curr[6:]
        
        # Calculate error only for actuated joints (indices 2,3,4,5)
        q_err = q_ref[2:6] - q_curr[2:6]
        v_err = 0.0 - v_curr[2:6]
        
        u_control = kp * q_err + kd * v_err
        x_curr = robot.f_fcn(x_curr, u_control)
        U_hist.append(u_control)
    
    print(f"Simulation finished in {time.time() - start_time:.4f}s")
    
    X_hist = jnp.array(X_hist)
    X_hist = X_hist[:len(tspan)]
    
    U_hist = jnp.array(U_hist)
    U_init = U_hist.T
    
    # Solver settings
    tol = 1e-5
    maxiter = 200 # More iterations for the harder problem
    
    # =========================================================================
    # --- 2. Instantiate System and Solver ---
    # =========================================================================
    print("Instantiating walker system...")
    
    
    ilqr_solver = iLQR(
        system=robot,
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


    anim = AnimationWalking6DoF(robot, X_bar.T, tspan, dt)
    anim.animate(fullscreen=True, save_video=False, filename="walking.mp4")

if __name__ == "__main__":
    main()