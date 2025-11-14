import jax
import jax.numpy as jnp
import numpy as np 
from typing import Sequence, Union
import time # Import time at the top
from jax import jit, lax # Added jit and lax for simulation test
import matplotlib.pyplot as plt # Import for plotting

# Corrected import path as you requested
from class_files.systems.system_base import System

class MyPendulum(System):
    """
    JAX-based Pendulum system.
    
    Implements the 3 abstract methods:
    - _f_cont_fcn (continuous dynamics)
    - _l_fcn (stage cost)
    - _l_f_fcn (terminal cost)
    """
    
    def __init__(self, 
                 dt: float, 
                 x_target: Union[np.ndarray, jnp.ndarray], 
                 Q: jnp.ndarray, 
                 R: jnp.ndarray, 
                 Q_f: jnp.ndarray, 
                 g: float = 9.81, 
                 l: float = 1.0, 
                 d: float = 0.01, 
                 use_jit: bool = True,
                 integrator: str = 'rk4'): # <-- New integrator arg
        """
        Constructor for the Pendulum system.
        """
        
        # 1. --- Define system properties ---
        self.n_x = 2  # [theta, theta_dot]
        self.n_u = 1  # [tau]
        
        self.g = g
        self.l = l
        self.d = d
        
        # 2. --- Store cost parameters as JAX arrays ---
        self.x_target = jnp.asarray(x_target)
        self.Q = jnp.asarray(Q)
        self.R = jnp.asarray(R)
        self.Q_f = jnp.asarray(Q_f)
        
        # 3. --- Call the base class constructor ---
        #     Pass all required arguments to the base class
        super().__init__(self.n_x, self.n_u, dt, 
                         use_jit=use_jit, 
                         integrator=integrator)
        

    # --- Implement the 3 Abstract Methods ---

    def _f_cont_fcn(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """
        Continuous-time dynamics: x_dot = f(x, u)
        """
        # Unpack state and control
        x1, x2 = x  # x1 = theta, x2 = theta_dot
        u1 = u[0]   # u1 = torque
        
        # Continuous-time dynamics (x_dot)
        # This is the ONLY part we need to define
        x_dot = jnp.array([
            x2,
            u1 - self.d * x2 - (self.g / self.l) * jnp.sin(x1)
        ])
        
        return x_dot

    def _l_fcn(self, x: jnp.ndarray, u: jnp.ndarray) -> float:
        """
        Stage cost (implementation is unchanged)
        """
        dx = x - self.x_target
        cost_x = 0.5 * dx.T @ self.Q @ dx
        cost_u = 0.5 * u.T @ self.R @ u
        # add a log barrier to keep input between -10 and 10
        # cost_u += -0.01 * (jnp.log(10 - u[0]) + jnp.log(10 + u[0]))
        
        # Your original MATLAB code scales cost by dt.
        # This is common in DDP/iLQR.
        val = (cost_x + cost_u) * self.dt 
        return val

    def _l_f_fcn(self, x: jnp.ndarray) -> float:
        """
        Terminal cost (implementation is unchanged)
        """
        dx = x - self.x_target
        val = 0.5 * dx.T @ self.Q_f @ dx
        return val
    
# --- End of Pendulum System Class ---
if __name__ == "__main__":
    # --- 1. Common Test Parameters ---
    dt = 0.01
    x_target = jnp.array([jnp.pi, 0.0])  # Target upright position
    Q = jnp.diag(jnp.array([10.0, 1.0]))
    R = jnp.array([[0.1]])
    Q_f = jnp.diag(jnp.array([100.0, 10.0]))
    g=9.81
    l=1.0
    d=0.01
    
    x_0 = jnp.array([jnp.pi - 0.1, 0.1]) # Start near target
    u_0 = jnp.array([0.5])               # Apply some control
    
    
    # --- 2. Test Euler (non-JIT) ---
    print("--- Testing 'euler' (non-JIT) ---")
    pendulum_sys_euler = MyPendulum(dt=dt, 
                                    x_target=x_target, 
                                    Q=Q, R=R, Q_f=Q_f,
                                    g=g, l=l, d=d,
                                    use_jit=False,
                                    integrator='euler')

    x_next_euler = pendulum_sys_euler.f_fcn(x_0, u_0)
    print(f"Next state (Euler): {x_next_euler}")
    f_x_euler = pendulum_sys_euler.f_x_fcn(x_0, u_0)
    print(f"f_x (Euler):\n{f_x_euler}")
    
    start_time = time.time()
    for _ in range(100):
        _ = pendulum_sys_euler.f_fcn(x_0, u_0)
        _ = pendulum_sys_euler.f_x_fcn(x_0, u_0)
    end_time = time.time()
    print(f"Time for 100 calls (non-JIT): {end_time - start_time:.6f} seconds\n")
    

    # --- 3. Test Midpoint (JIT-compiled) ---
    print("--- Testing 'midpoint' (JIT) ---")
    pendulum_sys_jit_mid = MyPendulum(dt=dt,
                                    x_target=x_target, 
                                    Q=Q, R=R, Q_f=Q_f,
                                    g=g, l=l, d=d,
                                    use_jit=True,
                                    integrator='midpoint')
    
    # Warm-up call (JIT compilation)
    print("JIT Compiling 'midpoint'...")
    x_next_mid_jit = pendulum_sys_jit_mid.f_fcn(x_0, u_0).block_until_ready()
    f_x_mid_jit = pendulum_sys_jit_mid.f_x_fcn(x_0, u_0).block_until_ready()
    print(f"Next state (Midpoint, JIT): {x_next_mid_jit}")
    print(f"f_x (Midpoint, JIT):\n{f_x_mid_jit}")

    start_time = time.time()
    for _ in range(100):
        _ = pendulum_sys_jit_mid.f_fcn(x_0, u_0).block_until_ready()
        _ = pendulum_sys_jit_mid.f_x_fcn(x_0, u_0).block_until_ready()
    end_time = time.time()
    print(f"Time for 100 calls (JIT): {end_time - start_time:.6f} seconds\n")

    
    # --- 3b. Test RK4 (JIT-compiled) ---
    print("--- Testing 'rk4' (JIT) ---")
    pendulum_sys_jit_rk4 = MyPendulum(dt=dt,
                                    x_target=x_target, 
                                    Q=Q, R=R, Q_f=Q_f,
                                    g=g, l=l, d=d,
                                    use_jit=True,
                                    integrator='rk4')
    
    # Warm-up call (JIT compilation)
    print("JIT Compiling 'rk4'...")
    x_next_rk4_jit = pendulum_sys_jit_rk4.f_fcn(x_0, u_0).block_until_ready()
    f_x_rk4_jit = pendulum_sys_jit_rk4.f_x_fcn(x_0, u_0).block_until_ready()
    print(f"Next state (RK4, JIT): {x_next_rk4_jit}")
    print(f"f_x (RK4, JIT):\n{f_x_rk4_jit}")

    start_time = time.time()
    for _ in range(100):
        _ = pendulum_sys_jit_rk4.f_fcn(x_0, u_0).block_until_ready()
        _ = pendulum_sys_jit_rk4.f_x_fcn(x_0, u_0).block_until_ready()
    end_time = time.time()
    print(f"Time for 100 calls (JIT): {end_time - start_time:.6f} seconds\n")

    
    # --- 4. Test Backward Euler (non-JIT) ---
    print("--- Testing 'backward_euler' (non-JIT) ---")
    pendulum_sys_be = MyPendulum(dt=dt, 
                                 x_target=x_target, 
                                 Q=Q, R=R, Q_f=Q_f,
                                 g=g, l=l, d=d,
                                 use_jit=False,
                                 integrator='backward_euler')

    x_next_be = pendulum_sys_be.f_fcn(x_0, u_0)
    print(f"Next state (Backward Euler): {x_next_be}")
    f_x_be = pendulum_sys_be.f_x_fcn(x_0, u_0)
    print(f"f_x (Backward Euler):\n{f_x_be}")
    
    # Check if results are reasonable (should be close to Euler for small dt)
    print(f"Difference from Euler (state): {jnp.linalg.norm(x_next_be - x_next_euler):.2e}")
    print(f"Difference from Euler (f_x): {jnp.linalg.norm(f_x_be - f_x_euler):.2e}\n")


    # --- 5. Test Backward Euler (JIT-compiled) ---
    print("--- Testing 'backward_euler' (JIT) ---")
    pendulum_sys_jit_be = MyPendulum(dt=dt,
                                    x_target=x_target, 
                                    Q=Q, R=R, Q_f=Q_f,
                                    g=g, l=l, d=d,
                                    use_jit=True,
                                    integrator='backward_euler')

    # Warm-up call (JIT compilation)
    print("JIT Compiling 'backward_euler'...")
    x_next_be_jit = pendulum_sys_jit_be.f_fcn(x_0, u_0).block_until_ready()
    f_x_be_jit = pendulum_sys_jit_be.f_x_fcn(x_0, u_0).block_until_ready()
    print(f"Next state (Backward Euler, JIT): {x_next_be_jit}")
    print(f"f_x (Backward Euler, JIT):\n{f_x_be_jit}")

    start_time = time.time()
    for _ in range(100):
        _ = pendulum_sys_jit_be.f_fcn(x_0, u_0).block_until_ready()
        _ = pendulum_sys_jit_be.f_x_fcn(x_0, u_0).block_until_ready()
    end_time = time.time()
    print(f"Time for 100 calls (JIT): {end_time - start_time:.6f} seconds\n")
    
    
    # --- 6. Forward Simulation Comparison ---
    print("--- 6. Forward Simulation Comparison ---")
    T_sim = 2.0 # Simulation time
    N_sim = int(T_sim / dt)
    
    # Use a simple sinusoidal control input
    U_sim = jnp.array([jnp.array([1.0 * jnp.sin(k * dt * 2 * jnp.pi / T_sim)]) for k in range(N_sim)])
    x_sim_0 = jnp.array([0.0, 0.0]) # Start from bottom
    
    print(f"Simulating for {T_sim}s ({N_sim} steps) from x_0 = {x_sim_0}...")

    # Define a simple simulation loop
    @jit
    def simulate_trajectory(f_fcn, x0, U):
        def scan_body(x_k, u_k):
            x_k_plus_1 = f_fcn(x_k, u_k)
            return x_k_plus_1, x_k_plus_1
        
        _, X_traj = lax.scan(scan_body, x0, U)
        # Prepend initial state for a full trajectory [x_0, ..., x_N]
        return jnp.vstack((x0, X_traj))

    # Get trajectories for each integrator
    # We use the non-JIT Euler for a clearer comparison
    X_euler = simulate_trajectory(pendulum_sys_euler.f_fcn, x_sim_0, U_sim)
    
    # We use the JIT-compiled versions for the others
    X_mid = simulate_trajectory(pendulum_sys_jit_mid.f_fcn, x_sim_0, U_sim)
    X_rk4 = simulate_trajectory(pendulum_sys_jit_rk4.f_fcn, x_sim_0, U_sim)
    X_be = simulate_trajectory(pendulum_sys_jit_be.f_fcn, x_sim_0, U_sim)

    # Print final states
    print("\n--- Final States (x_N) ---")
    print(f"Euler:          {X_euler[-1]}")
    print(f"Midpoint:       {X_mid[-1]}")
    print(f"Backward Euler: {X_be[-1]}")
    print(f"RK4 (baseline): {X_rk4[-1]}")

    # Print total difference from RK4 (sum of norms at each step)
    # Note: We compare X[1:] because X[0] is identical for all.
    print("\n--- Total Trajectory L2-Difference from RK4 ---")
    print(f"Euler vs RK4:          {jnp.linalg.norm(X_euler[1:] - X_rk4[1:]):.4e}")
    print(f"Midpoint vs RK4:       {jnp.linalg.norm(X_mid[1:] - X_rk4[1:]):.4e}")
    print(f"Backward Euler vs RK4: {jnp.linalg.norm(X_be[1:] - X_rk4[1:]):.4e}")
    
    print("\nNote: Lower difference suggests higher accuracy relative to RK4.")
    print("For a stiff system, Backward Euler would be more stable.")
    
    
    # --- 7. Plotting Trajectories ---
    print("\n--- 7. Plotting Trajectories ---")
    
    # Create time vector (N_sim + 1 points for X_traj which has N_sim + 1 states)
    t_span = jnp.linspace(0, T_sim, N_sim + 1)
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot Theta (Angle)
    axs[0].set_title(f"Pendulum State Trajectories (dt = {dt}s)")
    axs[0].plot(t_span, X_euler[:, 0], 'r:', label='Euler ($\theta$)')
    axs[0].plot(t_span, X_mid[:, 0], 'g--', label='Midpoint ($\theta$)')
    axs[0].plot(t_span, X_be[:, 0], 'b-.', label='Backward Euler ($\theta$)')
    axs[0].plot(t_span, X_rk4[:, 0], 'k-', label='RK4 (Baseline, $\theta$)')
    axs[0].set_ylabel("Angle (rad)")
    axs[0].legend(loc='upper left')
    axs[0].grid(True)
    
    # Plot Theta_dot (Angular Velocity)
    axs[1].plot(t_span, X_euler[:, 1], 'r:', label='Euler ($\dot{\theta}$)')
    axs[1].plot(t_span, X_mid[:, 1], 'g--', label='Midpoint ($\dot{\theta}$)')
    axs[1].plot(t_span, X_be[:, 1], 'b-.', label='Backward Euler ($\dot{\theta}$)')
    axs[1].plot(t_span, X_rk4[:, 1], 'k-', label='RK4 (Baseline, $\dot{\theta}$)')
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Angular Velocity (rad/s)")
    axs[1].legend(loc='upper left')
    axs[1].grid(True)
    
    plt.tight_layout()
    
    # Save the plot to a file
    plot_filename = "pendulum_trajectories.png"
    plt.savefig(plot_filename)
    
    print(f"\nTrajectory plot saved to '{plot_filename}'")
    print("Note: Run this script from your terminal to see the plot file.")