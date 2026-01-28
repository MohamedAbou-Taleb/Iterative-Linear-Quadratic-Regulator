import jax
import jax.numpy as jnp
import numpy as np
from typing import Union
import time 
from jax import jit, lax 
from jax import jacfwd, jacrev, grad, hessian
import matplotlib.pyplot as plt 

# --- IMPORT THE PARAMETRIC iLQR CLASS ---
# Ensure you have saved iLQR_Parametric in the class_files folder
try:
    from class_files.iLQR_class import iLQR_Parametric
except ImportError:
    # Fallback if in same directory
    from iLQR_parametric import iLQR_Parametric

# Robust import for System base
try:
    from class_files.systems.system_base import System
except ImportError:
    try:
        from system_base import System
    except ImportError:
        print("Warning: system_base not found.")

class SurfaceBoxMPC(System):

    def __init__(self, 
                 surface_box_sys: System,
                 T_horizon: float = 1.0,
                 Q: jnp.ndarray = None,
                 R: jnp.ndarray = None,
                 Q_f: jnp.ndarray = None,
                 ctrl_dt: float = None):
        """
        MPC for the rigid body box (3 DoF: x, y, phi).
        State x: [x, y, phi, vx, vy, vphi] (6x1)
        Control u: [Fx, Fy, Tau] (3x1)
        """
        self.T_horizon = T_horizon
        self.dt = surface_box_sys.dt
        
        # Default Weights
        if Q is None:
            self.Q = jnp.diag(jnp.array([10.0, 10.0, 1.0, 1.0, 1.0, 1.0]))
        else:
            self.Q = Q
            
        if R is None:
            self.R = jnp.diag(jnp.array([0.1, 0.1, 0.1]))
        else:
            self.R = R
            
        if Q_f is None:
            self.Q_f = self.Q * 10.0
        else:
            self.Q_f = Q_f

        # Physics parameters
        self.m_box = surface_box_sys.m_box
        self.theta_box = surface_box_sys.theta_box
        self.g = surface_box_sys.g

        # Integrator selection matching the system
        integrator = getattr(surface_box_sys, "integrator_name", "rk4")
        if integrator == "rk4":
            self._f_fcn = self._rk4_integrator
        elif integrator == "midpoint":
            self._f_fcn = self._midpoint_integrator
        elif integrator == "euler":
            self._f_fcn = self._euler_integrator
        else:
            self._f_fcn = self._rk4_integrator

        # JIT compile dynamics and derivatives
        self._f_fcn_jit = jit(self._f_fcn)
        _f_x = jacfwd(self._f_fcn, argnums=0)
        _f_u = jacfwd(self._f_fcn, argnums=1)
        
        self.f_fcn = self._f_fcn_jit
        self.f_x_fcn = jit(_f_x)
        self.f_u_fcn = jit(_f_u)
        
        # Target state [x, y, phi, vx, vy, vphi]
        if hasattr(surface_box_sys, 'box_target_state'):
             self.x_target = surface_box_sys.box_target_state
        else:
             self.x_target = jnp.zeros(6)

        self.n_x = 6
        self.n_u = 3
        
        # Time setup
        self.tspan = jnp.arange(0, T_horizon + self.dt, self.dt)
        self.N = len(self.tspan) - 1
        
        if ctrl_dt is None:
            self.ctrl_dt = self.dt
        else:
            self.ctrl_dt = ctrl_dt
        self.tspan_ctrl = jnp.arange(0, T_horizon + self.ctrl_dt, self.ctrl_dt)
        self.N_ctrl = len(self.tspan_ctrl) - 1

        # Instantiate PARAMETRIC iLQR
        x_0 = jnp.zeros(self.n_x)
        self.iLQR = iLQR_Parametric(
            system=self, 
            T=self.T_horizon, 
            x_0=x_0, 
            U_init=jnp.zeros((self.n_u, self.N_ctrl)), 
            ctrl_dt=self.ctrl_dt
        )

        # JIT compile Cost functions and derivatives
        # Note: These point to the functions defined below that accept 'params'
        self.l_fcn = jit(self._l_fcn)
        self.l_x_fcn = jit(grad(self._l_fcn, argnums=0))
        self.l_u_fcn = jit(grad(self._l_fcn, argnums=1))
        self.l_xx_fcn = jit(hessian(self._l_fcn, argnums=0))
        self.l_uu_fcn = jit(hessian(self._l_fcn, argnums=1))
        self.l_ux_fcn = jit(jacfwd(grad(self._l_fcn, argnums=1), argnums=0))

        self.l_f_fcn = jit(self._l_f_fcn)
        self.l_f_x_fcn = jit(grad(self._l_f_fcn, argnums=0))
        self.l_f_xx_fcn = jit(hessian(self._l_f_fcn, argnums=0))

        # --- Initial u_target Computation ---
        self.u_target = self.compute_u_target(self.x_target)
        print("Calculated u_target:\n", self.u_target)


    def compute_u_target(self, x_target_val):
        """Calculates u_target for a given x_target dynamically."""
        zero_u = jnp.zeros(self.n_u)
        zero_x = jnp.zeros(self.n_x)
        
        # Re-compute matrices at the NEW target
        A = self.f_x_fcn(x_target_val, zero_u)
        B = self.f_u_fcn(x_target_val, zero_u)
        bias = self.f_fcn(zero_x, zero_u)
        
        # u_target = pinv(B) @ ((I - A) @ x_target - bias)
        u_tgt = jnp.linalg.pinv(B) @ ((jnp.eye(self.n_x) - A) @ x_target_val - bias)
        return u_tgt


    # =========================================================
    # PHYSICS & INTEGRATORS (Restored to fix n_q error)
    # =========================================================

    def _f_cont_fcn(self, x, u):
        """
        Continuous time dynamics for the 3-DoF Box.
        x = [x, y, phi, vx, vy, vphi]
        u = [Fx, Fy, Tau]
        """
        # State: x, y, phi, vx, vy, vphi
        v = x[3:6]
        
        # Forces: Fx, Fy, Tau
        ax = u[0] / self.m_box
        ay = (u[1] / self.m_box) - self.g
        alpha = u[2] / self.theta_box
        
        acc = jnp.array([ax, ay, alpha])
        
        x_dot = jnp.concatenate([v, acc])
        return x_dot

    def _euler_integrator(self, x, u):
        x_dot = self._f_cont_fcn(x, u)
        return x + x_dot * self.dt

    def _midpoint_integrator(self, x, u):
        k1 = self._f_cont_fcn(x, u)
        x_mid = x + (self.dt / 2.0) * k1
        k2 = self._f_cont_fcn(x_mid, u)
        return x + self.dt * k2

    def _rk4_integrator(self, x, u):
        k1 = self._f_cont_fcn(x, u)
        k2 = self._f_cont_fcn(x + self.dt / 2 * k1, u)
        k3 = self._f_cont_fcn(x + self.dt / 2 * k2, u)
        k4 = self._f_cont_fcn(x + self.dt * k3, u)
        return x + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # =========================================================
    # ABSTRACT METHODS (Placeholders to fix TypeError)
    # =========================================================

    def _contact_jacobian(self, q):
         return None
    
    def _gap_function(self, q):
         return None
    
    def _generalized_forces(self, q, v, u):
         return None
    
    def _mass_matrix(self, q):
         return None

    # =========================================================
    # COST FUNCTIONS (Updated for params)
    # =========================================================

    def _l_f_fcn(self, x, params):
         # Unpack params
         x_ref, _ = params
         err_x = x - x_ref
         l_f = 0.5 * err_x.T @ self.Q_f @ err_x
         return l_f
    
    def _l_fcn(self, x, u, params):
         # Unpack params
         x_ref, u_ref = params
         err_x = x - x_ref
         err_u = u - u_ref
         l = 0.5 * err_x.T @ self.Q @ err_x + 0.5 * err_u.T @ self.R @ err_u
         return l
    
    # =========================================================
    # OPTIMIZATION LOOP
    # =========================================================

    def optimize_trajectory(self, x_0: jnp.ndarray, x_target_current: jnp.ndarray = None):
        """
        Runs the MPC optimization.
        Args:
            x_0: Current state.
            x_target_current: Optional new target state. If None, uses defaults.
        """
        # 1. Handle dynamic target
        if x_target_current is None:
            x_ref = self.x_target
            u_ref = self.u_target
        else:
            # Dynamically calculate u_target for the new x_target
            x_ref = x_target_current
            u_ref = self.compute_u_target(x_target_current)
            
        params = (x_ref, u_ref)

        # 2. Pass params to iLQR_Parametric
        self.iLQR.x_0 = x_0
        X_bar, U_bar, cost = self.iLQR.optimize_trajectory(params=params)
        
        # Calculate accelerations for reference (ddq)
        def compute_acc(u_k):
            return jnp.array([u_k[0]/self.m_box, 
                              (u_k[1]/self.m_box) - self.g, 
                              u_k[2]/self.theta_box])
            
        ddqdt = jax.vmap(compute_acc)(U_bar.T).T
        
        return X_bar, U_bar, ddqdt, cost


if __name__ == "__main__":
    # --- Test Block ---
    from class_files.systems.surface_box_manipulator_sys import MySurfaceBoxManipulator

    # --- Parameters ---
    dt = 0.01
    
    # Default Target: Hover at y=1.0, phi=0.0
    x_target_default = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    
    # --- Instantiate System (Mock or Real) ---
    manipulator = MySurfaceBoxManipulator(
        dt=dt,
        box_target_state=x_target_default,
        R=jnp.eye(6), 
        Q_box=jnp.eye(6), 
        RN_list=[], 
        Q_f=jnp.eye(6), 
        RN_f_list=[] 
    )

    # --- Instantiate MPC ---
    Q = jnp.diag(jnp.array([10.0, 10.0, 10.0, 10.0, 10.0, 1.0]))
    R = jnp.diag(jnp.array([1.0, 1.0, 1.0*1e-1]))
    Q_f = Q * 10.0
    
    mpc = SurfaceBoxMPC(surface_box_sys=manipulator, T_horizon=1.0,
                        Q=Q, R=R, Q_f=Q_f)

    # --- Initial State ---
    x0 = jnp.array([0.0, 2*manipulator.h_box/2, 20*jnp.pi/180, 0.0, 0.0, 0.0])

    print("Initial State:", x0)

    # --- Closed-Loop Simulation ---
    T_sim = 5.0
    tspan_sim = jnp.arange(0, T_sim + mpc.dt, mpc.dt)
    N_sim = len(tspan_sim) - 1
    
    x_current = x0
    X = jnp.zeros((mpc.n_x, N_sim + 1))
    U = jnp.zeros((mpc.n_u, N_sim))
    
    X = X.at[:, 0].set(x_current)
    
    print(f"\nStarting Closed-Loop Simulation ({T_sim}s)...")
    start_time = time.time()
    
    for k in range(N_sim):
        # --- DYNAMIC TARGET LOGIC ---
        # Example: Change target at t > 2.0s
        if k * mpc.dt > 2.0:
            # Move to x=0.5, y=1.5
            current_target = jnp.array([0.5, 1.5, 0.0, 0.0, 0.0, 0.0])
        else:
            current_target = mpc.x_target # Default
            
        # 1. Optimize Trajectory (MPC Step) with DYNAMIC TARGET
        X_bar, U_bar, ddqdt, cost = mpc.optimize_trajectory(
            x_0=x_current,
            x_target_current=current_target
        )
        
        # 2. Extract First Control Action
        uk = U_bar[:, 0]
        
        # 3. Apply Control to System (Step Dynamics)
        x_next = mpc.f_fcn(x_current, uk)
        
        # 4. Update State and Logs
        x_current = x_next
        X = X.at[:, k+1].set(x_current)
        U = U.at[:, k].set(uk)
        
        if k % 10 == 0:
            print(f"Step {k}/{N_sim} | Cost: {cost:.4f}")

    print(f"Simulation finished in {time.time() - start_time:.4f}s")

    # --- Plotting ---
    plt.figure(figsize=(10, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(tspan_sim, X[0, :], label='x')
    plt.plot(tspan_sim, X[1, :], label='y')
    plt.plot(tspan_sim, X[2, :], label='phi')
    plt.axvline(x=2.0, color='k', linestyle='--', label='Target Change')
    plt.title('Box Position/Orientation (Closed Loop)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(tspan_sim, X[3, :], label='vx')
    plt.plot(tspan_sim, X[4, :], label='vy')
    plt.plot(tspan_sim, X[5, :], label='vphi')
    plt.title('Box Velocities')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.step(tspan_sim[:-1], U[0, :], where='post', label='Fx')
    plt.step(tspan_sim[:-1], U[1, :], where='post', label='Fy')
    plt.step(tspan_sim[:-1], U[2, :], where='post', label='Tau')
    plt.title('Control Inputs')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()