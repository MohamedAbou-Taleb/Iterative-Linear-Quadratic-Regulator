import jax
import jax.numpy as jnp
import numpy as np
from typing import Union
import time 
from jax import jit, lax 
from jax import jacfwd, jacrev, grad, hessian
import matplotlib.pyplot as plt 
from class_files.iLQR_class import iLQR

# Robust import
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
        # We need these for the u_target calculation logic
        _f_x = jacfwd(self._f_fcn, argnums=0)
        _f_u = jacfwd(self._f_fcn, argnums=1)
        
        self.f_fcn = self._f_fcn_jit
        self.f_x_fcn = jit(_f_x)
        self.f_u_fcn = jit(_f_u)

        # Physics parameters
        self.m_box = surface_box_sys.m_box
        self.theta_box = surface_box_sys.theta_box
        self.g = surface_box_sys.g
        
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

        # Instantiate iLQR
        x_0 = jnp.zeros(self.n_x)
        self.iLQR = iLQR(system=self, T=self.T_horizon, x_0=x_0, 
                         U_init=jnp.zeros((self.n_u, self.N_ctrl)), 
                         ctrl_dt=self.ctrl_dt)

        # JIT compile Cost functions and derivatives
        self.l_fcn = jit(self._l_fcn)
        self.l_x_fcn = jit(grad(self._l_fcn, argnums=0))
        self.l_u_fcn = jit(grad(self._l_fcn, argnums=1))
        self.l_xx_fcn = jit(hessian(self._l_fcn, argnums=0))
        self.l_uu_fcn = jit(hessian(self._l_fcn, argnums=1))
        self.l_ux_fcn = jit(jacfwd(grad(self._l_fcn, argnums=1), argnums=0))

        self.l_f_fcn = jit(self._l_f_fcn)
        self.l_f_x_fcn = jit(grad(self._l_f_fcn, argnums=0))
        self.l_f_xx_fcn = jit(hessian(self._l_f_fcn, argnums=0))

        # --- u_target Computation (User Method) ---
        zero_u = jnp.zeros(self.n_u)
        zero_x = jnp.zeros(self.n_x)

        # Compute Jacobians at target state with zero control
        A = self.f_x_fcn(self.x_target, zero_u)
        B = self.f_u_fcn(self.x_target, zero_u)
        
        # Compute bias at zero state/zero control
        bias = self.f_fcn(zero_x, zero_u)

        print("A matrix at target state:\n", A)
        print("B matrix at target state:\n", B)
        print("I-A matrix at target state:\n", jnp.eye(self.n_x) - A)
        
        # Compute u_target using pseudo-inverse
        # Formula: u_target = pinv(B) @ ((I - A) @ x_target - bias)
        self.u_target = jnp.linalg.pinv(B) @ ((jnp.eye(self.n_x) - A) @ self.x_target - bias)
        print("Calculated u_target:\n", self.u_target)


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
    
    def _f_cont_fcn(self, x, u):
        # State: x, y, phi, vx, vy, vphi
        v = x[3:6]
        
        # Forces: Fx, Fy, Tau
        ax = u[0] / self.m_box
        ay = (u[1] / self.m_box) - self.g
        alpha = u[2] / self.theta_box
        
        acc = jnp.array([ax, ay, alpha])
        
        x_dot = jnp.concatenate([v, acc])
        return x_dot
    
    def _contact_jacobian(self, q):
         return super()._contact_jacobian(q)
    
    def _gap_function(self, q):
         return super()._gap_function(q)
    
    def _generalized_forces(self, q, v, u):
         return super()._generalized_forces(q, v, u)
    
    def _mass_matrix(self, q):
         return super()._mass_matrix(q)
    
    def _l_f_fcn(self, x):
         err_x = x - self.x_target
         l_f = 0.5 * err_x.T @ self.Q_f @ err_x
         return l_f
    
    def _l_fcn(self, x, u):
         err_x = x - self.x_target
         err_u = u - self.u_target
         l = 0.5 * err_x.T @ self.Q @ err_x + 0.5 * err_u.T @ self.R @ err_u
         return l
    
    def optimize_trajectory(self, x_0: jnp.ndarray):
        self.iLQR.x_0 = x_0
        X_bar, U_bar, cost = self.iLQR.optimize_trajectory()
        
        # Calculate accelerations for reference (ddq)
        def compute_acc(u_k):
            return jnp.array([u_k[0]/self.m_box, 
                              (u_k[1]/self.m_box) - self.g, 
                              u_k[2]/self.theta_box])
            
        ddqdt = jax.vmap(compute_acc)(U_bar.T).T
        
        return X_bar, U_bar, ddqdt, cost

    def u_box_of_lambda(self, q, _lambda):
            """
            Maps contact forces (lambda) to the net wrench (Force/Torque) on the box.
            
            Args:
                q: Full 9-DOF configuration [q_EE1, q_EE2, q_box] (numpy or jax array).
                _lambda: 12-DOF contact force vector.
            
            Returns:
                u_box: [Fx, Fy, Tau] (3x1) acting on the box center of mass.
            """
            # Call the contact jacobian from the system object
            # W has shape (9, 12) corresponding to the 9 DOFs of the full system
            W = self.sys._contact_jacobian(q)
            
            # Extract rows corresponding to the Box (indices 6, 7, 8)
            # q indices: 0-2 (EE1), 3-5 (EE2), 6-8 (Box)
            W_box = W[6:9, :]
            
            # Calculate net wrench: u = W * lambda
            u_box = W_box @ _lambda
            return u_box

if __name__ == "__main__":
    # --- Test Block ---

    from class_files.systems.surface_box_manipulator_sys import MySurfaceBoxManipulator


    # --- Parameters ---
    dt = 0.01
    
    # Target: Hover at y=1.0, phi=0.0
    x_target = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    
    # --- Instantiate System (Mock or Real) ---
    manipulator = MySurfaceBoxManipulator(
        dt=dt,
        box_target_state=x_target,
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
    print("Target State:", mpc.x_target)

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
        # 1. Optimize Trajectory (MPC Step)
        X_bar, U_bar, ddqdt, cost = mpc.optimize_trajectory(x_0=x_current)
        
        # 2. Extract First Control Action
        uk = U_bar[:, 0]
        
        # 3. Apply Control to System (Step Dynamics)
        # In a full physics loop, we would actuate EEs. Here, we simulate
        # the box reacting to the ideal wrench computed by MPC.
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
    plt.axhline(y=mpc.x_target[1], color='r', linestyle='--', alpha=0.5, label='Target Y')
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
    # U is (N_u, N_sim), tspan is N_sim + 1
    plt.step(tspan_sim[:-1], U[0, :], where='post', label='Fx')
    plt.step(tspan_sim[:-1], U[1, :], where='post', label='Fy')
    plt.step(tspan_sim[:-1], U[2, :], where='post', label='Tau')
    plt.axhline(y=mpc.u_target[1], color='r', linestyle='--', alpha=0.5, label='Gravity Comp')
    plt.title('Control Inputs')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    # --- Animation ---
    from class_files.animations.animation_surface_box import AnimationSurfaceBox
    
    print("\nPreparing Animation...")
    
    # 1. Prepare Full State Vector for Animation
    # Animation expects (9, N) matrix: [EE1(3), EE2(3), Box(3)]
    # MPC history X is (6, N): [x, y, phi, vx, vy, vphi]
    
    N_steps = X.shape[1]
    X_anim = np.zeros((9, N_steps))
    
    # 2. Set Box States (from Closed-Loop History)
    X_anim[6, :] = X[0, :] # x_box
    X_anim[7, :] = X[1, :] # y_box
    X_anim[8, :] = X[2, :] # phi_box
    
    # 3. Set Static EE States (Visualization Only)
    # EE1 (Left)
    X_anim[0, :] = -0.7  # x
    X_anim[1, :] = 0.5   # y
    X_anim[2, :] = 0.0   # phi
    # EE2 (Right)
    X_anim[3, :] = 0.7   # x
    X_anim[4, :] = 0.5   # y
    X_anim[5, :] = 0.0   # phi
    
    # 4. Run Animation
    anim = AnimationSurfaceBox(manipulator, X_anim, tspan_sim, mpc.dt)
    anim.animate(save_video=False, filename="mpc_box_closed_loop.mp4", fullscreen=False)