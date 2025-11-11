import jax
import jax.numpy as jnp
from typing import List
import numpy as np # Used for type hinting

# Import the base class
from .systems.system_base import System 

class iLQR:
    """
    Python/JAX implementation of the iLQR algorithm.
    
    This class takes a System object (defined in system_base.py)
    and optimizes a control trajectory U.
    """
    
    def __init__(self,
                 system: System,
                 T: float,
                 x_0: jnp.ndarray,
                 U_init: jnp.ndarray,
                 tol: float = 1e-6,
                 maxiter: int = 100,
                 alpha_factor: float = 0.5,
                 min_alpha: float = 1e-8):
        """
        Initializes the iLQR solver.
        
        Args:
            system: An instance of the System base class.
            T: Total time horizon (s).
            x_0: Initial state, shape (n_x,).
            U_init: Initial guess for control trajectory, shape (n_u, N).
            tol: Convergence tolerance for cost reduction.
            maxiter: Maximum number of iterations.
            alpha_factor: Factor for line search (e.g., 0.5).
            min_alpha: Minimum alpha for line search.
        """
        
        self.system = system
        self.T = T
        self.x_0 = jnp.asarray(x_0)
        
        # Solver settings
        self.tol = tol
        self.maxiter = maxiter
        self.alpha_factor = alpha_factor
        self.min_alpha = min_alpha

        # Get dims from system
        self.n_x = system.n_x
        self.n_u = system.n_u
        self.dt = system.dt
        
        # Time horizon
        self.tspan = jnp.arange(0, T + self.dt, self.dt)
        self.N = len(self.tspan) - 1

        # Check U_init shape
        expected_shape = (self.n_u, self.N)
        if U_init.shape != expected_shape:
            raise ValueError(f"U_init must have shape {expected_shape}, but got {U_init.shape}")
        
        # Trajectories (using (dim, time) convention from MATLAB)
        self.X = jnp.zeros((self.n_x, self.N + 1))
        self.U = jnp.asarray(U_init)
        
        # Gains
        # A list of (n_u, n_x) matrices
        self.K_cell_array: List[jnp.ndarray] = [jnp.zeros((self.n_u, self.n_x)) for _ in range(self.N)]
        # A (n_u, N) trajectory
        self.U_ff = jnp.zeros((self.n_u, self.N))


    def Q_fcn(self, x: jnp.ndarray, u: jnp.ndarray, V_x: jnp.ndarray, V_xx: jnp.ndarray):
        """
        Calculate the Q-function derivatives at a single time step.
        (Corresponds to MATLAB Q_fcn)
        
        Args:
            x: State, shape (n_x,)
            u: Control, shape (n_u,)
            V_x: Value function gradient from next step, shape (n_x,)
            V_xx: Value function Hessian from next step, shape (n_x, n_x)
            
        Returns:
            (Q_x, Q_u, Q_xx, Q_ux, Q_uu)
        """
        # Get all derivatives from the JIT-compiled system methods
        l_x = self.system.l_x_fcn(x, u)     # (n_x,)
        l_u = self.system.l_u_fcn(x, u)     # (n_u,)
        l_xx = self.system.l_xx_fcn(x, u)   # (n_x, n_x)
        l_ux = self.system.l_ux_fcn(x, u)   # (n_u, n_x)
        l_uu = self.system.l_uu_fcn(x, u)   # (n_u, n_u)
        f_x = self.system.f_x_fcn(x, u)     # (n_x, n_x)
        f_u = self.system.f_u_fcn(x, u)     # (n_x, n_u)

        # Calculate Q-function derivatives
        # (Translating from MATLAB's V_x*f_x to Python's f_x.T @ V_x)
        Q_x = l_x + f_x.T @ V_x
        Q_u = l_u + f_u.T @ V_x
        Q_xx = l_xx + f_x.T @ V_xx @ f_x
        Q_ux = l_ux + f_u.T @ V_xx @ f_x
        Q_uu = l_uu + f_u.T @ V_xx @ f_u
        
        return Q_x, Q_u, Q_xx, Q_ux, Q_uu


    def u_opt_fcn(self, x: jnp.ndarray, u: jnp.ndarray, V_x: jnp.ndarray, V_xx: jnp.ndarray):
        """
        Calculate the optimal control gains at a single time step.
        (Corresponds to MATLAB u_opt_fcn)
        """
        Q_x, Q_u, Q_xx, Q_ux, Q_uu = self.Q_fcn(x, u, V_x, V_xx)
        
        # TODO: Add regularization to Q_uu if it's not invertible
        # (e.g., Levenberg-Marquardt)
        
        # Solve for gains (MATLAB's -Q_uu \ Q_ux)
        K = -jnp.linalg.solve(Q_uu, Q_ux)
        u_ff = -jnp.linalg.solve(Q_uu, Q_u)
        
        # Update Value function derivatives for the previous time step
        # V_x_prev = Q_x + Q_u * K (MATLAB)
        # V_x_prev = Q_x + K.T @ Q_u (Python)
        V_x_prev = Q_x + K.T @ Q_u
        
        # V_xx_prev = Q_xx + Q_ux' * K (MATLAB)
        # V_xx_prev = Q_xx + Q_ux.T @ K (Python)
        V_xx_prev = Q_xx + Q_ux.T @ K
        
        return u_ff, K, V_x_prev, V_xx_prev


    def backward_pass(self):
        """
        Performs a backward pass to calculate the optimal gains.
        (Corresponds to MATLAB backward_pass)
        """
        # Get terminal cost derivatives
        x_N = self.X[:, -1]
        V_x = self.system.l_f_x_fcn(x_N)
        V_xx = self.system.l_f_xx_fcn(x_N)
        
        # Loop from k = N-1 down to 0 (MATLAB's N:-1:1)
        for k in range(self.N - 1, -1, -1):
            x = self.X[:, k]
            u = self.U[:, k]
            
            # Get optimal gains and new V-derivatives
            u_ff, K, V_x, V_xx = self.u_opt_fcn(x, u, V_x, V_xx)
            
            # Save gains
            self.U_ff = self.U_ff.at[:, k].set(u_ff)
            self.K_cell_array[k] = K


    def forward_pass(self, alpha: float):
        """
        Performs a forward pass (rollout) with new gains.
        (Corresponds to MATLAB forward_pass)
        
        Args:
            alpha: Line search parameter.
            
        Returns:
            (X_new, U_new, cost_new)
        """
        X_new = jnp.zeros_like(self.X)
        X_new = X_new.at[:, 0].set(self.x_0)
        U_new = jnp.zeros_like(self.U)
        cost_new = 0.0

        # Loop from k = 0 to N-1 (MATLAB's 1:N)
        for k in range(self.N):
            xk_new = X_new[:, k]
            xk_old = self.X[:, k]
            
            # Get controls
            delta_x = xk_new - xk_old
            K = self.K_cell_array[k]
            uk_old = self.U[:, k]
            uk_ff = self.U_ff[:, k]
            
            # Calculate new control
            # uk_new = uk_old + alpha * uk_ff + K * delta_x
            uk_new = uk_old + alpha * uk_ff + K @ delta_x
            
            # Store new control
            U_new = U_new.at[:, k].set(uk_new)
            
            # Simulate one step
            xkPlusOne = self.system.f_fcn(xk_new, uk_new)
            X_new = X_new.at[:, k+1].set(xkPlusOne)
            
            # Add stage cost
            cost_new += self.system.l_fcn(xk_new, uk_new)
            
        # Add terminal cost
        cost_new += self.system.l_f_fcn(X_new[:, -1])
        
        return X_new, U_new, cost_new


    def optimize_trajectory(self):
        """
        Runs the full iLQR optimization loop.
        (Corresponds to MATLAB optimize_trajectory)
        
        Returns:
            (X_star, U_star, cost)
        """
        
        # This initial forward pass (alpha=0) populates self.X
        # based on the initial self.U and zero'd K-gains,
        # exactly matching the MATLAB logic.
        self.X, self.U, cost = self.forward_pass(alpha=0.0)
        
        print(f"Initial cost: {cost:.4f}")
        cost_prev = cost
        
        for i in range(self.maxiter):
            # Check for convergence
            if i > 0 and abs(cost - cost_prev) <= self.tol:
                print(f"Converged at iteration {i}")
                break
            cost_prev = cost
            
            # 1. Backward pass
            self.backward_pass()
            
            # 2. Line search
            alpha = 1.0
            is_step_accepted = False
            for j in range(10): # Max 10 line search steps
                X_new, U_new, cost_new = self.forward_pass(alpha)
                
                if cost_new < cost:
                    # Accept step
                    self.X = X_new
                    self.U = U_new
                    cost = cost_new
                    is_step_accepted = True
                    print(f"  Iter {i+1} (alpha={alpha:.2e}): Cost improved to {cost:.4f}")
                    break
                else:
                    # Reduce alpha
                    alpha *= self.alpha_factor
                    if alpha < self.min_alpha:
                        break # Alpha is too small
            
            if not is_step_accepted:
                print(f"Warning: Line search failed at iteration {i+1}. Cost did not improve.")
                break
        
        if i == self.maxiter - 1:
             print(f"Warning: Reached max iterations ({self.maxiter}) without converging.")
        
        return self.X, self.U, cost
    
    if __name__ == "__main__":
        # create an instance of a simple system and run iLQR
        from systems.pendulum_sys import MyPendulum
        from iLQR_class import iLQR
        dt = 0.01
        x_target = jnp.array([jnp.pi, 0.0])  # Target upright position
        Q = jnp.diag(jnp.array([10.0, 1.0]))
        R = jnp.array([[0.1]])  
        Q_f = jnp.diag(jnp.array([100.0, 10.0]))
        use_jit = True
        g=9.81
        l=1.0
        d=0.01
        pendulum_sys = MyPendulum(dt=dt, 
                                  x_target=x_target, 
                                  Q=Q, 
                                  R=R, 
                                  Q_f=Q_f,
                                  g=g,
                                  l=l,
                                  d=d,
                                  use_jit=use_jit,
                                  integrator='rk4')
        T = 2.0  # Total time horizon
        n_u = pendulum_sys.n_u
        N = int(T / dt)
        U_init = jnp.zeros((n_u, N))  # Initial guess for control
        x_0 = jnp.array([0.0, 0.0])  # Initial state
        ilqr_solver = iLQR(system=pendulum_sys,
                           T=T,
                           x_0=x_0,
                           U_init=U_init,
                           tol=1e-6,
                           maxiter=50)
        X_star, U_star, final_cost = ilqr_solver.optimize_trajectory()
        print(f"Final optimized cost: {final_cost:.4f}")