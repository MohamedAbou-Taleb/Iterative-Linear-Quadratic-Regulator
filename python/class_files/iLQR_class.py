import jax
import jax.numpy as jnp
from jax import lax
from typing import List, Tuple, Callable
import numpy as np # Used for type hinting

# Import the base class
from .systems.system_base import System 

class iLQR:
    """
    Python/JAX implementation of the iLQR algorithm.
    
    This version uses jax.lax.scan to compile the entire
    backward and forward passes, eliminating Python loop overhead.
    """
    
    def __init__(self,
                 system: System,
                 T: float,
                 x_0: jnp.ndarray,
                 U_init: jnp.ndarray,
                 tol: float = 1e-6,
                 maxiter: int = 100,
                 alpha_factor: float = 0.5,
                 min_alpha: float = 1e-8,
                 verbose: bool = True): # <-- Added verbose flag
        
        self.system = system
        self.T = T
        self.x_0 = jnp.asarray(x_0)
        
        # Solver settings
        self.tol = tol
        self.maxiter = maxiter
        self.alpha_factor = alpha_factor
        self.min_alpha = min_alpha
        self.verbose = verbose # <-- Store verbose flag

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
        
        # Trajectories (using (dim, time) convention)
        self.X = jnp.zeros((self.n_x, self.N + 1))
        self.U = jnp.asarray(U_init)
        
        # Gains (N, n_u, n_x)
        self.K = jnp.zeros((self.N, self.n_u, self.n_x))
        # Feedforward (n_u, N)
        self.U_ff = jnp.zeros((self.n_u, self.N))

        # =====================================================================
        # --- PERFORMANCE OPTIMIZATION ---
        # JIT-compile the full scan-based passes.
        
        # 1. JIT-compiled function for the entire backward pass
        self.backward_pass = jax.jit(self._backward_pass_scan)
        
        # 2. JIT-compiled function for the entire forward pass
        #    x_0 is now a static argument, so JAX will re-compile
        #    if its *shape* changes, but it will use the new *value*
        #    on every call, which is what we need for MPC.
        #    We must pass x_0 in as an arg to avoid the "stale x_0" bug.
        self.forward_pass = jax.jit(self._forward_pass_scan)
        # =====================================================================


    def _backward_pass_body(self, carry: Tuple, inputs: Tuple):
        """
        The body function for the backward pass scan.
        Calculates gains for a single time step.
        
        Args:
            carry: (V_x, V_xx) from the *next* time step (k+1).
            inputs: (x_k, u_k) from the nominal trajectory.
            
        Returns:
            New carry: (V_x_prev, V_xx_prev) for time step k.
            Per-step outputs: (u_ff_k, K_k)
        """
        V_x, V_xx = carry
        x, u = inputs
        
        # --- 1. Get All Derivatives ---
        (l_x, l_u, l_xx, l_ux, l_uu, 
         f_x, f_u) = self._get_all_derivatives_for_backward_pass(x, u)

        # --- 2. Calculate Q-Function Derivatives ---
        Q_x = l_x + f_x.T @ V_x
        Q_u = l_u + f_u.T @ V_x
        Q_xx = l_xx + f_x.T @ V_xx @ f_x
        Q_ux = l_ux + f_u.T @ V_xx @ f_x
        Q_uu = l_uu + f_u.T @ V_xx @ f_u
        
        # TODO: Add regularization to Q_uu
        
        # --- 3. Solve for Gains (u_opt_fcn logic) ---
        # Note: Using solve is more stable than inv
        K_k = -jnp.linalg.solve(Q_uu, Q_ux)
        u_ff_k = -jnp.linalg.solve(Q_uu, Q_u)
        
        # --- 4. Update Value Function Derivatives ---
        V_x_prev = Q_x + K_k.T @ Q_u
        V_xx_prev = Q_xx + Q_ux.T @ K_k
        
        new_carry = (V_x_prev, V_xx_prev)
        outputs = (u_ff_k, K_k)
        
        return new_carry, outputs


    def _backward_pass_scan(self, X_nom: jnp.ndarray, U_nom: jnp.ndarray):
        """
        Performs a full backward pass using jax.lax.scan.
        This function is JIT-compiled.
        
        Args:
            X_nom: Nominal state trajectory, (n_x, N+1)
            U_nom: Nominal control trajectory, (n_u, N)
            
        Returns:
            U_ff: New feedforward gains, (n_u, N)
            K: New feedback gains, (N, n_u, n_x)
        """
        # Get terminal cost derivatives
        x_N = X_nom[:, -1]
        V_x = self.system.l_f_x_fcn(x_N)
        V_xx = self.system.l_f_xx_fcn(x_N)
        
        init_carry = (V_x, V_xx)
        
        # Prepare inputs for scanning: (x_k, u_k)
        # We need to scan over (X[:, :-1], U) in reverse
        # Transpose from (dim, time) to (time, dim) for scanning
        xs = (X_nom[:, :-1].T, U_nom.T)
        
        # Run the scan
        # 'reverse=True' makes it loop from N-1 down to 0
        final_carry, outputs = lax.scan(
            self._backward_pass_body, init_carry, xs, reverse=True
        )
        
        # Unpack outputs
        u_ff_stack, K_stack = outputs
        
        # Transpose U_ff from (N, n_u) back to (n_u, N)
        U_ff = u_ff_stack.T
        # K is already in (N, n_u, n_x) format, which is what we want
        K = K_stack
        
        return U_ff, K


    def _forward_pass_body(self, carry: Tuple, inputs: Tuple):
        """
        The body function for the forward pass scan.
        Simulates one time step.
        
        Args:
            carry: (x_k, cost_k)
            inputs: (xk_old, uk_old, uk_ff, K_k, alpha)
            
        Returns:
            New carry: (x_k+1, cost_k+1)
            Per-step outputs: (x_k, u_k)
        """
        xk_new, cost_new = carry
        xk_old, uk_old, uk_ff, K_k, alpha = inputs
        
        # Calculate new control
        delta_x = xk_new - xk_old
        uk_new = uk_old + alpha * uk_ff + K_k @ delta_x
        
        # Simulate one step and get stage cost
        xkPlusOne, cost_k = self._get_all_calcs_for_forward_pass(xk_new, uk_new)
        
        new_carry = (xkPlusOne, cost_new + cost_k)
        outputs = (xk_new, uk_new) # Store the state/control *used* at this step
        
        return new_carry, outputs


    def _forward_pass_scan(self,
                           x_0_arg: jnp.ndarray, # <-- Pass x_0 as an argument
                           alpha: float,
                           X_old: jnp.ndarray,
                           U_old: jnp.ndarray,
                           U_ff: jnp.ndarray,
                           K: jnp.ndarray):
        """
        Performs a full forward pass (rollout) using jax.lax.scan.
        This function is JIT-compiled.
        
        Args:
            x_0_arg: Initial state (as an argument, not from self).
            alpha: Line search parameter.
            X_old: Previous state trajectory, (n_x, N+1)
            U_old: Previous control trajectory, (n_u, N)
            U_ff: Feedforward gains, (n_u, N)
            K: Feedback gains, (N, n_u, n_x)
            
        Returns:
            (X_new, U_new, cost_new)
        """
        # Initial carry uses the passed-in x_0_arg
        init_carry = (x_0_arg, 0.0) # (x_0, cost=0.0)
        
        # Prepare inputs for scanning
        # We need (xk_old, uk_old, uk_ff, K_k, alpha)
        # Transpose to (time, dim)
        xk_old_T = X_old[:, :-1].T
        uk_old_T = U_old.T
        uk_ff_T = U_ff.T
        # K is already (N, n_u, n_x)
        # alpha needs to be broadcast to length N
        alpha_T = jnp.repeat(alpha, self.N)
        
        xs = (xk_old_T, uk_old_T, uk_ff_T, K, alpha_T)
        
        # Run the scan
        final_carry, outputs = lax.scan(
            self._forward_pass_body, init_carry, xs
        )
        
        # Unpack outputs
        final_x, final_cost = final_carry
        X_stack, U_stack = outputs
        
        # Add terminal state to X
        # X_stack is (N, n_x), U_stack is (N, n_u)
        X_new = jnp.vstack([X_stack, final_x[jnp.newaxis, :]]).T
        U_new = U_stack.T
        
        # Add terminal cost
        cost_new = final_cost + self.system.l_f_fcn(final_x)
        
        return X_new, U_new, cost_new
        

    def optimize_trajectory(self):
        """
        Runs the full iLQR optimization loop.
        """
        
        # Initial forward pass (alpha=0) to get initial trajectory
        # Pass self.x_0 as an argument
        self.X, self.U, cost = self.forward_pass(
            self.x_0, 0.0, self.X, self.U, self.U_ff, self.K
        )
        
        if self.verbose:
            print(f"Initial cost: {cost:.4f}")
        cost_prev = cost
        
        for i in range(self.maxiter):
            # Check for convergence
            if i > 0 and abs(cost - cost_prev) <= self.tol:
                if self.verbose:
                    print(f"Converged at iteration {i}")
                break
            cost_prev = cost
            
            # 1. Backward pass
            #    This call is now JIT-compiled and runs the whole scan
            self.U_ff, self.K = self.backward_pass(self.X, self.U)
            
            # 2. Line search
            alpha = 1.0
            is_step_accepted = False
            for j in range(10): # Max 10 line search steps
                
                # This call is also JIT-compiled
                # Pass self.x_0 as an argument
                X_new, U_new, cost_new = self.forward_pass(
                    self.x_0, alpha, self.X, self.U, self.U_ff, self.K
                )
                
                if cost_new < cost:
                    # Accept step
                    self.X = X_new
                    self.U = U_new
                    cost = cost_new
                    is_step_accepted = True
                    if self.verbose:
                        print(f"  Iter {i+1} (alpha={alpha:.2e}): Cost improved to {cost:.4f}")
                    break
                else:
                    # Reduce alpha
                    alpha *= self.alpha_factor
                    if alpha < self.min_alpha:
                        break # Alpha is too small
            
            if not is_step_accepted:
                if self.verbose:
                    print(f"Warning: Line search failed at iteration {i+1}. Cost did not improve.")
                break
        
        if i == self.maxiter - 1:
             if self.verbose:
                 print(f"Warning: Reached max iterations ({self.maxiter}) without converging.")
        
        return self.X, self.U, cost


    # --- These are now helper functions, not part of the main loop logic ---

    def _get_all_derivatives_for_backward_pass(self, x: jnp.ndarray, u: jnp.ndarray) -> Tuple:
        """
        Internal function to group all system calls for the backward pass
        into a single JIT-compilable unit.
        """
        # This function will be inlined into the scan body
        l_x = self.system.l_x_fcn(x, u)
        l_u = self.system.l_u_fcn(x, u)
        l_xx = self.system.l_xx_fcn(x, u)
        l_ux = self.system.l_ux_fcn(x, u)
        l_uu = self.system.l_uu_fcn(x, u)
        f_x = self.system.f_x_fcn(x, u)
        f_u = self.system.f_u_fcn(x, u)
        return l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u

    def _get_all_calcs_for_forward_pass(self, x: jnp.ndarray, u: jnp.ndarray) -> Tuple:
        """
        Internal function to group all system calls for the forward pass
        into a single JIT-compilable unit.
        """
        # This function will be inlined into the scan body
        x_next = self.system.f_fcn(x, u)
        cost = self.system.l_fcn(x, u)
        return x_next, cost