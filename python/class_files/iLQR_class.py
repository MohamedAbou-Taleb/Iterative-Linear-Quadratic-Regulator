import jax
import jax.numpy as jnp
from jax import lax
from typing import List, Tuple, Callable
import numpy as np

# Import the base class
from .systems.system_base import System


class iLQR:
    """
    Python/JAX implementation of the iLQR algorithm.
    """

    def __init__(
        self,
        system: System,
        T: float,
        x_0: jnp.ndarray,
        U_init: jnp.ndarray,
        tol: float = 1e-5,
        maxiter: int = 100,
        alpha_factor: float = 0.5,
        min_alpha: float = 1e-8,
        verbose: bool = True,
        ctrl_dt: float = None,  # <--- NEW ARGUMENT
        return_percussion: bool = False,
    ):

        self.system = system
        self.T = T
        self.x_0 = jnp.asarray(x_0)

        # Solver settings
        self.tol = tol
        self.maxiter = maxiter
        self.alpha_factor = alpha_factor
        self.min_alpha = min_alpha
        self.verbose = verbose

        # Get dims from system
        self.n_x = system.n_x
        self.n_u = system.n_u

        # --- NEW LOGIC FOR TIME STEPPING ---
        # self.sim_dt is the integration step size (fixed by system)
        self.sim_dt = system.dt
        
        # self.dt is now the CONTROL step size
        if ctrl_dt is None:
            self.dt = self.sim_dt
            self.sim_steps = 1
        else:
            self.dt = ctrl_dt
            # Calculate integer number of integration steps per control step
            # Using round to avoid floating point errors (e.g., 0.01 / 0.001 = 9.9999)
            ratio = self.dt / self.sim_dt
            self.sim_steps = int(round(ratio))
            
            # Sanity check to ensure time steps align
            if abs(ratio - self.sim_steps) > 1e-6:
                raise ValueError(
                    f"Control dt ({self.dt}) must be an integer multiple "
                    f"of system simulation dt ({self.sim_dt})"
                )
        # -----------------------------------

        # Time horizon (Control grid)
        self.tspan = jnp.arange(0, T + self.dt, self.dt)
        self.N = len(self.tspan) - 1

        # Check U_init shape
        expected_shape = (self.n_u, self.N)
        if U_init.shape != expected_shape:
            raise ValueError(
                f"U_init must have shape {expected_shape}, but got {U_init.shape}"
            )

        # Trajectories
        self.X = jnp.zeros((self.n_x, self.N + 1))
        self.U = jnp.asarray(U_init)

        # Gains
        self.K = jnp.zeros((self.N, self.n_u, self.n_x))
        self.U_ff = jnp.zeros((self.n_u, self.N))

        # =====================================================================
        # --- PERFORMANCE OPTIMIZATION ---
        self.backward_pass = jax.jit(self._backward_pass_scan)
        self.forward_pass = jax.jit(self._forward_pass_scan)
        # =====================================================================

    def _integrate_dynamics(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """
        Integrates the system dynamics for one CONTROL step.
        If sim_steps > 1, this loops the system dynamics sim_steps times.
        """
        
        # Define the loop body: x_{i+1} = f(x_i, u)
        # u is constant across the sub-steps (Zero-Order Hold)
        def body(i, val):
            return self.system.f_fcn(val, u)

        # Run the loop
        # If sim_steps == 1, this executes exactly once (same as before)
        # But for JAX efficiency, we use fori_loop
        x_next = lax.fori_loop(0, self.sim_steps, body, x)
        
        return x_next

    def _backward_pass_body(self, carry: Tuple, inputs: Tuple):
        V_x, V_xx = carry
        x, u = inputs

        # --- 1. Get All Derivatives ---
        (l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u) = (
            self._get_all_derivatives_for_backward_pass(x, u)
        )

        # --- 2. Calculate Q-Function Derivatives ---
        Q_x = l_x + f_x.T @ V_x
        Q_u = l_u + f_u.T @ V_x
        Q_xx = l_xx + f_x.T @ V_xx @ f_x
        Q_ux = l_ux + f_u.T @ V_xx @ f_x
        Q_uu = l_uu + f_u.T @ V_xx @ f_u

        # --- 3. Solve for Gains ---
        # Regularization can be added to Q_uu here if needed
        K_k = -jnp.linalg.solve(Q_uu, Q_ux)
        u_ff_k = -jnp.linalg.solve(Q_uu, Q_u)

        # --- 4. Update Value Function Derivatives ---
        V_x_prev = Q_x + K_k.T @ Q_u
        V_xx_prev = Q_xx + Q_ux.T @ K_k

        new_carry = (V_x_prev, V_xx_prev)
        outputs = (u_ff_k, K_k)

        return new_carry, outputs

    def _backward_pass_scan(self, X_nom: jnp.ndarray, U_nom: jnp.ndarray):
        x_N = X_nom[:, -1]
        V_x = self.system.l_f_x_fcn(x_N)
        V_xx = self.system.l_f_xx_fcn(x_N)

        init_carry = (V_x, V_xx)
        xs = (X_nom[:, :-1].T, U_nom.T)

        final_carry, outputs = lax.scan(
            self._backward_pass_body, init_carry, xs, reverse=True
        )

        u_ff_stack, K_stack = outputs
        U_ff = u_ff_stack.T
        K = K_stack

        return U_ff, K

    def _forward_pass_body(self, carry: Tuple, inputs: Tuple):
        xk_new, cost_new = carry
        xk_old, uk_old, uk_ff, K_k, alpha = inputs

        # Calculate new control
        delta_x = xk_new - xk_old
        uk_new = uk_old + alpha * uk_ff + K_k @ delta_x

        # Simulate one step and get stage cost
        xkPlusOne, cost_k = self._get_all_calcs_for_forward_pass(xk_new, uk_new)

        new_carry = (xkPlusOne, cost_new + cost_k)
        outputs = (xk_new, uk_new)

        return new_carry, outputs

    def _forward_pass_scan(
        self,
        x_0_arg: jnp.ndarray,
        alpha: float,
        X_old: jnp.ndarray,
        U_old: jnp.ndarray,
        U_ff: jnp.ndarray,
        K: jnp.ndarray,
    ):
        init_carry = (x_0_arg, 0.0)
        
        xk_old_T = X_old[:, :-1].T
        uk_old_T = U_old.T
        uk_ff_T = U_ff.T
        alpha_T = jnp.repeat(alpha, self.N)

        xs = (xk_old_T, uk_old_T, uk_ff_T, K, alpha_T)

        final_carry, outputs = lax.scan(self._forward_pass_body, init_carry, xs)

        final_x, final_cost = final_carry
        X_stack, U_stack = outputs

        X_new = jnp.vstack([X_stack, final_x[jnp.newaxis, :]]).T
        U_new = U_stack.T
        cost_new = final_cost + self.system.l_f_fcn(final_x)

        return X_new, U_new, cost_new

    def optimize_trajectory(self):
        # Initial forward pass
        self.X, self.U, cost = self.forward_pass(
            self.x_0, 0.0, self.X, self.U, self.U_ff, self.K
        )

        if self.verbose:
            print(f"Initial cost: {cost:.4f}")
        cost_prev = cost

        for i in range(self.maxiter):
            # Check convergence
            if i > 0 and abs(cost - cost_prev) <= self.tol:
                if self.verbose:
                    print(f"Converged at iteration {i}")
                break
            cost_prev = cost

            # 1. Backward pass
            self.U_ff, self.K = self.backward_pass(self.X, self.U)

            # 2. Line search
            alpha = 1.0
            is_step_accepted = False
            for j in range(10):
                X_new, U_new, cost_new = self.forward_pass(
                    self.x_0, alpha, self.X, self.U, self.U_ff, self.K
                )

                if cost_new <= cost:
                    self.X = X_new
                    self.U = U_new
                    cost = cost_new
                    is_step_accepted = True
                    if self.verbose:
                        print(
                            f"  Iter {i+1} (alpha={alpha:.2e}): Cost improved to {cost:.4f}"
                        )
                    break
                else:
                    alpha *= self.alpha_factor
                    if alpha < self.min_alpha:
                        break

            if not is_step_accepted:
                if self.verbose:
                    print(f"Warning: Line search failed at iteration {i+1}.")
                break
        
        if i == self.maxiter - 1 and self.verbose:
            print(f"Warning: Reached max iterations ({self.maxiter}).")

        return self.X, self.U, cost

    # --- HELPER FUNCTIONS ---

    def _get_all_derivatives_for_backward_pass(
        self, x: jnp.ndarray, u: jnp.ndarray
    ) -> Tuple:
        """
        Gets derivatives. Handles Multi-Step Integration if needed.
        """
        l_x = self.system.l_x_fcn(x, u)
        l_u = self.system.l_u_fcn(x, u)
        l_xx = self.system.l_xx_fcn(x, u)
        l_ux = self.system.l_ux_fcn(x, u)
        l_uu = self.system.l_uu_fcn(x, u)

        # --- DYNAMICS DERIVATIVES ---
        if self.sim_steps == 1:
            # OPTIMIZATION: If steps=1, use the system's provided methods directly.
            # This is faster if the system has analytical derivatives.
            f_x = self.system.f_x_fcn(x, u)
            f_u = self.system.f_u_fcn(x, u)
        else:
            # If sub-stepping is active, we must differentiate through the
            # loop to get the correct Jacobian A and B matrices for the
            # full control step.
            f_x = jax.jacfwd(self._integrate_dynamics, argnums=0)(x, u)
            f_u = jax.jacfwd(self._integrate_dynamics, argnums=1)(x, u)

        return l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u

    def _get_all_calcs_for_forward_pass(self, x: jnp.ndarray, u: jnp.ndarray) -> Tuple:
        """
        Calculates next state and cost. Handles Multi-Step Integration.
        """
        # Call the multi-step integrator
        x_next = self._integrate_dynamics(x, u)
        
        # Calculate cost (usually defined at the knot point)
        cost = self.system.l_fcn(x, u)
        
        return x_next, cost
    

# ===========================
import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple

# Import your original base class
# Adjust the import path if you save this in the same file or a different folder
try:
    from class_files.iLQR_class import iLQR
except ImportError:
    from iLQR_class import iLQR

class iLQR_Parametric(iLQR):
    """
    Extension of iLQR that supports passing dynamic parameters 
    (like changing targets) to the cost functions without re-compiling.
    
    This class overrides the optimization loops to thread a 'params' 
    argument through to the system's cost functions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Re-JIT compile the scan functions to use the new parametric methods.
        # This ensures we use the overridden _backward_pass_scan and _forward_pass_scan.
        self.backward_pass = jax.jit(self._backward_pass_scan)
        self.forward_pass = jax.jit(self._forward_pass_scan)

    def optimize_trajectory(self, params=None):
        """
        Optimizes the trajectory with optional dynamic parameters.
        
        Args:
            params: A tuple (or any Pytree) of parameters to pass to the 
                    cost functions (e.g., (x_target, u_target)). 
                    If None, passes an empty tuple.
        """
        # Ensure params is a valid JAX type (empty tuple if None)
        if params is None:
            params = ()

        # Initial forward pass
        self.X, self.U, cost = self.forward_pass(
            self.x_0, 0.0, self.X, self.U, self.U_ff, self.K, params
        )

        if self.verbose:
            print(f"Initial cost: {cost:.4f}")
        cost_prev = cost

        for i in range(self.maxiter):
            # Check convergence
            if i > 0 and abs(cost - cost_prev) <= self.tol:
                if self.verbose:
                    print(f"Converged at iteration {i}")
                break
            cost_prev = cost

            # 1. Backward pass
            self.U_ff, self.K = self.backward_pass(self.X, self.U, params)

            # 2. Line search
            alpha = 1.0
            is_step_accepted = False
            for j in range(10):
                X_new, U_new, cost_new = self.forward_pass(
                    self.x_0, alpha, self.X, self.U, self.U_ff, self.K, params
                )

                if cost_new <= cost:
                    self.X = X_new
                    self.U = U_new
                    cost = cost_new
                    is_step_accepted = True
                    if self.verbose:
                        print(
                            f"  Iter {i+1} (alpha={alpha:.2e}): Cost improved to {cost:.4f}"
                        )
                    break
                else:
                    alpha *= self.alpha_factor
                    if alpha < self.min_alpha:
                        break

            if not is_step_accepted:
                if self.verbose:
                    print(f"Warning: Line search failed at iteration {i+1}.")
                break
        
        if i == self.maxiter - 1 and self.verbose:
            print(f"Warning: Reached max iterations ({self.maxiter}).")

        return self.X, self.U, cost

    # =========================================================================
    # OVERRIDDEN HELPER FUNCTIONS
    # =========================================================================

    def _get_all_derivatives_for_backward_pass(
        self, x: jnp.ndarray, u: jnp.ndarray, params: Tuple
    ) -> Tuple:
        """
        Gets derivatives passing 'params' to the cost functions.
        """
        # Pass params to cost derivatives
        l_x = self.system.l_x_fcn(x, u, params)
        l_u = self.system.l_u_fcn(x, u, params)
        l_xx = self.system.l_xx_fcn(x, u, params)
        l_ux = self.system.l_ux_fcn(x, u, params)
        l_uu = self.system.l_uu_fcn(x, u, params)

        # Dynamics derivatives usually don't depend on cost params,
        # but if your dynamics DID depend on params, you would pass them here.
        if self.sim_steps == 1:
            f_x = self.system.f_x_fcn(x, u)
            f_u = self.system.f_u_fcn(x, u)
        else:
            f_x = jax.jacfwd(self._integrate_dynamics, argnums=0)(x, u)
            f_u = jax.jacfwd(self._integrate_dynamics, argnums=1)(x, u)

        return l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u

    def _backward_pass_body(self, carry: Tuple, inputs: Tuple):
        V_x, V_xx, params = carry  # Unpack params from carry
        x, u = inputs

        # Get derivatives (passing params)
        (l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u) = (
            self._get_all_derivatives_for_backward_pass(x, u, params)
        )

        # Calculate Q-Function Derivatives
        Q_x = l_x + f_x.T @ V_x
        Q_u = l_u + f_u.T @ V_x
        Q_xx = l_xx + f_x.T @ V_xx @ f_x
        Q_ux = l_ux + f_u.T @ V_xx @ f_x
        Q_uu = l_uu + f_u.T @ V_xx @ f_u

        # Solve for Gains
        K_k = -jnp.linalg.solve(Q_uu, Q_ux)
        u_ff_k = -jnp.linalg.solve(Q_uu, Q_u)

        # Update Value Function Derivatives
        V_x_prev = Q_x + K_k.T @ Q_u
        V_xx_prev = Q_xx + Q_ux.T @ K_k

        # Repack params into carry
        new_carry = (V_x_prev, V_xx_prev, params)
        outputs = (u_ff_k, K_k)

        return new_carry, outputs

    def _backward_pass_scan(self, X_nom: jnp.ndarray, U_nom: jnp.ndarray, params: Tuple):
        x_N = X_nom[:, -1]
        
        # Pass params to final cost derivatives
        V_x = self.system.l_f_x_fcn(x_N, params)
        V_xx = self.system.l_f_xx_fcn(x_N, params)

        # Initialize carry with params
        init_carry = (V_x, V_xx, params)
        xs = (X_nom[:, :-1].T, U_nom.T)

        final_carry, outputs = lax.scan(
            self._backward_pass_body, init_carry, xs, reverse=True
        )

        u_ff_stack, K_stack = outputs
        U_ff = u_ff_stack.T
        K = K_stack

        return U_ff, K

    def _forward_pass_body(self, carry: Tuple, inputs: Tuple):
        xk_new, cost_new, params = carry  # Unpack params
        xk_old, uk_old, uk_ff, K_k, alpha = inputs

        # Calculate new control
        delta_x = xk_new - xk_old
        uk_new = uk_old + alpha * uk_ff + K_k @ delta_x

        # Integrate dynamics (no params needed usually)
        x_next = self._integrate_dynamics(xk_new, uk_new)
        
        # Calculate stage cost WITH params
        cost_k = self.system.l_fcn(xk_new, uk_new, params)

        # Repack params
        new_carry = (x_next, cost_new + cost_k, params)
        outputs = (xk_new, uk_new)

        return new_carry, outputs

    def _forward_pass_scan(
        self,
        x_0_arg: jnp.ndarray,
        alpha: float,
        X_old: jnp.ndarray,
        U_old: jnp.ndarray,
        U_ff: jnp.ndarray,
        K: jnp.ndarray,
        params: Tuple,
    ):
        # Initialize carry with params
        init_carry = (x_0_arg, 0.0, params)
        
        xk_old_T = X_old[:, :-1].T
        uk_old_T = U_old.T
        uk_ff_T = U_ff.T
        alpha_T = jnp.repeat(alpha, self.N)

        xs = (xk_old_T, uk_old_T, uk_ff_T, K, alpha_T)

        final_carry, outputs = lax.scan(self._forward_pass_body, init_carry, xs)

        final_x, final_cost, _ = final_carry # Unpack params but ignore it here
        X_stack, U_stack = outputs

        X_new = jnp.vstack([X_stack, final_x[jnp.newaxis, :]]).T
        U_new = U_stack.T
        
        # Calculate final cost WITH params
        cost_new = final_cost + self.system.l_f_fcn(final_x, params)

        return X_new, U_new, cost_new