import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple

# Import the base class
from .systems.system_base import System


class MultipleShootingiLQR:
    """
    Multiple Shooting iLQR implementation (Gauss-Newton with Defects).
    
    Distinct features:
    1. Allows initialization of State Trajectory (X).
    2. Backward pass accounts for dynamic defects (gaps) f(x,u) - x_next.
    3. Forward pass closes the gaps (Feasibility-driven).
    """

    def __init__(
        self,
        system: System,
        T: float,
        x_0: jnp.ndarray,
        U_init: jnp.ndarray,
        X_init: jnp.ndarray = None,  # New: Allow state initialization
        tol: float = 1e-5,
        maxiter: int = 100,
        alpha_factor: float = 0.5,
        min_alpha: float = 1e-8,
        verbose: bool = True,
    ):
        self.system = system
        self.T = T
        self.x_0 = jnp.asarray(x_0)

        self.tol = tol
        self.maxiter = maxiter
        self.alpha_factor = alpha_factor
        self.min_alpha = min_alpha
        self.verbose = verbose

        self.n_x = system.n_x
        self.n_u = system.n_u
        self.dt = system.dt

        self.tspan = jnp.arange(0, T + self.dt, self.dt)
        self.N = len(self.tspan) - 1

        # Check U_init shape
        expected_shape = (self.n_u, self.N)
        if U_init.shape != expected_shape:
            raise ValueError(f"U_init shape mismatch. Expected {expected_shape}, got {U_init.shape}")

        self.U = jnp.asarray(U_init)

        # Handle X initialization
        if X_init is not None:
            # Check shape
            x_expected = (self.n_x, self.N + 1)
            if X_init.shape != x_expected:
                raise ValueError(f"X_init shape mismatch. Expected {x_expected}, got {X_init.shape}")
            self.X = jnp.asarray(X_init)
            self.has_initial_state_guess = True
        else:
            self.X = jnp.zeros((self.n_x, self.N + 1))
            self.has_initial_state_guess = False

        # Gains
        self.K = jnp.zeros((self.N, self.n_u, self.n_x))
        self.U_ff = jnp.zeros((self.n_u, self.N))

        # --- JIT Compilation ---
        self.backward_pass = jax.jit(self._backward_pass_scan)
        self.forward_pass = jax.jit(self._forward_pass_scan)

    def _backward_pass_body(self, carry: Tuple, inputs: Tuple):
        """
        Backward pass step accounting for defects.
        inputs: (x_k, u_k, x_k_plus_1_nominal)
        """
        V_x, V_xx = carry
        x, u, x_next_nom = inputs  # We now take x_next from the nominal trajectory

        # 1. Get Derivatives AND Dynamics evaluation
        # We need f(x,u) specifically to calculate the gap
        f_val = self.system.f_fcn(x, u)
        
        (l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u) = (
            self._get_all_derivatives_for_backward_pass(x, u)
        )

        # 2. Calculate Defect (Gap)
        # gap = predicted_state - stored_state
        defect = f_val - x_next_nom

        # 3. Calculate Q-Function Derivatives
        # NOTE: In Multiple Shooting, we linearize around the nominal (infeasible) trajectory.
        # However, V_x and V_xx are the Value function derivatives at x_next_nom.
        # But the dynamics land us at f(x,u). 
        # By Taylor expansion, V_x at f(x,u) approx = V_x_nom + V_xx_nom * defect
        
        V_x_corrected = V_x + V_xx @ defect

        Q_x = l_x + f_x.T @ V_x_corrected
        Q_u = l_u + f_u.T @ V_x_corrected
        
        Q_xx = l_xx + f_x.T @ V_xx @ f_x
        Q_ux = l_ux + f_u.T @ V_xx @ f_x
        Q_uu = l_uu + f_u.T @ V_xx @ f_u

        # 4. Solve for Gains
        # Regularization can be added to Q_uu here if needed for stability
        # Q_uu += jnp.eye(self.n_u)*1e-4
        K_k = -jnp.linalg.solve(Q_uu, Q_ux)
        u_ff_k = -jnp.linalg.solve(Q_uu, Q_u)

        # 5. Update Value Function Derivatives for previous step
        V_x_prev = Q_x + K_k.T @ Q_u
        V_xx_prev = Q_xx + Q_ux.T @ K_k

        # Symmeterize V_xx to avoid numerical drift
        V_xx_prev = 0.5 * (V_xx_prev + V_xx_prev.T)

        new_carry = (V_x_prev, V_xx_prev)
        outputs = (u_ff_k, K_k)

        return new_carry, outputs

    def _backward_pass_scan(self, X_nom: jnp.ndarray, U_nom: jnp.ndarray):
        """
        Scans backwards. Now includes x_next in the input tuple to calculate defects.
        """
        x_N = X_nom[:, -1]
        V_x = self.system.l_f_x_fcn(x_N)
        V_xx = self.system.l_f_xx_fcn(x_N)

        init_carry = (V_x, V_xx)

        # Prepare inputs: (x_k, u_k, x_{k+1})
        # x_k:   indices 0 to N-1
        # u_k:   indices 0 to N-1
        # x_k+1: indices 1 to N
        xs = (X_nom[:, :-1].T, U_nom.T, X_nom[:, 1:].T)

        final_carry, outputs = lax.scan(
            self._backward_pass_body, init_carry, xs, reverse=True
        )

        u_ff_stack, K_stack = outputs
        return u_ff_stack.T, K_stack

    def _forward_pass_body(self, carry: Tuple, inputs: Tuple):
        """
        Standard rollout (Simulated Single Shooting).
        This forces the trajectory to become feasible in the forward pass.
        """
        xk_new, cost_new = carry
        xk_old, uk_old, uk_ff, K_k, alpha = inputs

        # Calculate new control
        # Note: xk_old is the (possibly infeasible) state from the previous iteration
        delta_x = xk_new - xk_old
        uk_new = uk_old + alpha * uk_ff + K_k @ delta_x

        # Simulate dynamics (Closing the gap)
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
        """
        Main Optimization Loop.
        """
        cost = 0.0
        
        # 1. Initialization Strategy
        if self.has_initial_state_guess:
            # If we have an X guess, we skip the initial rollout.
            # We assume the user provided a "warm start" X and U.
            # We calculate the cost of this trajectory (even if infeasible) for logging.
            # Note: A proper MS cost would include penalty for defects, but for 
            # simple comparison, we usually just track the running cost.
            if self.verbose:
                print("Starting with provided State Guess (Warm Start).")
            
            # We need a baseline cost. We can run a forward pass with alpha=0 
            # effectively re-simulating to get the TRUE cost of the implied controls, 
            # OR we can just start optimizing. 
            # For MS-iLQR, usually we start immediately with Backward Pass.
            cost = float("inf") 
        else:
            # Standard Single Shooting Init: Rollout U_init to get X
            if self.verbose:
                print("No State Guess provided. Performing initial rollout.")
            self.X, self.U, cost = self.forward_pass(
                self.x_0, 0.0, self.X, self.U, self.U_ff, self.K
            )

        if self.verbose:
            print(f"Initial Cost: {cost:.4f}")

        cost_prev = cost

        for i in range(self.maxiter):
            # --- 1. Backward Pass (includes Defects) ---
            # Even if self.X is infeasible, this runs and computes gradients
            # that try to fix the cost AND the defects.
            self.U_ff, self.K = self.backward_pass(self.X, self.U)

            # --- 2. Line Search (Rollout closes gaps) ---
            alpha = 1.0
            is_step_accepted = False
            
            for j in range(10):
                # Forward pass always enforces feasibility (x' = f(x,u))
                X_new, U_new, cost_new = self.forward_pass(
                    self.x_0, alpha, self.X, self.U, self.U_ff, self.K
                )

                if cost_new <= cost:
                    self.X = X_new
                    self.U = U_new
                    cost = cost_new
                    is_step_accepted = True
                    if self.verbose:
                        print(f"  Iter {i+1} (alpha={alpha:.2e}): Cost {cost:.4f}")
                    break
                else:
                    alpha *= self.alpha_factor
                    if alpha < self.min_alpha:
                        break

            # Convergence Check
            if abs(cost_prev - cost) < self.tol and is_step_accepted:
                if self.verbose:
                    print(f"Converged at iteration {i+1}")
                break
            
            if not is_step_accepted:
                if self.verbose:
                    print(f"Line search failed at iter {i+1}")
                break
            
            cost_prev = cost

        return self.X, self.U, cost

    # --- Helpers ---
    def _get_all_derivatives_for_backward_pass(self, x, u):
        l_x = self.system.l_x_fcn(x, u)
        l_u = self.system.l_u_fcn(x, u)
        l_xx = self.system.l_xx_fcn(x, u)
        l_ux = self.system.l_ux_fcn(x, u)
        l_uu = self.system.l_uu_fcn(x, u)
        f_x = self.system.f_x_fcn(x, u)
        f_u = self.system.f_u_fcn(x, u)
        return l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u

    def _get_all_calcs_for_forward_pass(self, x, u):
        x_next = self.system.f_fcn(x, u)
        cost = self.system.l_fcn(x, u)
        return x_next, cost