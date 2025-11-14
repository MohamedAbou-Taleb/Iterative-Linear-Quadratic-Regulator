import jax
import jax.numpy as jnp
from jax import jit, grad, jacfwd, hessian, lax
from abc import ABC, abstractmethod
from typing import Callable, Union
import numpy as np
from jax.scipy.linalg import lu_factor, lu_solve # <-- 1. Import LU functions

class System(ABC):
    """
    JAX-based abstract base class for iLQR.
    
    This version separates continuous dynamics from the integration.
    
    Subclasses MUST implement:
        - _f_cont_fcn (continuous dynamics: x_dot)
        - _l_fcn (stage cost)
        - _l_f_fcn (terminal cost)
        
    The constructor will then automatically create the discrete-time
    dynamics `self.f_fcn` using the chosen integrator, and
    JIT-compile all required derivatives.
    """
    
    def __init__(self, 
                 n_x: int, 
                 n_u: int, 
                 dt: float, 
                 use_jit: bool = True, 
                 integrator: str = 'rk4'):
        """
        Initializes the system.
        
        Args:
            n_x (int): State dimension.
            n_u (int): Control dimension.
            dt (float): Time step.
            use_jit (bool): If True, JIT-compile all functions.
            integrator (str): Integration scheme ('euler', 'midpoint', 'rk4', 'backward_euler').
        """
        self.n_x = n_x
        self.n_u = n_u
        self.dt = dt
        self.use_jit = use_jit
        # self.integrator_type = integrator # This was unused

        # --- 1. Define the discrete-time dynamics (f_fcn) ---
        #     We build the discrete-time function `_f_fcn` based on the
        #     subclass's continuous-time implementation `_f_cont_fcn`.

        # --- 0. Pre-compile continuous dynamics derivatives ---
        # This is the core optimization for Backward Euler.
        # We get the Jacobians of the continuous dynamics *once*.
        _f_cont_x = jacfwd(self._f_cont_fcn, argnums=0)
        _f_cont_u = jacfwd(self._f_cont_fcn, argnums=1)
        
        def _euler_integrator(x, u):
            """Discrete-time dynamics using Forward Euler (RK1)."""
            x_dot = self._f_cont_fcn(x, u)
            return x + x_dot * self.dt

        def _midpoint_integrator(x, u):
            """
            Discrete-time dynamics using Explicit Midpoint Method (RK2).
            Assumes Zero-Order Hold (ZOH) on control input 'u'.
            """
            k1 = self._f_cont_fcn(x, u)
            x_mid = x + (self.dt / 2.0) * k1
            k2 = self._f_cont_fcn(x_mid, u)
            return x + self.dt * k2

        def _rk4_integrator(x, u):
            """
            Discrete-time dynamics using 4th-Order Runge-Kutta (RK4).
            Assumes Zero-Order Hold (ZOH) on control input 'u'.
            """
            k1 = self._f_cont_fcn(x, u)
            k2 = self._f_cont_fcn(x + self.dt/2 * k1, u)
            k3 = self._f_cont_fcn(x + self.dt/2 * k2, u)
            k4 = self._f_cont_fcn(x + self.dt * k3, u)
            return x + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # --- Backward Euler (Implicit) Integrator ---
        # This is a more complex case as it requires solving an
        # implicit equation, which we do with a custom JVP rule.
        
        # 1. Define the residual function and its Jacobians
        #    We need to find `xkPlusOne = x_next` such that:
        #    F(xkPlusOne, x, u) = xkPlusOne - x - dt * _f_cont_fcn(xkPlusOne, u) = 0
        def _be_residual(xkPlusOne, x, u):
            """ Residual F(xkPlusOne, p) = xkPlusOne - x - dt * f_cont(xkPlusOne, u) """
            return xkPlusOne - x - self.dt * self._f_cont_fcn(xkPlusOne, u)

        # dF/dxkPlusOne (Jacobian w.r.t. the variable we solve for, xkPlusOne=x_next)
        _be_dF_dxkPlusOne_func = jax.jacobian(_be_residual, argnums=0)
        
        # dF/dx (Jacobian w.r.t. parameter x=x_prev)
        _be_dF_dx_func = jax.jacobian(_be_residual, argnums=1)
        
        # dF/du (Jacobian w.r.t. parameter u)
        _be_dF_du_func = jax.jacobian(_be_residual, argnums=2)

        @jax.custom_jvp
        def _backward_euler_integrator(x, u):
            """ 
            Primal: Solves F(xkPlusOne, x, u) = 0 for xkPlusOne=x_next.
            This function is the discrete-time dynamics f(x, u).
            """
            
            # Use Newton's method to find xkPlusOne = x_next
            def cond_fun(state):
                xkPlusOne_k, f_val, f_norm, k = state
                # f_val = _be_residual(xkPlusOne_k, x, u)
                # norm_F = jnp.linalg.norm(f_val)
                # Stop when residual is small or max iterations hit
                return (f_norm > 1e-5) & (k < 20)

            def body_fun(state):
                xkPlusOne_k, f_val, f_norm, k = state
                delta_xkPlusOne_vec = lu_solve(lu_factor_stale, -f_val)
                
                xkPlusOne_k_plus_1 = xkPlusOne_k + delta_xkPlusOne_vec

                # --- Compute residual for the *next* iteration's cond_fun ---
                f_val_new = _be_residual(xkPlusOne_k_plus_1, x, u)
                f_norm_new = jnp.linalg.norm(f_val_new)
                return (xkPlusOne_k_plus_1, f_val_new, f_norm_new, k + 1)

            # Use x (current state) as the initial guess for xkPlusOne (next state)
            # This is a good "warm start" for small dt
            # xkPlusOne_guess = x 
            xkPlusOne_guess = x + self.dt * self._f_cont_fcn(x, u)  # Better guess with Euler step
            f_val_guess = _be_residual(xkPlusOne_guess, x, u)
            f_norm_guess = jnp.linalg.norm(f_val_guess)

                        # --- OPTIMIZATION (Quasi-Newton): ---
            # Calculate the Jacobian *once* using the initial guess.
            J_cont_x_guess = _f_cont_x(xkPlusOne_guess, u)
            j_xkPlusOne_val_stale = jnp.eye(self.n_x) - self.dt * J_cont_x_guess
            # Add regularization for numerical stability
            j_xkPlusOne_stable = j_xkPlusOne_val_stale # + jnp.eye(self.n_x) * 1e-6
            lu_factor_stale = lu_factor(j_xkPlusOne_stable) # <-- 2. Compute LU factor
            initial_state = (xkPlusOne_guess, f_val_guess, f_norm_guess, 0)
            xkPlusOne_solution, _, _, _ = lax.while_loop(cond_fun, body_fun, initial_state)
            return xkPlusOne_solution

        @_backward_euler_integrator.defjvp
        def _backward_euler_jvp(primals, tangents):
            """ 
            JVP rule for the implicit solver.
            This implements the Implicit Function Theorem.
            """
            x, u = primals
            j_x, j_u = tangents # tangents dx/dt, du/dt

            # 1. Run the primal to get the solution xkPlusOne=x_next
            xkPlusOne = _backward_euler_integrator(x, u)

            # # 2. Get Jacobians at the solution (xkPlusOne, x, u)
            # J_xkPlusOne = _be_dF_dxkPlusOne_func(xkPlusOne, x, u) # dF/dxkPlusOne
            # J_x = _be_dF_dx_func(xkPlusOne, x, u) # dF/dx
            # J_u = _be_dF_du_func(xkPlusOne, x, u) # dF/du

            # 2. --- OPTIMIZATION ---
            #    Get Jacobians at the solution (xkPlusOne, x, u)
            #    by analytically constructing them.
            J_cont_x_sol = _f_cont_x(xkPlusOne, u)
            J_cont_u_sol = _f_cont_u(xkPlusOne, u)
            
            # dF/dxkPlusOne = I - dt * df_cont/dxkPlusOne
            J_xkPlusOne = jnp.eye(self.n_x) - self.dt * J_cont_x_sol
            
            # dF/dx = -I
            J_x = -jnp.eye(self.n_x)
            
            # dF/du = -dt * df_cont/du
            J_u = -self.dt * J_cont_u_sol
            
            # 3. Solve the forward-mode linear system for j_xkPlusOne (tangent of x_next)
            #    J_xkPlusOne @ j_xkPlusOne = - (J_x @ j_x + J_u @ j_u)
            
            # Compute RHS:
            rhs = - (J_x @ j_x + J_u @ j_u)
            
            # Solve for j_xkPlusOne
            # J_xkPlusOne_stable = J_xkPlusOne + jnp.eye(self.n_x) * 1e-6
            J_xkPlusOne_stable = J_xkPlusOne
            j_xkPlusOne = jnp.linalg.solve(J_xkPlusOne_stable, rhs)
            
            return xkPlusOne, j_xkPlusOne
        
        # --- End of Backward Euler ---


        # Select the chosen integrator
        if integrator == 'rk4':
            self._f_fcn = _rk4_integrator
        elif integrator == 'midpoint':
            self._f_fcn = _midpoint_integrator
        elif integrator == 'euler':
            self._f_fcn = _euler_integrator
        elif integrator == 'backward_euler':
            self._f_fcn = _backward_euler_integrator
        else:
            raise ValueError(
                f"Unknown integrator: '{integrator}'. "
                "Supported: 'rk4', 'midpoint', 'euler', 'backward_euler'."
            )

        # --- 2. Define all derivative functions ---
        #     These JAX transformations are now applied to the *generated*
        #     `self._f_fcn`. If 'backward_euler' is chosen, JAX will
        #     automatically use our custom JVP rule to compute f_x and f_u.
        
        _f_x = jacfwd(self._f_fcn, argnums=0)
        _f_u = jacfwd(self._f_fcn, argnums=1)
        
        _l_x = grad(self._l_fcn, argnums=0)
        _l_u = grad(self._l_fcn, argnums=1)
        _l_xx = hessian(self._l_fcn, argnums=0)
        _l_uu = hessian(self._l_fcn, argnums=1)
        _l_ux = jacfwd(grad(self._l_fcn, argnums=1), argnums=0)
        
        _l_f_x = grad(self._l_f_fcn, argnums=0)
        _l_f_xx = hessian(self._l_f_fcn, argnums=0)

        # --- 3. Create public-facing methods (conditionally JIT) ---
        
        if self.use_jit:
            self.f_fcn: Callable = jit(self._f_fcn)
            self.f_x_fcn: Callable = jit(_f_x)
            self.f_u_fcn: Callable = jit(_f_u)
            
            self.l_fcn: Callable = jit(self._l_fcn)
            self.l_x_fcn: Callable = jit(_l_x)
            self.l_u_fcn: Callable = jit(_l_u)
            self.l_xx_fcn: Callable = jit(_l_xx)
            self.l_uu_fcn: Callable = jit(_l_uu)
            self.l_ux_fcn: Callable = jit(_l_ux)
            
            self.l_f_fcn: Callable = jit(self._l_f_fcn)
            self.l_f_x_fcn: Callable = jit(_l_f_x)
            self.l_f_xx_fcn: Callable = jit(_l_f_xx)
        else:
            # Assign the raw, un-compiled functions
            self.f_fcn: Callable = self._f_fcn
            self.f_x_fcn: Callable = _f_x
            self.f_u_fcn: Callable = _f_u
            self.l_fcn: Callable = self._l_fcn
            self.l_x_fcn: Callable = _l_x
            self.l_u_fcn: Callable = _l_u
            self.l_xx_fcn: Callable = _l_xx
            self.l_uu_fcn: Callable = _l_uu
            self.l_ux_fcn: Callable = _l_ux
            self.l_f_fcn: Callable = self._l_f_fcn
            self.l_f_x_fcn: Callable = _l_f_x
            self.l_f_xx_fcn: Callable = _l_f_xx

    # --- Abstract Methods (to be implemented by subclass) ---
    
    @abstractmethod
    def _f_cont_fcn(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """
        [Subclass implementation] Continuous-time dynamics.
        Should return the state derivative, x_dot = f(x, u).
        """
        pass

    @abstractmethod
    def _l_fcn(self, x: jnp.ndarray, u: jnp.ndarray) -> float:
        """
        [Subclass implementation] Stage cost function: l(x, u).
        """
        pass

    @abstractmethod
    def _l_f_fcn(self, x: jnp.ndarray) -> float:
        """
        [Subclass implementation] Terminal cost function: l_f(x).
        """
        pass