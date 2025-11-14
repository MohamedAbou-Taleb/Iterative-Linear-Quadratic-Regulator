import jax
import jax.numpy as jnp
from jax import jit, grad, jacfwd, hessian, lax
from jax.scipy.linalg import lu_factor, lu_solve
from abc import ABC, abstractmethod
from typing import Callable, Union
import numpy as np

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

        # --- 1. Define the discrete-time dynamics (f_fcn) ---
        #    We build the discrete-time function `_f_fcn` based on the
        #    subclass's continuous-time implementation `_f_cont_fcn`.
        
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

        # Select the chosen integrator
        if integrator == 'rk4':
            self._f_fcn = _rk4_integrator
        elif integrator == 'midpoint':
            self._f_fcn = _midpoint_integrator
        elif integrator == 'euler':
            self._f_fcn = _euler_integrator
        elif integrator == 'backward_euler':
            # --- Backward Euler Implementation (Optimized) ---
            
            # --- This is now a standalone solver, NOT a custom_jvp function ---
            # We have removed the @jax.custom_jvp decorator
            def _backward_euler_integrator(x, u):
                """
                Solves for x_k+1 using an optimized Quasi-Newton method.
                x_k+1 = x_k + dt * f_cont(x_k+1, u_k)
                """
                
                # --- Get continuous-time dynamics jacobians (pre-compiled) ---
                _f_cont_x = self._f_cont_x_fcn # (JIT-compiled jacfwd)
                _f_cont_u = self._f_cont_u_fcn # (JIT-compiled jacfwd)
                
                # --- Define the implicit residual function ---
                # F(xkPlusOne, x, u) = xkPlusOne - x - dt * f_cont(xkPlusOne, u) = 0
                @jit
                def _be_residual(xkPlusOne, x, u):
                    return xkPlusOne - x - self.dt * self._f_cont_fcn(xkPlusOne, u)

                # --- Newton Solver (Optimized) ---
                def cond_fun(state):
                    xkPlusOne_k, f_val, f_norm, k = state
                    return (f_norm > 1e-5) & (k < 20) # <-- CHANGED TOLERANCE

                def body_fun(state):
                    xkPlusOne_k, f_val, f_norm, k = state
                    
                    # Solve J_stale * delta = -F using pre-computed LU
                    delta_xkPlusOne_vec = lu_solve(lu_factor_stale, -f_val)
                    
                    xkPlusOne_k_plus_1 = xkPlusOne_k + delta_xkPlusOne_vec
                    
                    f_val_new = _be_residual(xkPlusOne_k_plus_1, x, u)
                    f_norm_new = jnp.linalg.norm(f_val_new)
                    
                    return (xkPlusOne_k_plus_1, f_val_new, f_norm_new, k + 1)

                # Use x (current state) as the initial guess if none provided
                
                xkPlusOne_guess = x + self.dt * self._f_cont_fcn(x, u)
                
                f_val_guess = _be_residual(xkPlusOne_guess, x, u)
                f_norm_guess = jnp.linalg.norm(f_val_guess)
                
                # --- Quasi-Newton: Calculate Jacobian *once* ---
                J_cont_x_guess = _f_cont_x(xkPlusOne_guess, u)
                j_xkPlusOne_val_stale = jnp.eye(self.n_x) - self.dt * J_cont_x_guess
                j_xkPlusOne_stable = j_xkPlusOne_val_stale # + jnp.eye(self.n_x) * 1e-5 # <-- CHANGED TOLERANCE (in comment)
                
                # --- Pre-compute LU factorization ---
                lu_factor_stale = lu_factor(j_xkPlusOne_stable)
                
                initial_state = (xkPlusOne_guess, f_val_guess, f_norm_guess, 0)
                
                xkPlusOne_solution, _, _, _ = lax.while_loop(cond_fun, body_fun, initial_state)
                return xkPlusOne_solution
            
            # --- NEW: Analytical Jacobian Functions ---
            # These functions compute the Jacobians f_x and f_u directly
            # using the Implicit Function Theorem, avoiding jacfwd(custom_jvp).
            
            def _be_f_x_fcn(x, u):
                """Computes df/dx for the backward euler integrator."""
                # 1. Find the solution x_k+1
                xkPlusOne_solution = _backward_euler_integrator(x, u)
                
                # 2. Get true Jacobians at the solution
                _f_cont_x = self._f_cont_x_fcn
                J_cont_x = _f_cont_x(xkPlusOne_solution, u)

                # 3. Define the residual Jacobians
                # F = xkPlusOne - x - dt * f_cont(xkPlusOne, u)
                # J_xkPlusOne = dF/dxkPlusOne = I - dt * df_cont/dxkPlusOne
                # J_x = dF/dx = -I
                J_xkPlusOne_val = jnp.eye(self.n_x) - self.dt * J_cont_x
                J_x_val = -jnp.eye(self.n_x)

                # 4. Solve the IFT system: J_xkPlusOne @ f_x = -J_x
                # This gives f_x = d(xkPlusOne)/dx
                f_x = jnp.linalg.solve(J_xkPlusOne_val, -J_x_val)
                return f_x

            def _be_f_u_fcn(x, u):
                """Computes df/du for the backward euler integrator."""
                # 1. Find the solution x_k+1
                xkPlusOne_solution = _backward_euler_integrator(x, u)
                
                # 2. Get true Jacobians at the solution
                _f_cont_x = self._f_cont_x_fcn
                _f_cont_u = self._f_cont_u_fcn
                J_cont_x = _f_cont_x(xkPlusOne_solution, u)
                J_cont_u = _f_cont_u(xkPlusOne_solution, u)

                # 3. Define the residual Jacobians
                # F = xkPlusOne - x - dt * f_cont(xkPlusOne, u)
                # J_xkPlusOne = dF/dxkPlusOne = I - dt * df_cont/dxkPlusOne
                # J_u = dF/du = -dt * df_cont/du
                J_xkPlusOne_val = jnp.eye(self.n_x) - self.dt * J_cont_x
                J_u_val = -self.dt * J_cont_u

                # 4. Solve the IFT system: J_xkPlusOne @ f_u = -J_u
                # This gives f_u = d(xkPlusOne)/du
                f_u = jnp.linalg.solve(J_xkPlusOne_val, -J_u_val)
                return f_u
            
            # --- End of Backward Euler Implementation ---
            
            self._f_fcn = _backward_euler_integrator
            # --- Point to the new analytical Jacobian functions ---
            _f_x = _be_f_x_fcn
            _f_u = _be_f_u_fcn
            
        else:
            raise ValueError(f"Unknown integrator: '{integrator}'. Supported: 'rk4', 'midpoint', 'euler', 'backward_euler'.")

        # --- 2. Define all derivative functions ---
        
        # If not backward_euler, compute f_x and f_u using jacfwd
        if integrator not in ['backward_euler']:
            _f_x = jacfwd(self._f_fcn, argnums=0)
            _f_u = jacfwd(self._f_fcn, argnums=1)
        
        # --- Store (now JIT-compiled) continuous dynamics derivatives ---
        # We need these for the backward_euler analytical functions
        self._f_cont_x_fcn = jit(jacfwd(self._f_cont_fcn, argnums=0))
        self._f_cont_u_fcn = jit(jacfwd(self._f_cont_fcn, argnums=1))
        
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