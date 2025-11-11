import jax
import jax.numpy as jnp
from jax import jit, grad, jacfwd, hessian
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
            integrator (str): Integration scheme ('euler', 'midpoint', or 'rk4').
        """
        self.n_x = n_x
        self.n_u = n_u
        self.dt = dt
        self.use_jit = use_jit
        self.integrator_type = integrator

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
        else:
            raise ValueError(f"Unknown integrator: '{integrator}'. Supported: 'rk4', 'midpoint', 'euler'.")

        # --- 2. Define all derivative functions ---
        #    These JAX transformations are now applied to the *generated*
        #    `self._f_fcn`, which could be Euler or RK4.
        
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