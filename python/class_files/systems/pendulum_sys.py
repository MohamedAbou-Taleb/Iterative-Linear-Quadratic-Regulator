import jax
import jax.numpy as jnp
import numpy as np 
from typing import Sequence, Union

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
        #    Pass all required arguments to the base class
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
    # Simple test to instantiate the Pendulum system
    dt = 0.01
    x_target = jnp.array([jnp.pi, 0.0])  # Target upright position
    Q = jnp.diag(jnp.array([10.0, 1.0]))
    R = jnp.array([[0.1]])
    Q_f = jnp.diag(jnp.array([100.0, 10.0]))
    use_jit = False
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
                              integrator='euler')
    print("Pendulum system instantiated successfully.")
    x_0 = jnp.array([jnp.pi, 0.0])
    u_0 = jnp.array([1.0])
    x_next = pendulum_sys.f_fcn(x_0, u_0)
    print(f"Next state from f_fcn: {x_next}")
    f_x = pendulum_sys.f_x_fcn(x_0, u_0)
    print(f"f_x_fcn at (x_0, u_0): {f_x}")
    # make a timing comaprison between jit and non-jit
    import time
    start_time = time.time()
    for _ in range(100):
        x_next = pendulum_sys.f_fcn(x_0, u_0)
        f_x = pendulum_sys.f_x_fcn(x_0, u_0)
    end_time = time.time()
    print(f"Time taken for 100 f_fcn calls (non-jit): {end_time - start_time} seconds")
    # now test with jit enabled
    pendulum_sys_jit = MyPendulum(dt=dt,
                                    x_target=x_target, 
                                    Q=Q, 
                                    R=R, 
                                    Q_f=Q_f,
                                    g=g,
                                    l=l,
                                    d=d,
                                    use_jit=True,
                                    integrator='midpoint')
    start_time = time.time()
    for _ in range(100):
        x_next = pendulum_sys_jit.f_fcn(x_0, u_0)
        f_x = pendulum_sys_jit.f_x_fcn(x_0, u_0)
    end_time = time.time()
    print(f"Time taken for 1000 f_fcn calls (jit): {end_time - start_time} seconds")