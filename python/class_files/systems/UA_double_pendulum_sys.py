import jax
import jax.numpy as jnp
import numpy as np
from typing import Union
import time 
from jax import jit, lax 
import matplotlib.pyplot as plt 

# Robust import
try:
    from .system_base import System
except ImportError:
    from system_base import System

class MyUADoublePendulum(System):
    """
    JAX-based Under-Actuated Double Pendulum (Acrobot style).
    """
    
    def __init__(self, 
                 dt: float, 
                 x_target: Union[np.ndarray, jnp.ndarray], 
                 Q: jnp.ndarray, 
                 R: jnp.ndarray, 
                 Q_f: jnp.ndarray, 
                 # --- Physical parameters ---
                 g: float = 9.81, 
                 m1: float = 1.0, 
                 m2: float = 1.0,
                 l1: float = 1.0,
                 l2: float = 1.0,
                 d1: float = 0.01,
                 d2: float = 0.01,
                 theta1: float = 0.0, 
                 theta2: float = 0.0, 
                 # --- System settings ---
                 use_jit: bool = True,
                 integrator: str = 'rk4',
                 mu: float = jnp.array([0.0, 0.0]),
                 smooth_epsilon: float = 1.0,
                 d_wall: float = 0.1,
                 e_restitution=jnp.array([0.0, 0.0]),
                 **kwargs):
        
        # 1. --- Define system properties ---
        n_q = 2  # [q1, q2]
        n_v = 2  # [q1_dot, q2_dot]
        n_u = 1  # [tau1] - Underactuated
        n_c = 2  # No contacts
        
        self.g = g
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.d1 = d1
        self.d2 = d2  
        self.theta1 = theta1
        self.theta2 = theta2
        
        # 2. --- Store cost parameters ---
        self.x_target = jnp.asarray(x_target)
        self.Q = jnp.asarray(Q)
        self.R = jnp.asarray(R)
        self.Q_f = jnp.asarray(Q_f)
        
        # 3. --- Call base class ---
        super().__init__(n_q, n_v, n_u, n_c, dt, 
                         integrator=integrator,
                         mu=mu,
                         smooth_epsilon=smooth_epsilon,
                         e_restitution=e_restitution,
                         **kwargs)
        
        self.d_wall = d_wall
    # --- Physics Implementation ---

    def _mass_matrix(self, q: jnp.ndarray) -> jnp.ndarray:
        """Returns M(q) (2x2)."""
        q1, q2 = q
        
        m11 = (self.m1*self.l1**2)/4 + self.m2*self.l1**2 + (self.m2*self.l2**2)/4 + \
              self.m2*self.l1*self.l2*jnp.cos(q2) + self.theta1 + self.theta2
        
        m12 = (self.m2*self.l2**2)/4 + (self.m2*self.l1*self.l2*jnp.cos(q2))/2 + self.theta2
        
        m21 = m12
        m22 = (self.m2*self.l2**2)/4 + self.theta2
        
        return jnp.array([[m11, m12], [m21, m22]])

    def _generalized_forces(self, q: jnp.ndarray, v: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Returns h(q, v, u) (2,)."""
        q1, q2 = q
        q1d, q2d = v
        tau = u # shape (1,)
        
        s1 = jnp.sin(q1)
        s2 = jnp.sin(q2)
        s12 = jnp.sin(q1 + q2)
        
        # Coriolis
        f_c1 =  (self.m2*self.l1*self.l2*s2*(2*q1d*q2d + q2d**2))/2
        f_c2 = -(self.m2*self.l1*self.l2*s2*(q1d**2))/2
        f_c = jnp.array([f_c1, f_c2])

        # Gravity
        f_g1 = -self.m2*self.g*(self.l2*s12/2 + self.l1*s1) - (self.m1*self.g*self.l1*s1)/2
        f_g2 = -self.m2*self.g*(self.l2*s12/2)
        f_g = jnp.array([f_g1, f_g2])

        # Damping
        f_d1 = -self.d1*q1d
        f_d2 = -self.d2*q2d
        f_d = jnp.array([f_d1, f_d2])

        # Actuation (Only on joint 1)
        f_act = jnp.array([tau[0], 0.0])

        return f_act + f_c + f_g + f_d

    def _contact_jacobian(self, q: jnp.ndarray) -> jnp.ndarray:
        """Returns W(q) of shape (n_v, 2*n_c)."""
        w_T1 = jnp.array([self.l1 * jnp.sin(q[0]), 0])
        w_N1 = jnp.array([-self.l1 * jnp.cos(q[0]), 0])
        w_T2 = jnp.array([self.l2 * jnp.sin(q[0] + q[1]) + self.l1 * jnp.sin(q[0]), self.l2 + jnp.sin(q[0] + q[1])])
        w_N2 = jnp.array([-self.l2*jnp.cos(q[0] + q[1]) - self.l1 * jnp.cos(q[0]), -self.l2 * jnp.cos(q[0] + q[1])])
        W = jnp.vstack([w_T1.T, w_N1.T, w_T2.T, w_N2.T]).T
        return W
        
    def _gap_function(self, q: jnp.ndarray) -> jnp.ndarray:
        """Returns gap vector g(q) of shape (n_c,)."""
        g_N1 = self.d_wall - self.l1 * jnp.sin(q[0])
        g_N2 = self.d_wall - self.l2 * jnp.sin(q[0] + q[1]) - self.l1 * jnp.sin(q[0])
        g_N = jnp.array([g_N1, g_N2])
        return g_N
        
    def _contact_velocity_function(self, q: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Returns tangential contact velocity gamma(q, v) of shape (n_c,)."""
        gamma_T1 = v[0] * self.l1 * jnp.sin(q[0])
        gamma_T2 = -v[0]*( self.l2 * jnp.cos(q[0] + q[1]) + self.l1 * jnp.cos(q[0]) ) - v[1] * self.l2 * jnp.cos(q[0] + q[1])
        gamma_T = jnp.array([gamma_T1, gamma_T2])
        return gamma_T
    # --- Cost ---

    def _l_fcn(self, x: jnp.ndarray, u: jnp.ndarray) -> float:
        dx = x - self.x_target
        cost_x = 0.5 * dx.T @ self.Q @ dx
        cost_u = 0.5 * u.T @ self.R @ u
        return (cost_x + cost_u) * self.dt 

    def _l_f_fcn(self, x: jnp.ndarray) -> float:
        dx = x - self.x_target
        return 0.5 * dx.T @ self.Q_f @ dx

# --- Test ---
if __name__ == "__main__":
    dt = 0.01
    x_target = jnp.array([jnp.pi, 0.0, 0.0, 0.0]) 
    Q = jnp.eye(4)
    R = jnp.eye(1)
    Q_f = jnp.eye(4) * 10
    
    sys = MyUADoublePendulum(dt, x_target, Q, R, Q_f, e_restitution=jnp.array([0.0, 0.0]))
    x0 = jnp.zeros(4)
    u0 = jnp.array([1.0])
    
    print("Step:", sys.f_fcn(x0, u0))