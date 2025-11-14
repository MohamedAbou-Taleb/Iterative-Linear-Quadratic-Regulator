import jax
import jax.numpy as jnp
import numpy as np
from typing import Union

# Import the base class from the same 'systems' package
from .system_base import System 

class MyUADoublePendulum(System):
    """
    JAX-based template for a Double Pendulum system.
    
    This class implements the multibody dynamics using the
    M(q)q_ddot + C(q, q_dot) + G(q) = tau form.
    
    You must fill in the TODO sections with your specific
    equations for M, C, and G.
    """
    
    def __init__(self, 
                 dt: float, 
                 x_target: Union[np.ndarray, jnp.ndarray], 
                 Q: jnp.ndarray, 
                 R: jnp.ndarray, 
                 Q_f: jnp.ndarray, 
                 # --- Add your physical parameters ---
                 g: float = 9.81, 
                 m1: float = 1.0, 
                 m2: float = 1.0,
                 l1: float = 1.0,
                 l2: float = 1.0,
                 d1: float = 0.01,
                 d2: float = 0.01,
                 theta1: float = 0.0, # 1/12 m l^2
                 theta2: float = 0.0,
                 # --- ---
                 use_jit: bool = True,
                 integrator: str = 'rk4'):
        """
        Constructor for the Double Pendulum system.
        
        Args:
            dt: Time step (s)
            x_target: Target state [q1, q2, q1_dot, q2_dot]
            Q: Stage cost state weight matrix (n_x, n_x)
            R: Stage cost control weight matrix (n_u, n_u)
            Q_f: Terminal cost state weight matrix (n_x, n_x)
            g, m1, m2, l1, l2: Physical parameters
            use_jit: Whether to JIT-compile functions
            integrator: Integration scheme ('rk4', 'midpoint', 'euler')
        """
        
        # 1. --- Define system properties ---
        self.n_x = 4  # [q1, q2, q1_dot, q2_dot]
        self.n_u = 1  # [tau1, tau2] (Assuming actuation on one joint)
                     # If only one joint is actuated, set n_u = 1 and
                     # adjust the physics equations accordingly.
        
        # --- Store physical parameters ---
        self.g = g
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.d1 = d1
        self.d2 = d2  
        self.theta1 = theta1
        self.theta2 = theta2
        
        # 2. --- Store cost parameters as JAX arrays ---
        self.x_target = jnp.asarray(x_target)
        self.Q = jnp.asarray(Q)
        self.R = jnp.asarray(R)
        self.Q_f = jnp.asarray(Q_f)
        
        # 3. --- Call the base class constructor ---
        super().__init__(self.n_x, self.n_u, dt, 
                         use_jit=use_jit, 
                         integrator=integrator)
        

    # --- Implement the 3 Abstract Methods ---

    def _f_cont_fcn(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """
        Continuous-time dynamics: x_dot = f(x, u)
        
        x_dot = [q_dot, q_ddot]
        
        Calculates q_ddot by solving the linear system:
        M(q) * q_ddot = b(q, q_dot, u)
        """
        # Unpack state and control
        q = x[:2]     # [q1, q2]
        q_dot = x[2:]   # [q1_dot, q2_dot]
        tau = u       # [tau1]
        
        # 1. Build the Mass Matrix M(q)
        M = self._build_mass_matrix(q)
        
        # 2. Build the vector of generalized forces
        #    b = tau - C(q, q_dot) - G(q)
        h = self._build_rhs_vector(q, q_dot, tau)
        
        # 3. Solve for accelerations
        #    q_ddot = M_inv * b
        q_ddot = jnp.linalg.solve(M, h)
        
        # 4. Stack derivatives to form x_dot
        #    x_dot = [q_dot, q_ddot]
        return jnp.concatenate([q_dot, q_ddot])


    def _l_fcn(self, x: jnp.ndarray, u: jnp.ndarray) -> float:
        """
        Stage cost.
        (This is a standard quadratic cost, modify as needed)
        """
        dx = x - self.x_target
        cost_x = 0.5 * dx.T @ self.Q @ dx
        cost_u = 0.5 * u.T @ self.R @ u
        # add a log barrier to keep input between -5 and 5
        # cost_u += -0.01 * (jnp.log(5 - u[0]) + jnp.log(5 + u[0]))
        
        val = (cost_x + cost_u) * self.dt 
        return val


    def _l_f_fcn(self, x: jnp.ndarray) -> float:
        """
        Terminal cost.
        (This is a standard quadratic cost, modify as needed)
        """
        dx = x - self.x_target
        val = 0.5 * dx.T @ self.Q_f @ dx
        return val

    # --- Physics Helper Methods (FILL THESE IN) ---

    def _build_mass_matrix(self, q: jnp.ndarray) -> jnp.ndarray:
        """
        Builds the 2x2 Mass Matrix M(q).
        
        Args:
            q: Joint positions [q1, q2]
            
        Returns:
            M: 2x2 Mass Matrix
        """
        q1, q2 = q
        
        m11 = (self.m1*self.l1**2)/4 + self.m2*self.l1**2 + (self.m2*self.l2**2)/4 + self.m2*self.l1*self.l2*jnp.cos(q2) + self.theta1 + self.theta2
        m12 = (self.m2*self.l2**2)/4 + (self.m2*self.l1*self.l2*jnp.cos(q2))/2 + self.theta2
        m21 = m12
        m22 = (self.m2*self.l2**2)/4 + self.theta2
        M = jnp.array([
            [m11, m12],
            [m21, m22]
        ])

        
        return M

    def _build_rhs_vector(self, q: jnp.ndarray, q_dot: jnp.ndarray, tau: jnp.ndarray) -> jnp.ndarray:
        """
        Builds the right-hand-side vector b(q, q_dot, tau)
        where:
        b = tau - C(q, q_dot) - G(q)
        
        C = Coriolis/Centripetal vector
        G = Gravity vector
        tau = Control torque vector
        
        Args:
            q: Joint positions [q1, q2]
            q_dot: Joint velocities [q1_dot, q2_dot]
            tau: Joint torques [tau1, tau2]
            
        Returns:
            b: 2x1 RHS vector
        """
        q1, q2 = q
        q1d, q2d = q_dot
        tau = tau
        
        # --- JAX-ified equations for h1 and h2 ---
        
        s1 = jnp.sin(q1)
        s2 = jnp.sin(q2)
        s12 = jnp.sin(q1 + q2)
        
        f_c1 =  (self.m2*self.l1*self.l2*s2*(2*q1d*q2d + q2d**2))/2
        f_c2 = -(self.m2*self.l1*self.l2*s2*(q1d**2))/2
        f_c = jnp.array([f_c1, f_c2])

        f_g1 = -self.m2*self.g*(self.l2*s12/2 + self.l1*s1) - (self.m1*self.g*self.l1*s1)/2
        f_g2 = -self.m2*self.g*(self.l2*s12)/2
        f_g = jnp.array([f_g1, f_g2])

        f_d1 = -self.d1*q1d
        f_d2 = -self.d2*q2d
        f_d = jnp.array([f_d1, f_d2])

        f_act = jnp.array([tau[0], 0.0])

        h = f_act + f_c + f_g + f_d
        
        return h