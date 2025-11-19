import jax.numpy as jnp
from system_base import System
import jax

class BouncingBallSystem(System):
    def __init__(self, dt=0.01, mu=0.1, smooth_epsilon=1.0):
        # 2D Bouncing Ball
        # q = [x, y]
        # v = [vx, vy]
        # u = [Fx, Fy] (External forces)
        n_q = 2
        n_v = 2
        n_u = 1
        n_c = 1 # One contact (ground)
        
        super().__init__(n_q, n_v, n_u, n_c, dt, 
                         integrator='contact_euler', # Use the new integrator
                         mu=mu, 
                         smooth_epsilon=smooth_epsilon)
        
        self.mass = 1.0
        self.g = 9.81
        
        # Target for cost (example)
        self.target_state = jnp.array([0.0, 1.0, 0.0, 0.0]) 

    # --- Physics Implementation ---

    def _mass_matrix(self, q):
        # Simple identity mass matrix
        return jnp.eye(2) * self.mass

    def _generalized_forces(self, q, v, u):
        # Gravity + Control
        # Gravity acts on Y (index 1)
        f_gravity = jnp.array([0.0, -self.mass * self.g])
        return f_gravity + jnp.array([0.0, u])

    def _contact_jacobian(self, q):
        # Contact Jacobian W. 
        # Rows = DOFs (x, y), Cols = Contact Impulse Directions (Tangent, Normal)
        # Tangent is X direction, Normal is Y direction
        # W = [[1, 0], [0, 1]]
        return jnp.array([[1.0, 0.0], 
                          [0.0, 1.0]])

    def _gap_function(self, q):
        # Gap is simply the Y height
        return jnp.array([q[1]])

    def _contact_velocity_function(self, q, v):
        # Tangential velocity is vx
        return jnp.array([v[0]])

    # --- Cost Implementation ---

    def _l_fcn(self, x, u):
        # Simple Quadratic Cost
        err = x - self.target_state
        Q = jnp.diag(jnp.array([1.0, 10.0, 0.1, 0.1]))
        R = jnp.eye(self.n_u) * 0.01
        return err.T @ Q @ err + u.T @ R @ u

    def _l_f_fcn(self, x):
        # Terminal Cost
        err = x - self.target_state
        Q_final = jnp.diag(jnp.array([10.0, 100.0, 1.0, 1.0]))
        return err.T @ Q_final @ err

# --- Example Usage ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create system
    sys = BouncingBallSystem(dt=0.01)
    
    # Initial state: x=0, y=1.0, vx=1.0, vy=0
    x0 = jnp.array([0.0, 1.0, 1.0, 0.0])
    u0 = 0.0
    
    # Test step
    x_next = sys.f_fcn(x0, u0)
    print(f"One step f(x0, u0):\n {x_next}")
    
    # Test Jacobian (using the custom JVP logic implicitly via jacfwd)
    # This verifies that standard JAX AD works on our non-smooth system
    A = sys.f_x_fcn(x0, u0)
    B = sys.f_u_fcn(x0, u0)
    print(f"\nJacobian df/dx shape: {A.shape}")
    print(f"Jacobian df/du shape: {B.shape}")
    
    # Run a simulation
    dt = sys.dt
    T = 3 # seconds
    tspan = jnp.arange(0, T, dt)
    N = tspan.shape[0]
    X_hist = [x0]
    curr_x = x0
    for i in range(N):
        curr_x = sys.f_fcn(curr_x, u0)
        X_hist.append(curr_x)
        
    X_hist = jnp.array(X_hist)
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.axhline(0, color='k', linestyle='--', label='Ground')
    plt.plot(X_hist[:, 0], X_hist[:, 1], '-', label='Trajectory')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Bouncing Ball Simulation with Contact-Implicit Base Class')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()