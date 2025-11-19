import jax
import jax.numpy as jnp
from jax import lax
import matplotlib.pyplot as plt
from functools import partial

# --- 1. Configuration ---
m = 1.0
dt = 0.01
g = 9.81
mu = 0.1        # Low friction
# mu = 10        # High friction
T_end = 1.5     # 1.5 seconds
tol = 1e-6
W = jnp.array([[1.0, 0.0], [0.0, 1.0]]) 
EPSILON = 1     # Stable "Beanbag" parameter

# --- 2. Physics Helpers ---
@jax.jit
def prox_R0minus(x): 
    return jnp.minimum(x, 0.0)

@jax.jit
def prox_R0minus_smooth(x): 
    return -EPSILON * lax.log(1.0 + lax.exp(-x / EPSILON))

@jax.jit
def prox_CT(x, limit): 
    return jnp.clip(x, -limit, limit)

@jax.jit
def R_x(x, y, qk, uk, tauk):
    dq, du = x[:2], x[2:]
    res_pos = dq - dt * (du + uk)
    res_vel_x = m * du[0] - y[0]
    res_vel_y = m * du[1] + dt * m * g - y[1] - dt * tauk
    return jnp.concatenate([res_pos, jnp.array([res_vel_x, res_vel_y])])

@jax.jit
def J_x_fun(x, y, qk, uk, tauk):
    return jax.jacfwd(lambda x_: R_x(x_, y, qk, uk, tauk))(x)

# --- 3. Solvers ---
def solve_newton(yk, qk, uk, tauk, x_init):
    def cond(s): return jnp.logical_and(s[2] < 100, jnp.logical_not(s[3]))
    def body(s):
        x, _, i, _ = s
        resid = R_x(x, yk, qk, uk, tauk)
        jac = J_x_fun(x, yk, qk, uk, tauk)
        delta = jnp.linalg.solve(jac, resid)
        return (x - delta, x, i + 1, jnp.linalg.norm(delta) < tol)
    return lax.while_loop(cond, body, (x_init, x_init, 0, False))[0]

def fp_hard(x, y, q, u, tauk, rT, rN):
    vn = -y[1] + rN*(q[1] + x[1])
    dpn = -prox_R0minus(vn)
    vt = -y[0] + rT*(u[0] + x[2])
    dpt = -prox_CT(vt, mu*dpn)
    y_new = jnp.array([dpt, dpn])
    return solve_newton(y_new, q, u, tauk, x), y_new

# --- New Helper ---
@jax.jit

# --- Updated Backward Pass ---
def fp_smooth(x, y, q, u, tauk, rT, rN):
    vn = -y[1] + rN*(q[1] + x[1])
    dpn = -prox_R0minus_smooth(vn)  # Smooth Normal
    
    vt = -y[0] + rT*(u[0] + x[2])
    
    # USE THE NEW SMOOTH FRICTION HERE
    dpt = -prox_CT(vt, mu*dpn) 
    
    y_new = jnp.array([dpt, dpn])
    return solve_newton(y_new, q, u, tauk, x), y_new
# --- 4. Differentiable Contact Solver ---                 
@jax.custom_jvp
def solve_contact(x_guess, y_guess, qk, uk, tauk, rT, rN):
    def cond(s): return jnp.logical_and(s[4] < 1000, jnp.logical_not(s[5]))
    def body(s):
        x, y = s[0], s[1]
        xn, yn = fp_hard(x, y, qk, uk, tauk, rT, rN)
        return (xn, yn, x, y, s[4]+1, jnp.linalg.norm(xn-s[2]) < tol)
    init = (x_guess, y_guess, x_guess, y_guess, 0, False)
    final = lax.while_loop(cond, body, init)
    return final[0], final[1], final[4].astype(jnp.float32)
 

# --- 3. Modify the JVP (The Backward Pass) ---
@solve_contact.defjvp
def solve_contact_jvp(primals, tangents):
    x_g, y_g, q, u, tau, rT, rN = primals
    dx_g, dy_g, dq, du, dtau, drT, drN = tangents
    
    # 1. Primal Pass (Forward) - USES TRUE PHYSICS
    # We use fp_hard (or fp_smooth without halo) so the simulation looks real.
    x_s, y_s, i_s = solve_contact(x_g, y_g, q, u, tau, rT, rN)
    
    # 2. Implicit Diff (Backward) - USES HALO PHYSICS
    def resid_fn(z_flat, q_, u_, tau_, rT_, rN_):
        x, y = z_flat[:4], z_flat[4:]
        
        xn, yn = fp_hard(x, y, q_, u_, tau_, rT_, rN_)
        # xn, yn = fp_smooth(x, y, q_, u_, tau_, rT_, rN_)

        return z_flat - jnp.concatenate([xn, yn])
    
    z_s = jnp.concatenate([x_s, y_s])
    
    # The rest of the Jacobian logic remains the same...
    A = jax.jacfwd(resid_fn, 0)(z_s, q, u, tau, rT, rN)
    A_damped = A + 1e-6 * jnp.eye(A.shape[0])
    
    _, b = jax.jvp(
        lambda q_, u_, tau_, rT_, rN_: resid_fn(z_s, q_, u_, tau_, rT_, rN_), 
        (q, u, tau, rT, rN), 
        (dq, du, dtau, drT, drN)
    )
    
    dz = jnp.linalg.solve(A_damped, -b)
    return (x_s, y_s, i_s), (dz[:4], dz[4:], 0.0)
# --- 5. Simulation ---
def simulate_trajectory(q0, u0, tau):
    # Generates t = dt, 2dt, ..., T_end
    t_span = jnp.arange(0, T_end, dt)
    def scan_fn(carrier, _):
        q, u, x, y = carrier
        
        inv_M = jnp.diag(jnp.array([1.0/m, 1.0/m]))
        G = W.T @ inv_M @ W
        r = 1.0 / (dt * jnp.diag(G))
        x_sol, y_sol, _ = solve_contact(x, y, q, u, tau, r[0]*dt, r[1])
        
        q_next = q + x_sol[:2]
        u_next = u + x_sol[2:]
        
        return (q_next, u_next, x_sol, y_sol), (q_next, u_next)

    init = (q0, u0, jnp.zeros(4), jnp.zeros(2))
    _, (q_stack, u_stack) = lax.scan(scan_fn, init, t_span)
    
    # PREPEND Initial Conditions so trajectory includes t=0
    q_full = jnp.vstack([q0, q_stack])
    u_full = jnp.vstack([u0, u_stack])
    
    return q_full, u_full

def simulate_q_only(q0, u0, tau):
    # This wrapper now returns the full trajectory including t=0
    q_full, _ = simulate_trajectory(q0, u0, tau)
    return q_full

# --- 6. Execution ---
if __name__ == "__main__":
    q0_val = jnp.array([0.0, 1.0])
    u0_val = jnp.array([1.0, 0.0])

    print("Calculating Trajectories...")
    tau_val = 0.0  # No actuation for now
    q_traj, u_traj = simulate_trajectory(q0_val, u0_val, tau_val)

    print("Calculating Sensitivities (Jacobian)...")
    jac_fn = jax.jit(jax.jacfwd(simulate_q_only, argnums=1))
    jac_traj = jac_fn(q0_val, u0_val, tau_val)

    # Time axis must now start at 0.0 and have length N+1
    t_axis = jnp.arange(0, T_end + dt, dt)
    
    # Handle potential off-by-one error if floating point math makes arange inclusive/exclusive inconsistent
    # We ensure t_axis is same length as data
    if len(t_axis) != len(q_traj):
        t_axis = jnp.linspace(0, T_end, len(q_traj))

    # --- PLOTTING ---
    
    # Figure 1: State Trajectories
    fig1 = plt.figure(figsize=(12, 8))
    gs1 = fig1.add_gridspec(2, 2)

    ax1 = fig1.add_subplot(gs1[:, 0])
    ax1.plot(q_traj[:, 0], q_traj[:, 1], 'b-o', markersize=3)
    ax1.axhline(0, color='k', linewidth=2)
    ax1.set_title("Spatial Path (Y vs X)")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.axis('equal')
    ax1.grid(True)

    ax2 = fig1.add_subplot(gs1[0, 1])
    ax2.plot(t_axis, q_traj[:, 1], 'g-', linewidth=2)
    ax2.set_title("Height vs Time")
    ax2.set_ylabel("Y (m)")
    ax2.grid(True)

    ax3 = fig1.add_subplot(gs1[1, 1])
    ax3.plot(t_axis, u_traj[:, 0], 'r--', label='$u_x$')
    ax3.plot(t_axis, u_traj[:, 1], 'b-', label='$u_y$')
    ax3.set_title("Velocities vs Time")
    ax3.set_ylabel("m/s")
    ax3.legend()
    ax3.grid(True)

    # Figure 2: Sensitivities
    fig2, axes = plt.subplots(2, 2, figsize=(10, 8))
    labels = [
        (r'$\frac{\partial q_x}{\partial u_x}$', 0, 0),
        (r'$\frac{\partial q_x}{\partial u_y}$', 0, 1),
        (r'$\frac{\partial q_y}{\partial u_x}$', 1, 0),
        (r'$\frac{\partial q_y}{\partial u_y}$', 1, 1)
    ]
    
    for label, r, c in labels:
        ax = axes[r, c]
        data = jac_traj[:, r, c]
        ax.plot(t_axis, data, linewidth=2)
        ax.set_title(label)
        ax.grid(True)
        
        if (r==0 and c==0) or (r==1 and c==1):
            ax.plot(t_axis, t_axis, 'r--', alpha=0.5, label='y=t')
            ax.legend()

    fig2.suptitle(r"Sensitivity of Position $q(t)$ to Initial Velocity $u(0)$")
    plt.tight_layout()
    
    plt.show()