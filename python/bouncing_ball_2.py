import jax
import jax.numpy as jnp
from jax import lax
import matplotlib.pyplot as plt
from functools import partial

# --- 1. Configuration ---
m = 1.0
dt = 0.01
g = 9.81
mu = 0.1        # High friction
T_end = 1.5     # 1.5 seconds
tol = 1e-6
W = jnp.array([[1.0, 0.0], [0.0, 1.0]]) 
EPSILON = 5  # Stable "Beanbag" parameter

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
def R_x(x, y, qk, uk):
    dq, du = x[:2], x[2:]
    res_pos = dq - dt * (du + uk)
    res_vel_x = m * du[0] - y[0]
    res_vel_y = m * du[1] + dt * m * g - y[1]
    return jnp.concatenate([res_pos, jnp.array([res_vel_x, res_vel_y])])

@jax.jit
def J_x_fun(x, y, qk, uk):
    return jax.jacfwd(lambda x_: R_x(x_, y, qk, uk))(x)

# --- 3. Solvers ---
def solve_newton(yk, qk, uk, x_init):
    """Inner Newton solver (Forward Mode Differentiable)"""
    def cond(s): return jnp.logical_and(s[2] < 100, jnp.logical_not(s[3]))
    def body(s):
        x, _, i, _ = s
        resid = R_x(x, yk, qk, uk)
        jac = J_x_fun(x, yk, qk, uk)
        delta = jnp.linalg.solve(jac, resid)
        return (x - delta, x, i + 1, jnp.linalg.norm(delta) < tol)
    return lax.while_loop(cond, body, (x_init, x_init, 0, False))[0]

def fp_hard(x, y, q, u, rT, rN):
    vn = -y[1] + rN*(q[1] + x[1])
    dpn = -prox_R0minus(vn)
    vt = -y[0] + rT*(u[0] + x[2])
    dpt = -prox_CT(vt, mu*dpn)
    y_new = jnp.array([dpt, dpn])
    return solve_newton(y_new, q, u, x), y_new

# --- New Helper ---
@jax.jit
def prox_CT_smooth(x, limit):
    """Smooth approximation of Coulomb Friction using Tanh"""
    # limit * tanh(x / limit) approximates clip(x, -limit, limit)
    # The '3.0' factor makes the slope steeper near zero (closer to true stick)
    safe_lim = limit + 1e-6
    return safe_lim * jnp.tanh(x / safe_lim)

# --- Updated Backward Pass ---
def fp_smooth(x, y, q, u, rT, rN):
    vn = -y[1] + rN*(q[1] + x[1])
    dpn = -prox_R0minus_smooth(vn)  # Smooth Normal
    
    vt = -y[0] + rT*(u[0] + x[2])
    
    # USE THE NEW SMOOTH FRICTION HERE
    dpt = -prox_CT_smooth(vt, mu*dpn) 
    
    y_new = jnp.array([dpt, dpn])
    return solve_newton(y_new, q, u, x), y_new

# --- 4. Differentiable Contact Solver (FIXED) ---

# FIX: Removed 'nondiff_argnums'. All args are now treated as differentiable inputs.
@jax.custom_jvp
def solve_contact(x_guess, y_guess, qk, uk, rT, rN):
    def cond(s): return jnp.logical_and(s[4] < 1000, jnp.logical_not(s[5]))
    def body(s):
        x, y = s[0], s[1]
        xn, yn = fp_hard(x, y, qk, uk, rT, rN)
        return (xn, yn, x, y, s[4]+1, jnp.linalg.norm(xn-s[2]) < tol)
    init = (x_guess, y_guess, x_guess, y_guess, 0, False)
    final = lax.while_loop(cond, body, init)
    return final[0], final[1], final[4].astype(jnp.float32)

@solve_contact.defjvp
def solve_contact_jvp(primals, tangents):
    # FIX: Unpack rT, rN from primals (they are now part of the differentiable signature)
    x_g, y_g, q, u, rT, rN = primals
    dx_g, dy_g, dq, du, drT, drN = tangents
    
    # 1. Primal Pass
    x_s, y_s, i_s = solve_contact(x_g, y_g, q, u, rT, rN)
    
    # 2. Implicit Diff
    def resid_fn(z_flat, q_, u_, rT_, rN_):
        x, y = z_flat[:4], z_flat[4:]
        xn, yn = fp_smooth(x, y, q_, u_, rT_, rN_)
        return z_flat - jnp.concatenate([xn, yn])
    
    z_s = jnp.concatenate([x_s, y_s])
    
    # Jacobian A = dResid/dZ
    A = jax.jacfwd(resid_fn, 0)(z_s, q, u, rT, rN)
    
    # Tikhonov Regularization (Stability Fix)
    A_damped = A + 1e-6 * jnp.eye(A.shape[0])
    
    # RHS = dResid/dParams * dParams
    # We pass drT and drN even if they are zero, to satisfy JAX tracing
    _, b = jax.jvp(
        lambda q_, u_, rT_, rN_: resid_fn(z_s, q_, u_, rT_, rN_), 
        (q, u, rT, rN), 
        (dq, du, drT, drN)
    )
    
    dz = jnp.linalg.solve(A_damped, -b)
    
    return (x_s, y_s, i_s), (dz[:4], dz[4:], 0.0)

# --- 5. Simulation ---
def simulate_trajectory(q0, u0):
    t_span = jnp.arange(0, T_end, dt)
    
    def scan_fn(carrier, _):
        q, u, x, y = carrier
        
        inv_M = jnp.diag(jnp.array([1.0/m, 1.0/m]))
        G = W.T @ inv_M @ W
        r = 1.0 / (dt * jnp.diag(G))
        
        x_sol, y_sol, _ = solve_contact(x, y, q, u, r[0]*dt, r[1])
        
        q_next = q + x_sol[:2]
        u_next = u + x_sol[2:]
        
        return (q_next, u_next, x_sol, y_sol), (q_next, u_next)

    init = (q0, u0, jnp.zeros(4), jnp.zeros(2))
    _, (q_stack, u_stack) = lax.scan(scan_fn, init, t_span)
    return q_stack, u_stack

def simulate_q_only(q0, u0):
    q_stack, _ = simulate_trajectory(q0, u0)
    return q_stack

# --- 6. Execution ---
if __name__ == "__main__":
    q0_val = jnp.array([0.0, 1.0])
    u0_val = jnp.array([1.0, 0.0])

    print("Calculating Trajectories...")
    q_traj, u_traj = simulate_trajectory(q0_val, u0_val)

    print("Calculating Sensitivities (Jacobian)...")
    jac_fn = jax.jit(jax.jacfwd(simulate_q_only, argnums=1))
    jac_traj = jac_fn(q0_val, u0_val)

    t_axis = jnp.arange(0, T_end, dt)

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
    
    # REMOVED: plt.show() here to prevent blocking

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
    
    # FINAL SHOW: Displays all created figures at once
    plt.show()