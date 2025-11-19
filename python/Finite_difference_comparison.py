import jax
import jax.numpy as jnp
from jax import lax
import matplotlib.pyplot as plt
from functools import partial

# --- 1. Configuration ---
m = 1.0
dt = 0.01
g = 9.81
mu = 0.1
T_end = 1.5
tol = 1e-6
W = jnp.array([[1.0, 0.0], [0.0, 1.0]]) 
EPSILON = 10.0  # Halo smoothing

# --- 2. Physics Helpers ---
@jax.jit
def prox_R0minus(x): return jnp.minimum(x, 0.0)
@jax.jit
def prox_R0minus_smooth(x): return -EPSILON * lax.log(1.0 + lax.exp(-x / EPSILON))
@jax.jit
def prox_CT(x, limit): return jnp.clip(x, -limit, limit)

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

def fp_smooth(x, y, q, u, tauk, rT, rN):
    vn = -y[1] + rN*(q[1] + x[1])
    dpn = -prox_R0minus_smooth(vn) 
    vt = -y[0] + rT*(u[0] + x[2])
    dpt = -prox_CT(vt, mu*dpn)
    y_new = jnp.array([dpt, dpn])
    return solve_newton(y_new, q, u, tauk, x), y_new

# --- 4. Differentiable Contact Solver ---
@jax.custom_jvp
def solve_contact(x_guess, y_guess, qk, uk, tauk, rT, rN):
    def cond(s): return jnp.logical_and(s[4] < 1000, jnp.logical_not(s[5]))
    def body(s):
        x, y = s[0], s[1]
        xn, yn = fp_smooth(x, y, qk, uk, tauk, rT, rN)
        return (xn, yn, x, y, s[4]+1, jnp.linalg.norm(xn-s[2]) < tol)
    init = (x_guess, y_guess, x_guess, y_guess, 0, False)
    final = lax.while_loop(cond, body, init)
    return final[0], final[1], final[4].astype(jnp.float32)

@solve_contact.defjvp
def solve_contact_jvp(primals, tangents):
    x_g, y_g, q, u, tau, rT, rN = primals
    dx_g, dy_g, dq, du, dtau, drT, drN = tangents
    x_s, y_s, i_s = solve_contact(x_g, y_g, q, u, tau, rT, rN)
    
    # Implicit Diff using SMOOTH (Halo) Physics
    def resid_fn(z_flat, q_, u_, tau_, rT_, rN_):
        x, y = z_flat[:4], z_flat[4:]
        xn, yn = fp_hard(x, y, q_, u_, tau_, rT_, rN_) 
        return z_flat - jnp.concatenate([xn, yn])
    
    z_s = jnp.concatenate([x_s, y_s])
    A = jax.jacfwd(resid_fn, 0)(z_s, q, u, tau, rT, rN)
    A_damped = A + 1e-6 * jnp.eye(A.shape[0])
    _, b = jax.jvp(lambda q_, u_, tau_, rT_, rN_: resid_fn(z_s, q_, u_, tau_, rT_, rN_), 
                   (q, u, tau, rT, rN), (dq, du, dtau, drT, drN))
    dz = jnp.linalg.solve(A_damped, -b)
    return (x_s, y_s, i_s), (dz[:4], dz[4:], 0.0)

# --- 5. Simulation ---
def simulate_trajectory(q0, u0, tau):
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
    
    q_full = jnp.vstack([q0, q_stack])
    u_full = jnp.vstack([u0, u_stack])
    return q_full, u_full

def simulate_u_only(q0, u0, tau):
    _, u_full = simulate_trajectory(q0, u0, tau)
    return u_full

# --- 6. Finite Difference Implementation ---
def compute_finite_differences(q0, u0, tau, delta=1e-4):
    """
    Computes Jacobian d(u_traj)/d(u0) using Central Differences.
    """
    print(f"Running Finite Differences (delta={delta})...")
    
    # 1. Perturb u0_x
    u0_p_x = u0 + jnp.array([delta, 0.0])
    u0_m_x = u0 - jnp.array([delta, 0.0])
    _, u_p_x = simulate_trajectory(q0, u0_p_x, tau)
    _, u_m_x = simulate_trajectory(q0, u0_m_x, tau)
    d_ux = (u_p_x - u_m_x) / (2 * delta)
    
    # 2. Perturb u0_y
    u0_p_y = u0 + jnp.array([0.0, delta])
    u0_m_y = u0 - jnp.array([0.0, delta])
    _, u_p_y = simulate_trajectory(q0, u0_p_y, tau)
    _, u_m_y = simulate_trajectory(q0, u0_m_y, tau)
    d_uy = (u_p_y - u_m_y) / (2 * delta)
    
    # Stack to create Jacobian shape (Time, 2_u_traj, 2_u0)
    # Axis 0: Time
    # Axis 1: Output Dimension (u_x, u_y)
    # Axis 2: Input Dimension (u0_x, u0_y)
    return jnp.stack([d_ux, d_uy], axis=-1)

# --- 7. Execution ---
if __name__ == "__main__":
    q0_val = jnp.array([0.0, 1.0])
    u0_val = jnp.array([1.0, 0.0])
    tau_val = 0.0 

    # 1. Analytical Gradient (Implicit Differentiation)
    print("Computing Analytical Gradients...")
    jac_u_fn = jax.jit(jax.jacfwd(simulate_u_only, argnums=1))
    jac_analytical = jac_u_fn(q0_val, u0_val, tau_val)

    # 2. Numerical Gradient (Finite Differences)
    jac_fd = compute_finite_differences(q0_val, u0_val, tau_val)

    # Time axis
    t_axis = jnp.arange(0, T_end + dt, dt)
    if len(t_axis) != len(jac_analytical):
        t_axis = jnp.linspace(0, T_end, len(jac_analytical))

    # --- PLOTTING COMPARISON ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    labels = [
        (r'$\frac{\partial u_x}{\partial u_x}$', 0, 0),
        (r'$\frac{\partial u_x}{\partial u_y}$', 0, 1),
        (r'$\frac{\partial u_y}{\partial u_x}$', 1, 0),
        (r'$\frac{\partial u_y}{\partial u_y}$', 1, 1)
    ]
    
    for label, r, c in labels:
        ax = axes[r, c]
        
        # Plot Analytical
        ax.plot(t_axis, jac_analytical[:, r, c], 'b-', linewidth=3, alpha=0.6, label='Implicit (Smooth)')
        
        # Plot Finite Difference
        ax.plot(t_axis, jac_fd[:, r, c], 'r--', linewidth=1.5, label='Finite Diff (Hard)')
        
        ax.set_title(label)
        ax.grid(True)
        if r==0 and c==0: ax.legend()

    fig.suptitle("Comparison: Implicit Diff (Halo) vs Finite Differences (Hard Physics)")
    plt.tight_layout()
    plt.show()
    # plt.savefig('fd_vs_implicit.png')