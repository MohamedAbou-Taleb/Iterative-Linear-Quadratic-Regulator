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
T_end = 5     # 1.5 seconds
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
@jax.custom_jvp
def solve_newton(yk, qk, uk, tauk, x_init):
    def cond(s): return jnp.logical_and(s[2] < 100, jnp.logical_not(s[3]))
    def body(s):
        x, _, i, _ = s
        resid = R_x(x, yk, qk, uk, tauk)
        jac = J_x_fun(x, yk, qk, uk, tauk)
        delta = jnp.linalg.solve(jac, resid)
        return (x - delta, x, i + 1, jnp.linalg.norm(delta) < tol)
    return lax.while_loop(cond, body, (x_init, x_init, 0, False))[0]

@solve_newton.defjvp
def solve_newton_jvp(primals, tangents):
    # Primals: (yk, qk, uk, tauk, x_init)
    # Tangents: (dyk, dqk, duk, dtauk, dx_init)
    yk, qk, uk, tauk, x_init = primals
    dyk, dqk, duk, dtauk, dx_init = tangents

    # 1. Solve the primal problem (get the converged solution x_bar)
    x_bar = solve_newton(yk, qk, uk, tauk, x_init)
    
    # The derivative wrt x_init is zero since the converged solution 
    # should be independent of the initial guess, assuming convergence.
    # We will ignore dx_init in the JVP calculation.

    # 2. Get the Jacobian of R w.r.t x evaluated at the solution x_bar
    # This is J_x = dR/dx, which is the 'jac' from the while_loop.
    J_x = J_x_fun(x_bar, yk, qk, uk, tauk)

    # 3. Compute the JVP of R w.r.t. parameters p = (yk, qk, uk, tauk)
    # R_p * dot_p = dR/dp * dot_p 
    # This is done by applying jvp to R_x w.r.t the parameters (yk, qk, uk, tauk)
    
    # Define a helper function for R_x w.r.t. parameters for JVP
    def R_params(p1, p2, p3, p4, x_fixed):
        return R_x(x_fixed, p1, p2, p3, p4)

    # Compute the JVP of R_x w.r.t. the parameters (yk, qk, uk, tauk)
    # The primal output of the JVP is R(x_bar, p) which should be near zero.
    # The tangent output is (dR/dp) * dot_p
    _, R_p_dot_p = jax.jvp(
        lambda p1, p2, p3, p4: R_params(p1, p2, p3, p4, x_bar),
        (yk, qk, uk, tauk),
        (dyk, dqk, duk, dtauk)
    )

    # 4. Solve the linear system for the JVP (dot_x)
    # J_x * dot_x = - (dR/dp) * dot_p
    dot_x = jnp.linalg.solve(J_x, -R_p_dot_p)

    return x_bar, dot_x

def fp_hard(x, y, q, u, tauk, rT, rN):
    vn = -y[1] + rN*(q[1] + x[1])
    gap = q[1] + dt*u[1]/2
    gamma_N = x[3] + u[1]   
    xi_N = gamma_N + 1*u[1]
    dpn_contact = -prox_R0minus(-y[1] + dt*rN*xi_N)
    dpn = jnp.where(gap > 0.0, 0.0, dpn_contact)
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
        xn, yn = fp_hard(x, y, qk, uk, tauk, rT, rN)
        return (xn, yn, x, y, s[4]+1, jnp.linalg.norm(xn-s[2]) < tol)
    init = (x_guess, y_guess, x_guess, y_guess, 0, False)
    final = lax.while_loop(cond, body, init)
    return final[0], final[1], final[4].astype(jnp.float32)
 

@solve_contact.defjvp
def solve_contact_jvp(primals, tangents):
    x_g, y_g, q, u, tau, rT, rN = primals
    dx_g, dy_g, dq, du, dtau, drT, drN = tangents
    
    # 1. Primal Pass (Forward) - USES TRUE PHYSICS
    x_s, y_s, i_s = solve_contact(x_g, y_g, q, u, tau, rT, rN)
    
    # 2. Implicit Diff (Backward) - USES HALO PHYSICS
    def resid_fn(z_flat, q_, u_, tau_, rT_, rN_):
        x, y = z_flat[:4], z_flat[4:]

        # xn, yn = fp_smooth(x, y, q_, u_, tau_, rT_, rN_)
        xn, yn = fp_hard(x, y, q_, u_, tau_, rT_, rN_) 
        return z_flat - jnp.concatenate([xn, yn])
    
    z_s = jnp.concatenate([x_s, y_s])
    
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

def simulate_q_only(q0, u0, tau):
    q_full, _ = simulate_trajectory(q0, u0, tau)
    return q_full

# --- NEW WRAPPER: Velocity Only ---
def simulate_u_only(q0, u0, tau):
    _, u_full = simulate_trajectory(q0, u0, tau)
    return u_full

# --- 6. Execution ---
if __name__ == "__main__":
    q0_val = jnp.array([0.0, 1.0])
    u0_val = jnp.array([1.0, 0.0])
    tau_val = 0.0 

    print("Calculating Trajectories...")
    q_traj, u_traj = simulate_trajectory(q0_val, u0_val, tau_val)

    print("Calculating Position Sensitivities (d q / d u0)...")
    jac_q_fn = jax.jit(jax.jacfwd(simulate_q_only, argnums=1))
    jac_traj_q = jac_q_fn(q0_val, u0_val, tau_val)

    print("Calculating Velocity Sensitivities (d u / d u0)...")
    jac_u_fn = jax.jit(jax.jacfwd(simulate_u_only, argnums=1))
    jac_traj_u = jac_u_fn(q0_val, u0_val, tau_val)

    t_axis = jnp.arange(0, T_end + dt, dt)
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
    fig1.suptitle("Figure 1: State Trajectories")

    # Figure 2: Position Sensitivities (q vs u0)
    fig2, axes = plt.subplots(2, 2, figsize=(10, 8))
    labels_q = [
        (r'$\frac{\partial q_x}{\partial u_x}$', 0, 0),
        (r'$\frac{\partial q_x}{\partial u_y}$', 0, 1),
        (r'$\frac{\partial q_y}{\partial u_x}$', 1, 0),
        (r'$\frac{\partial q_y}{\partial u_y}$', 1, 1)
    ]
    
    for label, r, c in labels_q:
        ax = axes[r, c]
        data = jac_traj_q[:, r, c]
        ax.plot(t_axis, data, linewidth=2)
        ax.set_title(label)
        ax.grid(True)
        
        if (r==0 and c==0) or (r==1 and c==1):
            ax.plot(t_axis, t_axis, 'r--', alpha=0.5, label='y=t')
            ax.legend()
    fig2.suptitle(r"Figure 2: Sensitivity of Position $q(t)$ to $u(0)$")
    plt.tight_layout()

    # Figure 3: Velocity Sensitivities (u vs u0)
    fig3, axes = plt.subplots(2, 2, figsize=(10, 8))
    labels_u = [
        (r'$\frac{\partial u_x}{\partial u_x}$', 0, 0),
        (r'$\frac{\partial u_x}{\partial u_y}$', 0, 1),
        (r'$\frac{\partial u_y}{\partial u_x}$', 1, 0),
        (r'$\frac{\partial u_y}{\partial u_y}$', 1, 1)
    ]
    
    for label, r, c in labels_u:
        ax = axes[r, c]
        data = jac_traj_u[:, r, c]
        ax.plot(t_axis, data, linewidth=2)
        ax.set_title(label)
        ax.grid(True)
        
        # Theoretical expectations for u vs u0 in free space:
        # u_t = u0 + a*t.  du/du0 = 1.
        if (r==0 and c==0) or (r==1 and c==1):
            ax.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='y=1')
            ax.legend()
    fig3.suptitle(r"Figure 3: Sensitivity of Velocity $u(t)$ to $u(0)$")
    plt.tight_layout()

    plt.show()