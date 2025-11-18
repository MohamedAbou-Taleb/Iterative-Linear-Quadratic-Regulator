import jax
import jax.numpy as jnp
from jax import lax
import matplotlib.pyplot as plt
from functools import partial
import time

# --- Configuration & Constants ---
m = 1.0
dt = 0.01
g = 9.81
mu = 0.1
tol_newton = 1e-5
max_iter_newton = 1000
tol_fixed_point = 1e-5
max_iter_fixed_point = 1000
T_end = 3.0

# --- Helper Math Functions ---

def prox_R0minus(x):
    """Projection onto negative reals (-infinity, 0]."""
    return jnp.minimum(x, 0.0)

def prox_CT(x, limit):
    """Projection onto Coulomb Friction Cone (Interval [-limit, limit] in 1D)."""
    return jnp.clip(x, -limit, limit)

# --- Physics Residuals & Jacobian ---

def R_x(x_vec, y_vec, qk, uk):
    """
    Dynamics Residual: R(x) = 0
    x_vec: [dq_x, dq_y, du_x, du_y] (Change in state)
    y_vec: [P_T, P_N] (Impulses)
    qk: Position at k
    uk: Velocity at k
    """
    dq = x_vec[0:2]
    du = x_vec[2:4]
    
    # Eq 1: q_{k+1} - q_k - dt * (u_{k+1}) = 0 
    # So: dq - dt * (du + uk) = 0
    res_pos = dq - dt * (du + uk)
    
    # Eq 2: M*du - Forces = 0
    # x-direction: m * du_x - P_T = 0
    res_vel_x = m * du[0] - y_vec[0]
    
    # y-direction: m * du_y + dt*m*g - P_N = 0
    res_vel_y = m * du[1] + dt * m * g - y_vec[1]
    
    return jnp.concatenate([res_pos, jnp.array([res_vel_x, res_vel_y])])

def J_x_fun(x_vec, y_vec, qk, uk):
    """
    Computes the Jacobian of the Dynamics Residual R_x with respect to x_vec 
    using JAX's forward-mode automatic differentiation (jacfwd).
    """
    # Define a helper function that only exposes x_vec for differentiation.
    # The other arguments (y_vec, qk, uk) are fixed/closed over.
    def residual_of_x(x):
        return R_x(x, y_vec, qk, uk)
        
    return jax.jacfwd(residual_of_x)(x_vec)

# --- Inner Solver: Newton's Method ---

def solve_newton(yk, qk, uk, x_init):
    """
    Solves R_x(x, yk, ...) = 0 for x using Newton's method.
    Compatible with lax.while_loop.
    """
    
    def cond_fun(state):
        x_curr, _, i, conv = state
        return jnp.logical_and(i < max_iter_newton, jnp.logical_not(conv))

    def body_fun(state):
        x_curr, _, i, _ = state
        
        resid = R_x(x_curr, yk, qk, uk)
        # UPDATED: Pass qk and uk to J_x_fun
        jac = J_x_fun(x_curr, yk, qk, uk) 
        
        # Newton step: x_new = x_curr - J^-1 * R
        delta = jnp.linalg.solve(jac, resid)
        x_new = x_curr - delta
        
        conv = jnp.linalg.norm(x_new - x_curr) < tol_newton
        return (x_new, x_curr, i + 1, conv)

    # State: (x_current, x_prev, iteration, converged)
    init_state = (x_init, x_init, 0, False)
    
    final_state = lax.while_loop(cond_fun, body_fun, init_state)
    return final_state[0]

# --- Outer Solver: Fixed Point Iteration (The Contact Solver) ---

def fixed_point_step_logic(x_curr, y_curr, qk, uk, r_T, r_N):
    """
    Performs one iteration of the Fixed Point scheme (Prox -> Newton).
    Returns (x_next, y_next).
    """
    # 1. Update Forces (y) based on Contact Laws (Proximal mappings)
    
    # Normal Impulse Update
    # dPN = -prox_R0minus(-y(2) + r_N * (q(2) + dq_y))
    val_N = -y_curr[1] + r_N * (qk[1] + x_curr[1])
    dPN_next = -prox_R0minus(val_N)
    
    # Friction Impulse Update
    # dPT = -prox_CT(-y(1) + r_T * (u(1) + du_x), mu * dPN)
    val_T = -y_curr[0] + r_T * (uk[0] + x_curr[2])
    dPT_next = -prox_CT(val_T, mu * dPN_next)
    
    y_next = jnp.array([dPT_next, dPN_next])
    
    # 2. Update State (x) using Newton solver given new Forces
    # We use x_curr as warm start
    x_next = solve_newton(y_next, qk, uk, x_curr)
    
    return x_next, y_next


# --- Implicit Differentiation Wrapper ---

@partial(jax.custom_jvp, nondiff_argnums=(2, 3, 4, 5))
def solve_contact_system(x_guess, y_guess, qk, uk, r_T, r_N):
    """
    Finds the fixed point (x*, y*) such that:
    (x*, y*) = fixed_point_step_logic(x*, y*, ...)
    
    This function is decorated with custom_jvp to perform implicit differentiation
    on the backward pass, avoiding unrolling the while loop.
    """
    
    def cond_fun(state):
        x, y, _, _, i, conv = state
        return jnp.logical_and(i < max_iter_fixed_point, jnp.logical_not(conv))

    def body_fun(state):
        x_prev, y_prev, _, _, i, _ = state
        
        x_next, y_next = fixed_point_step_logic(x_prev, y_prev, qk, uk, r_T, r_N)
        
        # Check convergence on x (state update)
        diff = jnp.linalg.norm(x_next - x_prev)
        conv = diff < tol_fixed_point
        
        return (x_next, y_next, x_prev, y_prev, i + 1, conv)

    # State: (x_curr, y_curr, x_prev, y_prev, iter, converged)
    init_state = (x_guess, y_guess, x_guess, y_guess, 0, False)
    
    final_state = lax.while_loop(cond_fun, body_fun, init_state)
    
    # Return the fixed point (x, y) and iteration count (for debug)
    return final_state[0], final_state[1], final_state[4]


@solve_contact_system.defjvp
def solve_contact_system_jvp(qk, uk, r_T, r_N, primals, tangents):
    """
    Custom JVP rule for the contact solver.
    Uses the Implicit Function Theorem on the combined residual equation.
    """
    x_guess, y_guess = primals
    dx_guess, dy_guess = tangents # These are usually zero or irrelevant for the fixed point itself
    
    # 1. Forward Pass: Solve for the fixed point exactly
    x_star, y_star, iters = solve_contact_system(x_guess, y_guess, qk, uk, r_T, r_N)
    
    # 2. Define the Combined Residual Function G(z, params) = 0
    # where z = [x, y] and params = [qk, uk]
    # The fixed point satisfies: z = Step(z). So Residual = z - Step(z).
    
    def combined_residual(z_flat, q_in, u_in):
        x_in = z_flat[:4]
        y_in = z_flat[4:]
        
        # One step of the fixed point logic
        x_out, y_out = fixed_point_step_logic(x_in, y_in, q_in, u_in, r_T, r_N)
        
        z_out = jnp.concatenate([x_out, y_out])
        return z_flat - z_out

    z_star = jnp.concatenate([x_star, y_star])
    
    # 3. Implicit Differentiation
    # We need dz_star / d(params).
    # Relation: J_G_z * dz + J_G_params * dparams = 0
    # dz = - (J_G_z)^-1 * (J_G_params * dparams)
    
    # A = dG/dz at z_star
    A = jax.jacobian(combined_residual, argnums=0)(z_star, qk, uk)
    
    # B_times_dparams = (dG/dq * dq_dot) + (dG/du * du_dot)
    
    dqk = tangents[2]
    duk = tangents[3]
    
    # Compute the RHS vector: (dG/dparams) @ dparams
    # We capture this by running jvp on the residual function with respect to params
    _, rhs_vec = jax.jvp(
        lambda q, u: combined_residual(z_star, q, u), 
        (qk, uk), 
        (dqk, duk)
    )
    
    # Solve linear system: A * dz = -rhs
    dz = jnp.linalg.solve(A, -rhs_vec)
    
    dx_out = dz[:4]
    dy_out = dz[4:]
    d_iters = 0.0 # Iterations are an integer, no gradient
    
    return (x_star, y_star, iters), (dx_out, dy_out, d_iters)

# --- Main Integration Step (Scan Body) ---

def step_fn(carrier, t_curr):
    q_curr, u_curr, x_warm, y_warm = carrier
    
    # 1. Compute Preconditioner (r_vec)
    # G_diag_T = 1/m, G_diag_N = 1/m
    G_diag_T = 1.0/m
    G_diag_N = 1.0/m
    
    # MATLAB: r_T = 1/G, r_N = 1/(dt*G)
    r_T_val = (1.0 / (dt * G_diag_T)) * dt
    r_N_val = 1.0 / (dt * G_diag_N)
    
    # 2. Solve Contact Problem
    x_sol, y_sol, iters = solve_contact_system(x_warm, y_warm, q_curr, u_curr, r_T_val, r_N_val)
    
    dq = x_sol[0:2]
    du = x_sol[2:4]
    
    # 3. Update State
    q_next = q_curr + dq
    u_next = u_curr + du
    
    # 4. Pack Carrier for next step
    new_carrier = (q_next, u_next, x_sol, y_sol)
    
    # 5. Output for storage
    output = {
        'q': q_next,
        'u': u_next,
        'PN': y_sol[1],
        'PT': y_sol[0],
        'iters': iters
    }
    
    return new_carrier, output

# --- Execution ---

def run_simulation():
    # Time setup
    tspan = jnp.arange(0, T_end + dt, dt)
    N = len(tspan)
    
    # Initial Conditions
    q0 = jnp.array([0.0, 1.0])
    u0 = jnp.array([1.0, 0.0])
    
    # Initial Guesses (Warm starts)
    x_init = jnp.zeros(4)
    y_init = jnp.zeros(2)
    
    init_carrier = (q0, u0, x_init, y_init)
    
    print("JIT Compiling and Running Simulation...")
    start_time = time.time()
    
    # Run Scan
    final_carrier, history = jax.lax.scan(step_fn, init_carrier, tspan)
    
    # Unpack History
    q_hist = history['q'].T
    u_hist = history['u'].T
    PN_hist = history['PN']
    PT_hist = history['PT']
    iter_hist = history['iters']
    
    # Prepend initial conditions for plotting to match MATLAB size
    q_full = jnp.hstack([q0[:, None], q_hist])
    u_full = jnp.hstack([u0[:, None], u_hist])
    # Pad forces with 0 for t=0
    PN_full = jnp.hstack([jnp.array([0.0]), PN_hist])
    PT_full = jnp.hstack([jnp.array([0.0]), PT_hist])
    t_full = tspan 
    
    end_time = time.time()
    print(f"Simulation complete. Time elapsed: {end_time - start_time:.4f}s")
    
    return t_full, q_full, u_full, PN_full, PT_full, iter_hist

# --- Plotting ---

def plot_results(t, q, u, PN, PT, iters):
    plt.figure(figsize=(12, 10))
    
    # 1. Trajectory
    plt.subplot(2, 2, 1)
    plt.plot(q[0, :], q[1, :], 'b-o', markersize=3, linewidth=1.0)
    plt.axhline(0, color='k', linewidth=2)
    plt.title('Trajectory (Position Space)')
    plt.xlabel('Distance X (m)')
    plt.ylabel('Height Y (m)')
    plt.axis('equal')
    plt.grid(True)
    plt.ylim([-0.1, max(q[1, :]) * 1.1])
    
    # 2. Velocities
    t_plot = jnp.hstack([jnp.array([0.0]), t]) if len(t) < q.shape[1] else t
    
    plt.subplot(2, 2, 2)
    plt.plot(t_plot, u[0, :], 'r-', linewidth=1.5, label='Horizontal v_x')
    plt.plot(t_plot, u[1, :], 'b--', linewidth=1.0, label='Vertical v_y')
    plt.title('Velocities')
    plt.legend(loc='best')
    plt.xlabel('Time (s)')
    plt.ylabel('m/s')
    plt.grid(True)
    
    # 3. Normal Percussion
    plt.subplot(2, 2, 3)
    plt.plot(t_plot, PN, 'g-', linewidth=1.5)
    plt.title('Normal Percussion (P_N)')
    plt.xlabel('Time (s)')
    plt.ylabel('Impulse (N*s)')
    plt.grid(True)
    
    # 4. Friction Percussion
    plt.subplot(2, 2, 4)
    plt.plot(t_plot, PT, 'm-', linewidth=1.5)
    plt.title('Friction Percussion (P_T)')
    plt.xlabel('Time (s)')
    plt.ylabel('Impulse (N*s)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('simulation_results.png')
    print("Results saved to 'simulation_results.png'")

if __name__ == "__main__":
    # Run
    t, q, u, PN, PT, iters = run_simulation()
    
    # Verify Differentiability (Simple check)
    print("\nVerifying Differentiability (computing gradient of final x-position w.r.t initial y-velocity)...")
    
    def final_x_pos(u0_y):
        carrier = (jnp.array([0.0, 1.0]), jnp.array([1.0, u0_y]), jnp.zeros(4), jnp.zeros(2))
        _, hist = jax.lax.scan(step_fn, carrier, jnp.arange(0, T_end + dt, dt))
        return hist['q'][-1, 0] # Final x position

    # This will crash if implicit diff isn't working correctly or custom_jvp is wrong
    grad_fn = jax.jit(jax.grad(final_x_pos))
    g_val = grad_fn(0.0)
    print(f"Gradient computed successfully: {g_val}")
    
    # Plot
    plot_results(t, q, u, PN, PT, iters)