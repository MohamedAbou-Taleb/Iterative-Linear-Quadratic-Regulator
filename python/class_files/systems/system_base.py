import jax
import jax.numpy as jnp
from jax import jit, grad, jacfwd, hessian, lax
from jax.scipy.linalg import lu_factor, lu_solve
from abc import ABC, abstractmethod
from typing import Callable, Tuple
from functools import partial

# --- Physics Helper Functions (JIT-compiled globally) ---
@jit
def prox_R0minus(x): 
    return jnp.minimum(x, 0.0)

@jit
def prox_R0minus_smooth(x, epsilon=1.0): 
    # Smooth approximation of -max(0, -x) = min(x, 0)
    # Using softplus-like smoothing: -epsilon * log(1 + exp(-x/epsilon))
    return -epsilon * lax.log(1.0 + lax.exp(-x / epsilon))

@jit
def prox_CT(x, limit): 
    return jnp.clip(x, -limit, limit)

class System(ABC):
    """
    JAX-based abstract base class for iLQR with support for Frictional Contact.
    
    Supports standard smooth integrators (RK4, Euler) and a specific
    Position-Based Contact-Implicit Backward Euler scheme.
    """
    
    def __init__(self, 
                 n_q: int, # Config dimension
                 n_v: int, # Velocity dimension (usually same as n_q)
                 n_u: int, # Control dimension
                 n_c: int, # Number of contacts
                 dt: float, 
                 integrator: str = 'rk4',
                 mu: jnp.ndarray = jnp.array([0.0]),  # Friction coefficient
                 e_restitution: jnp.ndarray = jnp.array([0.0]), # Coefficient of restitution per contact
                 smooth_epsilon: float = 1.0): # Smoothing parameter for gradients
        
        self.n_q = n_q
        self.n_v = n_v
        self.n_x = n_q + n_v
        self.n_u = n_u
        self.n_c = n_c
        self.dt = dt
        self.mu = mu
        self.e_restitution = e_restitution
        self.epsilon = smooth_epsilon
        self.integrator_name = integrator

        # --- Defines ---
        
        # Select Integrator
        if integrator in ['contact_euler', 'backward_euler']:
            self._f_fcn = self._contact_euler_integrator
            # For contact solver, we rely on jacfwd of the custom_jvp function
            _f_x = jacfwd(self._f_fcn, argnums=0)
            _f_u = jacfwd(self._f_fcn, argnums=1)

        elif integrator in ['elastic_contact_euler']:
            self._f_fcn = self._elastic_contact_euler_integrator
            # For contact solver, we rely on jacfwd of the custom_jvp function
            _f_x = jacfwd(self._f_fcn, argnums=0)
            _f_u = jacfwd(self._f_fcn, argnums=1)
        
            
        elif integrator in ['rk4', 'midpoint', 'euler']:
            # Standard smooth integrators
            if integrator == 'rk4':
                self._f_fcn = self._rk4_integrator
            elif integrator == 'midpoint':
                self._f_fcn = self._midpoint_integrator
            elif integrator == 'euler':
                self._f_fcn = self._euler_integrator
                
            _f_x = jacfwd(self._f_fcn, argnums=0)
            _f_u = jacfwd(self._f_fcn, argnums=1)
        else:
            raise ValueError(f"Unknown integrator: {integrator}")

        # --- JIT Compile Dynamics & Derivatives ---
        self.f_fcn = jit(self._f_fcn)
        self.f_x_fcn = jit(_f_x)
        self.f_u_fcn = jit(_f_u)
        
        # --- Cost Derivatives ---
        self.l_fcn = jit(self._l_fcn)
        self.l_x_fcn = jit(grad(self._l_fcn, argnums=0))
        self.l_u_fcn = jit(grad(self._l_fcn, argnums=1))
        self.l_xx_fcn = jit(hessian(self._l_fcn, argnums=0))
        self.l_uu_fcn = jit(hessian(self._l_fcn, argnums=1))
        self.l_ux_fcn = jit(jacfwd(grad(self._l_fcn, argnums=1), argnums=0))
        
        self.l_f_fcn = jit(self._l_f_fcn)
        self.l_f_x_fcn = jit(grad(self._l_f_fcn, argnums=0))
        self.l_f_xx_fcn = jit(hessian(self._l_f_fcn, argnums=0))

    # =========================================================
    # Abstract Methods (Subclasses MUST implement)
    # =========================================================

    @abstractmethod
    def _mass_matrix(self, q: jnp.ndarray) -> jnp.ndarray:
        """Returns M(q) of shape (n_v, n_v)."""
        pass

    @abstractmethod
    def _generalized_forces(self, q: jnp.ndarray, v: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Returns h(q, v, u) of shape (n_v,)."""
        pass

    @abstractmethod
    def _contact_jacobian(self, q: jnp.ndarray) -> jnp.ndarray:
        """Returns W(q) of shape (n_v, 2*n_c). Columns: [T1, N1, T2, N2...]"""
        pass
        
    @abstractmethod
    def _gap_function(self, q: jnp.ndarray) -> jnp.ndarray:
        """Returns gap vector g(q) of shape (n_c,)."""
        pass
        
    @abstractmethod
    def _contact_velocity_function(self, q: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Returns tangential contact velocity gamma(q, v) of shape (n_c,)."""
        pass

    # =========================================================
    # Smooth Dynamics
    # =========================================================
    
    def _f_cont_fcn(self, x, u):
        q, v = x[:self.n_q], x[self.n_q:]
        M = self._mass_matrix(q)
        h = self._generalized_forces(q, v, u)
        v_dot = jnp.linalg.solve(M, h)
        return jnp.concatenate([v, v_dot])

    def _euler_integrator(self, x, u):
        x_dot = self._f_cont_fcn(x, u)
        return x + x_dot * self.dt

    def _midpoint_integrator(self, x, u):
        k1 = self._f_cont_fcn(x, u)
        x_mid = x + (self.dt / 2.0) * k1
        k2 = self._f_cont_fcn(x_mid, u)
        return x + self.dt * k2

    def _rk4_integrator(self, x, u):
        k1 = self._f_cont_fcn(x, u)
        k2 = self._f_cont_fcn(x + self.dt/2 * k1, u)
        k3 = self._f_cont_fcn(x + self.dt/2 * k2, u)
        k4 = self._f_cont_fcn(x + self.dt * k3, u)
        return x + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # =========================================================
    # Position-Based Contact Implicit Integrator
    # =========================================================

    def _contact_euler_integrator(self, x_state, u_control):
        qk = x_state[:self.n_q]
        vk = x_state[self.n_q:]
        
        # 1. Compute Effective Mass / Stiffness for conditioning (Approximation at k)
        M_k = self._mass_matrix(qk)
        W_k = self._contact_jacobian(qk) 
        inv_M_k = jnp.linalg.inv(M_k) 
        G = W_k.T @ inv_M_k @ W_k
        diag_G = jnp.diag(G)
        
        # Scale: dP is Impulse. 
        # r * gap = [1/(dt/m)] * [m] = [m * m / dt] -> Mass * Velocity = Momentum/Impulse
        r = 1.0 / (self.dt * diag_G)
        
        # 2. Solve Contact
        # Initial guess for impulses (dP)
        dP_guess = jnp.zeros(W_k.shape[1])
        
        # Pass 'self' explicitly because the custom_jvp decorated function 
        # is treated as a static function by JAX
        q_next, v_next, _ = self._solve_contact_dynamics(
            self, qk, vk, u_control, r, dP_guess
        )
        
        return jnp.concatenate([q_next, v_next])

    @partial(jax.custom_jvp, nondiff_argnums=(0,))
    def _solve_contact_dynamics(self, qk, vk, uk, r, dP_guess):
        """
        Primal solver: Nested Fixed Point (Contact) -> Newton (Dynamics)
        Uses dP (Impulse) as the contact variable.
        """

        # Define the residual function for the implicit dynamics
        def dyn_residual(z_flat, dP_val):
            qk1, vk1 = z_flat[:self.n_q], z_flat[self.n_q:]
            
            # Evaluate dynamics at NEW state (k+1)
            M = self._mass_matrix(qk1)
            h = self._generalized_forces(qk1, vk1, uk)
            W = self._contact_jacobian(qk1)
            
            # 1. Position Integration: qk1 = qk + dt * vk1
            res_q = qk1 - qk - self.dt * vk1
            
            # 2. Velocity Integration: M(vk1 - vk) = dt * h + W @ dP
            res_v = M @ (vk1 - vk) - self.dt * h - (W @ dP_val)
            
            return jnp.concatenate([res_q, res_v])
        
        # Define the Jacobian of the residual w.r.t z (state)
        # Returns d(residual)/dz
        dyn_residual_jac = jacfwd(dyn_residual, argnums=0)
        
        # --- Optimization: Pre-compute Fixed Jacobian and LU Factors ---
        # We evaluate the Jacobian ONCE at the explicit Euler prediction
        # and reuse it for all inner Newton steps within this timestep.
        z_guess_init = jnp.concatenate([qk + self.dt*vk, vk])
        J_fixed = dyn_residual_jac(z_guess_init, dP_guess)
        # Factorize once. 'lu_piv' is a tuple (LU_matrix, pivots)
        fixed_lu_piv = lu_factor(J_fixed)

        # --- Inner Newton Solver (Modified Newton using Fixed Jacobian) ---
        # Note: No @custom_jvp needed here; differentiability is handled by the outer function.
        def solve_dynamics_newton(dP_curr, z_init):
            
            def cond_fun(state):
                z, iter_i, err = state
                # Convergence criteria: max 100 iterations OR error < 1e-5
                return jnp.logical_and(iter_i < 100, err > 1e-5)

            def body_fun(state):
                z, iter_i, _ = state
                res = dyn_residual(z, dP_curr)
                
                # OPTIMIZED: Use re-usable LU factors
                # This replaces the expensive solve(jac(z), res)
                delta = -lu_solve(fixed_lu_piv, res)
                
                z_new = z + delta
                err_new = jnp.linalg.norm(res)
                return (z_new, iter_i + 1, err_new)
            
            # Calculate initial error
            init_res = dyn_residual(z_init, dP_curr)
            init_err = jnp.linalg.norm(init_res)
            
            final_state = lax.while_loop(cond_fun, body_fun, (z_init, 0, init_err))
            z_star = final_state[0]
            
            return z_star

        # --- Outer Fixed Point Solver for Contact Impulses (dP) ---
        
        def outer_loop_cond(state):
            dP, z, iter_c, err = state
            # Limit max iterations
            return jnp.logical_and(iter_c < 50, err > 1e-5)
            
        def outer_loop_body(state):
            dP_curr, z_prev, iter_c, _ = state
            
            # 1. Solve Dynamics using pre-computed LU factors
            z_star = solve_dynamics_newton(dP_curr, z_prev)
            qk1 = z_star[:self.n_q]
            vk1 = z_star[self.n_q:]
            
            # 2. Update Contact Impulses (Fixed Point / Prox)
            gap_val = self._gap_function(qk1)
            W_next = self._contact_jacobian(qk1)
            rel_vel = W_next.T @ vk1
            
            dP_new_list = []
            for i in range(self.n_c):
                idx_t = 2*i
                idx_n = 2*i + 1
                
                # Tangential stiffness requires dt scaling to map velocity -> impulse
                r_t = r[idx_t]*self.dt 
                r_n = r[idx_n]
                mu_i = self.mu[i]
                # --- Normal Update ---
                target_n = dP_curr[idx_n] - r_n * gap_val[i]
                dP_n = -prox_R0minus(-target_n)
                
                # --- Tangent Update ---
                vt = rel_vel[idx_t]
                limit = mu_i * dP_n
                dP_t = -prox_CT(-dP_curr[idx_t] + r_t * vt, limit)
                
                dP_new_list.append(dP_t)
                dP_new_list.append(dP_n)
                
            dP_next = jnp.array(dP_new_list)
            
            # Check convergence
            err = jnp.linalg.norm(dP_next - dP_curr)
            
            return (dP_next, z_star, iter_c + 1, err)

        # Initial guess for state (Euler step)
        init_state = (dP_guess, z_guess_init, 0, 1.0)
        final_state = lax.while_loop(outer_loop_cond, outer_loop_body, init_state)
        
        final_dP = final_state[0]
        
        # Ensure final state is consistent (run one last dynamics solve)
        z_star_fin = solve_dynamics_newton(final_dP, final_state[1])
        qk1_fin, vk1_fin = z_star_fin[:self.n_q], z_star_fin[self.n_q:]
        
        return qk1_fin, vk1_fin, final_dP

    @_solve_contact_dynamics.defjvp
    def _solve_contact_dynamics_jvp(self, primals, tangents):
        # FIXED: 'self' is unpacked from primals[0]
        qk, vk, uk, r, dP_guess = primals
        dq, dv, du, dr, ddP = tangents 
        
        # 1. Run Primal Solver
        qk1, vk1, dP_star = self._solve_contact_dynamics(self, qk, vk, uk, r, dP_guess)
        
        # 2. Total Residual Implicit Differentiation
        # We define the coupled residual R(z_total, params) = 0
        # where z_total = [qk1, vk1, dP]
        
        def total_smooth_residual(z_total, q_old, v_old, u_old, r_old):
            q_new = z_total[:self.n_q]
            v_new = z_total[self.n_q:self.n_x]
            dP = z_total[self.n_x:]
            
            # --- A. Dynamics Residual (Implicit Euler) ---
            M = self._mass_matrix(q_new)
            h = self._generalized_forces(q_new, v_new, u_old)
            W = self._contact_jacobian(q_new)
            
            res_q = q_new - q_old - self.dt * v_new
            # FIXED: Removed self.dt * (W @ dP)
            res_v = M @ (v_new - v_old) - self.dt * h - (W @ dP)
            
            # --- B. Contact Smooth Residual ---
            gap_val = self._gap_function(q_new)
            rel_vel = W.T @ v_new
            
            dP_smooth_list = []
            for i in range(self.n_c):
                idx_t = 2*i
                idx_n = 2*i + 1
                
                # FIX: Ensure gradient logic matches Primal loop logic for r_t
                r_t = r_old[idx_t] * self.dt 
                r_n = r_old[idx_n]
                mu_i = self.mu[i]
                # Smooth Normal
                target_n = dP[idx_n] - r_n * gap_val[i]
                dP_n_new = -prox_R0minus_smooth(-target_n, self.epsilon)
                
                # Smooth Tangent
                vt = rel_vel[idx_t]
                target_t = dP[idx_t] - r_t * vt
                limit = mu_i * dP_n_new
                dP_t_new = limit * jnp.tanh(target_t / (limit + 1e-6))
                
                dP_smooth_list.append(dP_t_new)
                dP_smooth_list.append(dP_n_new)
            
            res_dP = dP - jnp.array(dP_smooth_list)
            
            return jnp.concatenate([res_q, res_v, res_dP])

        # IFT Application
        # F(z, p) = 0  => dz/dp = - (dF/dz)^-1 (dF/dp)
        
        z_star_total = jnp.concatenate([qk1, vk1, dP_star])
        params = (qk, vk, uk, r)
        d_params = (dq, dv, du, dr)
        
        # 1. Jacobian w.r.t solution vars (q, v, dP)
        J_z = jacfwd(total_smooth_residual, argnums=0)(z_star_total, *params)
        
        # 2. RHS: - dF/dp * dp (computed via JVP)
        _, rhs_val = jax.jvp(
            lambda *p: total_smooth_residual(z_star_total, *p),
            params,
            d_params
        )
        
        # 3. Solve linear system for total variation
        d_z_total = -jnp.linalg.solve(J_z , rhs_val)
        
        # Unpack derivatives
        dqk1 = d_z_total[:self.n_q]
        dvk1 = d_z_total[self.n_q:self.n_x]
        ddP_star = d_z_total[self.n_x:]
        
        return (qk1, vk1, dP_star), (dqk1, dvk1, ddP_star)
    
    def _elastic_contact_euler_integrator(self, x_state, u_control):
        qk = x_state[:self.n_q]
        vk = x_state[self.n_q:]
        
        # 1. Compute Effective Mass / Stiffness for conditioning (Approximation at k)
        M_k = self._mass_matrix(qk)
        W_k = self._contact_jacobian(qk) 
        inv_M_k = jnp.linalg.inv(M_k) 
        G = W_k.T @ inv_M_k @ W_k
        diag_G = jnp.diag(G)
        
        # Scale: dP is Impulse. 
        # r * gap = [1/(dt/m)] * [m] = [m * m / dt] -> Mass * Velocity = Momentum/Impulse
        r = 1.0 / (self.dt * diag_G)
        
        # 2. Solve Contact
        # Initial guess for impulses (dP)
        dP_guess = jnp.zeros(W_k.shape[1])
        
        # Pass 'self' explicitly because the custom_jvp decorated function 
        # is treated as a static function by JAX
        q_next, v_next, _ = self._solve_elastic_contact_dynamics(
            self, qk, vk, u_control, r, dP_guess
        )
        
        return jnp.concatenate([q_next, v_next])

    @partial(jax.custom_jvp, nondiff_argnums=(0,))
    def _solve_elastic_contact_dynamics(self, qk, vk, uk, r, dP_guess):
        """
        Primal solver: Nested Fixed Point (Contact) -> Newton (Dynamics)
        Uses dP (Impulse) as the contact variable.
        """

        # Define the residual function for the implicit dynamics
        def dyn_residual(z_flat, dP_val):
            qk1, vk1 = z_flat[:self.n_q], z_flat[self.n_q:]
            
            # Evaluate dynamics at NEW state (k+1)
            M = self._mass_matrix(qk1)
            h = self._generalized_forces(qk1, vk1, uk)
            W = self._contact_jacobian(qk1)
            
            # 1. Position Integration: qk1 = qk + dt * vk1
            res_q = qk1 - qk - self.dt * vk1
            
            # 2. Velocity Integration: M(vk1 - vk) = dt * h + W @ dP
            res_v = M @ (vk1 - vk) - self.dt * h - (W @ dP_val)
            
            return jnp.concatenate([res_q, res_v])
        
        # Define the Jacobian of the residual w.r.t z (state)
        # Returns d(residual)/dz
        dyn_residual_jac = jacfwd(dyn_residual, argnums=0)
        
        # --- Optimization: Pre-compute Fixed Jacobian and LU Factors ---
        # We evaluate the Jacobian ONCE at the explicit Euler prediction
        # and reuse it for all inner Newton steps within this timestep.
        z_guess_init = jnp.concatenate([qk + self.dt*vk, vk])
        J_fixed = dyn_residual_jac(z_guess_init, dP_guess)
        # Factorize once. 'lu_piv' is a tuple (LU_matrix, pivots)
        fixed_lu_piv = lu_factor(J_fixed)

        # --- Inner Newton Solver (Modified Newton using Fixed Jacobian) ---
        # Note: No @custom_jvp needed here; differentiability is handled by the outer function.
        def solve_dynamics_newton(dP_curr, z_init):
            
            def cond_fun(state):
                z, iter_i, err = state
                # Convergence criteria: max 100 iterations OR error < 1e-5
                return jnp.logical_and(iter_i < 100, err > 1e-5)

            def body_fun(state):
                z, iter_i, _ = state
                res = dyn_residual(z, dP_curr)
                
                # OPTIMIZED: Use re-usable LU factors
                # This replaces the expensive solve(jac(z), res)
                delta = -lu_solve(fixed_lu_piv, res)
                
                z_new = z + delta
                err_new = jnp.linalg.norm(res)
                return (z_new, iter_i + 1, err_new)
            
            # Calculate initial error
            init_res = dyn_residual(z_init, dP_curr)
            init_err = jnp.linalg.norm(init_res)
            
            final_state = lax.while_loop(cond_fun, body_fun, (z_init, 0, init_err))
            z_star = final_state[0]
            
            return z_star

        # --- Outer Fixed Point Solver for Contact Impulses (dP) ---
        
        def outer_loop_cond(state):
            dP, z, iter_c, err = state
            # Limit max iterations
            return jnp.logical_and(iter_c < 500, err > 1e-5)
            
        def outer_loop_body(state):
            dP_curr, z_prev, iter_c, _ = state
            
            # 1. Solve Dynamics using pre-computed LU factors
            z_star = solve_dynamics_newton(dP_curr, z_prev)
            qk1 = z_star[:self.n_q]
            vk1 = z_star[self.n_q:]
            
            # 2. Update Contact Impulses (Fixed Point / Prox)
            qk_mid = qk + self.dt*vk/2
            gap_val = self._gap_function(qk_mid)  

            W_mid = self._contact_jacobian(qk_mid)
            W_next = self._contact_jacobian(qk1)
            rel_vel = W_next.T @ vk1
            
            dP_new_list = []
            for i in range(self.n_c):
                idx_t = 2*i
                idx_n = 2*i + 1
                
                # Tangential stiffness requires dt scaling to map velocity -> impulse
                r_t = r[idx_t]*self.dt 
                r_n = r[idx_n]*self.dt
                e_restitution = self.e_restitution[i]
                mu_i = self.mu[i]
                # --- Normal Update ---
                gamma_N_next = W_next.T[idx_n] @ vk1
                xi_N = gamma_N_next + e_restitution * self._contact_jacobian(qk_mid)[:, idx_n].T @ vk
                target_n = dP_curr[idx_n] - r_n * xi_N
                dP_n_contact = -prox_R0minus(-dP_curr[idx_n] + r_n*xi_N)
                dP_n = jnp.where(gap_val[i] > 0.0, 0.0, dP_n_contact)
                
                # --- Tangent Update ---
                vt = rel_vel[idx_t]
                limit = mu_i * dP_n
                dP_t = -prox_CT(-dP_curr[idx_t] + r_t * vt, limit)
                
                dP_new_list.append(dP_t)
                dP_new_list.append(dP_n)
                
            dP_next = jnp.array(dP_new_list)
            
            # Check convergence
            err = jnp.linalg.norm(dP_next - dP_curr)
            
            return (dP_next, z_star, iter_c + 1, err)

        # Initial guess for state (Euler step)
        init_state = (dP_guess, z_guess_init, 0, 1.0)
        final_state = lax.while_loop(outer_loop_cond, outer_loop_body, init_state)
        
        final_dP = final_state[0]
        
        # Ensure final state is consistent (run one last dynamics solve)
        z_star_fin = solve_dynamics_newton(final_dP, final_state[1])
        qk1_fin, vk1_fin = z_star_fin[:self.n_q], z_star_fin[self.n_q:]
        
        return qk1_fin, vk1_fin, final_dP

    @_solve_elastic_contact_dynamics.defjvp
    def _solve_elastic_contact_dynamics_jvp(self, primals, tangents):
        # FIXED: 'self' is unpacked from primals[0]
        qk, vk, uk, r, dP_guess = primals
        dq, dv, du, dr, ddP = tangents 
        
        # 1. Run Primal Solver
        qk1, vk1, dP_star = self._solve_elastic_contact_dynamics(self, qk, vk, uk, r, dP_guess)
        
        # 2. Total Residual Implicit Differentiation
        # We define the coupled residual R(z_total, params) = 0
        # where z_total = [qk1, vk1, dP]
        
        def total_smooth_residual(z_total, q_old, v_old, u_old, r_old):
            q_new = z_total[:self.n_q]
            v_new = z_total[self.n_q:self.n_x]
            dP = z_total[self.n_x:]
            
            # --- A. Dynamics Residual (Implicit Euler) ---
            M = self._mass_matrix(q_new)
            h = self._generalized_forces(q_new, v_new, u_old)
            W = self._contact_jacobian(q_new)
            
            res_q = q_new - q_old - self.dt * v_new
            # FIXED: Removed self.dt * (W @ dP)
            res_v = M @ (v_new - v_old) - self.dt * h - (W @ dP)
            
            # --- B. Contact Smooth Residual ---
            gap_val = self._gap_function(q_new)
            rel_vel = W.T @ v_new
            
            dP_smooth_list = []
            for i in range(self.n_c):
                idx_t = 2*i
                idx_n = 2*i + 1
                
                # FIX: Ensure gradient logic matches Primal loop logic for r_t
                r_t = r_old[idx_t] * self.dt 
                r_n = r_old[idx_n] * self.dt
                mu_i = self.mu[i]
                
                # Smooth Normal
                target_n = dP[idx_n] - r_n * gap_val[i]
                # dP_n_new = -prox_R0minus_smooth(-target_n, self.epsilon)
                dP_n_new = -prox_R0minus(-target_n)
                
                # Smooth Tangent
                vt = rel_vel[idx_t]
                target_t = dP[idx_t] - r_t * vt
                limit = mu_i * dP_n_new
                dP_t_new = limit * jnp.tanh(target_t / (limit + 1e-6))
                
                dP_smooth_list.append(dP_t_new)
                dP_smooth_list.append(dP_n_new)
            
            res_dP = dP - jnp.array(dP_smooth_list)
            
            return jnp.concatenate([res_q, res_v, res_dP])

        # IFT Application
        # F(z, p) = 0  => dz/dp = - (dF/dz)^-1 (dF/dp)
        
        z_star_total = jnp.concatenate([qk1, vk1, dP_star])
        params = (qk, vk, uk, r)
        d_params = (dq, dv, du, dr)
        
        # 1. Jacobian w.r.t solution vars (q, v, dP)
        J_z = jacfwd(total_smooth_residual, argnums=0)(z_star_total, *params)
        
        # 2. RHS: - dF/dp * dp (computed via JVP)
        _, rhs_val = jax.jvp(
            lambda *p: total_smooth_residual(z_star_total, *p),
            params,
            d_params
        )
        
        # 3. Solve linear system for total variation
        d_z_total = -jnp.linalg.solve(J_z , rhs_val)
        
        # Unpack derivatives
        dqk1 = d_z_total[:self.n_q]
        dvk1 = d_z_total[self.n_q:self.n_x]
        ddP_star = d_z_total[self.n_x:]
        
        return (qk1, vk1, dP_star), (dqk1, dvk1, ddP_star)


    # =========================================================
    # Cost Functions
    # =========================================================
    @abstractmethod
    def _l_fcn(self, x, u): pass
        
    @abstractmethod
    def _l_f_fcn(self, x): pass