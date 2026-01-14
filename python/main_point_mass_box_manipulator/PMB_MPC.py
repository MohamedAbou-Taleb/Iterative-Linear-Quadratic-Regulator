import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time

# Import your custom classes from the other files
from class_files.systems.system_base import System
from class_files.systems.point_mass_box_manipulator_sys import MyPointMassBoxManipulator # <-- Import MyDoublePendulum
from class_files.box_MPC import box_MPC
from class_files.animations.animation_point_mass_box import AnimationPointMassBox
import casadi as ca
from casadi_low_level import CasadiLowLevelController

dt = 0.0001
dt = 0.001
dt_control = 0.01*10
control_ratio = int(dt_control / dt)
box_width = 0.5
box_height = 0.3
ball_radius = 0.05
x_box_target = jnp.array([1.0, 5*box_height / 2, 0.0, 0.0])

R = jnp.diag(jnp.array([1.0, 1.0, 1.0, 1.0]))
Q_box = jnp.diag(jnp.array([1.0, 1.0, 1.0, 1.0]))
Q_f = jnp.diag(jnp.array([1.0, 1.0, 1.0, 1.0]))
RN1 = 1.0
RN2 = 1.0
RN1_f = 1.0
RN2_f = 1.0
m_box = 0.5
m_ball = 1
reg_friction = jnp.array([1e-2, 1e-2, 1e-2])
# mu = jnp.array([0.3, 0.3, 0.0])
mu = jnp.array([0.5, 0.5, 0.0])*1
# --- Instantiate System ---
manipulator = MyPointMassBoxManipulator(
    dt=dt,
    box_target_state=x_box_target,
    R=R,
    Q_box=Q_box,
    RN1=RN1,
    RN2=RN2,
    Q_f=Q_f,
    RN1_f=RN1_f,
    RN2_f=RN2_f,
    integrator="rk4",
    box_height=box_height,
    box_width=box_width,
    ball_radius=ball_radius,
    m_box=m_box,
    m_ball=m_ball,
    g=9.81,
    mu=mu,
    reg_friction=reg_friction,
)


S = jnp.eye(6,4)
D = jnp.array([[0.0, 1.0, 0.0, -1.0], [1.0, 0.0, 1.0, 0.0]])
M = manipulator._mass_matrix(jnp.zeros(manipulator.n_q))
W = manipulator._contact_jacobian(jnp.zeros(manipulator.n_q))[:, 0:4]
h = manipulator._generalized_forces(jnp.zeros(manipulator.n_q), jnp.zeros(manipulator.n_v), jnp.zeros(manipulator.n_u))
A = jnp.block([ [M, -S, -W], [W.T, jnp.zeros((4, 8))] ])
C = jnp.hstack( [jnp.zeros((2,4)), jnp.eye(2,2)] )
epsilon = 1e-3
R_tau = jnp.diag(jnp.array([1.0, 1.0, 1.0, 1.0]))*1e-1
b = jnp.hstack( [h, jnp.zeros(4)] )


    


# Q = jnp.diag(jnp.array([10.0, 100.0, 10.0 , 100.0]))
# R = jnp.diag(jnp.array([0.1, 0.1]))*1e-3
# Q_f = jnp.diag(jnp.array([100.0, 100.0, 100.0, 100.0]))

Q = jnp.diag(jnp.array([10.0, 100.0, 10.0 , 100.0]))
R = jnp.diag(jnp.array([0.1, 0.1]))*1e2
Q_f = jnp.diag(jnp.array([100.0, 100.0, 100.0, 100.0]))

T_horizon = 1.0
# T_horizon = 0.2

box_MPC = box_MPC(point_mass_box_sys=manipulator, T_horizon=T_horizon,
                    Q=Q, R=R, Q_f=Q_f)


Q_box_acc = jnp.diag(jnp.array([10.0, 100.0]))
R_box_force = jnp.diag(jnp.array([0.1, 0.1]))

import numpy as np
A_np = np.array(A)
b_np = np.array(b)

# Instantiate the controller
casadi_controller = CasadiLowLevelController(
    manipulator=manipulator,
    box_MPC=box_MPC,
    Q_box_acc=Q_box_acc,
    R_box_force=R_box_force,
    R_tau=R_tau,
    C=C,
    epsilon=epsilon
)

def cost_function(xla, u_box_ref, ddqdt_box_ref):
    ddqdt = xla[0:6]
    tau = xla[6:10]
    _lambda = xla[10:14]

    d_u_box = box_MPC.u_box_of_lambda(_lambda) - u_box_ref
    d_ddqdt_box = C @ ddqdt - ddqdt_box_ref
    cost = 0.5 * (d_u_box.T @ R_box_force @ d_u_box + d_ddqdt_box.T @ Q_box_acc @ d_ddqdt_box +
                  tau.T @ R_tau @ tau + epsilon * _lambda.T @ _lambda) 
    return cost

def constraint_eq(xla):
    ddqdt = xla[0:6]
    tau = xla[6:10]
    _lambda = xla[10:14]
    return A @ jnp.hstack( [ddqdt, tau, _lambda] ) - b

def lagrangian(xla, u_box_ref, ddqdt_box_ref):
    ddqdt = xla[0:6]
    tau = xla[6:10]
    _lambda = xla[10:14]
    la = xla[14:]
    L = cost_function(xla, u_box_ref, ddqdt_box_ref) + la.T @ constraint_eq(xla)
    return L
def lagrangian_grad(xla, u_box_ref, ddqdt_box_ref):
    return jax.grad(lagrangian, argnums=0)(xla, u_box_ref, ddqdt_box_ref)

lagrangian_grad_test = lagrangian_grad(jnp.zeros(14 + 10), jnp.array([0.0, 0.0]), jnp.array([0.0, 0.0]))
print('lagrangian_grad_test:', lagrangian_grad_test)

def lagrangian_hess(xla, u_box_ref, ddqdt_box_ref):
    return jax.hessian(lagrangian, argnums=0)(xla, u_box_ref, ddqdt_box_ref)

xla = -jnp.linalg.solve(lagrangian_hess(jnp.zeros(14 + 10), jnp.array([0.0, 0.0]), jnp.array([0.0, 0.0])),
                        lagrangian_grad(jnp.zeros(14 + 10), jnp.array([0.0, 0.0]), jnp.array([0.0, 0.0])))
print('xla:', xla)


def _solve_optim_problem(u_box_ref, ddqdt_box_ref):
    xla = -jnp.linalg.solve(lagrangian_hess(jnp.zeros(14 + 10), u_box_ref, ddqdt_box_ref),
                        lagrangian_grad(jnp.zeros(14 + 10), u_box_ref, ddqdt_box_ref))
    ddqdt = xla[0:6]
    tau = xla[6:10]
    _lambda = xla[10:14]
    return ddqdt, tau, _lambda
solve_optim_problem = jax.jit(_solve_optim_problem)
ddqdt, tau, _lambda = solve_optim_problem(jnp.array([0.0, 0.0]), jnp.array([0.0, 0.0]))
print('ddqdt:', ddqdt)
print('tau:', tau)
print('_lambda:', _lambda)

# W_full = manipulator._contact_jacobian(jnp.zeros(manipulator.n_q))
W_full = W
def continous_dynamics(x, u):
    q = x[0:6]
    v = x[6:12]
    A = jnp.block( [[M, -W_full], [-W_full.T, jnp.zeros((4, 4))]]  )
    b = jnp.hstack( [h + S @ u, jnp.zeros(4)] )
    ddqdt = jnp.linalg.solve(A, b)[:manipulator.n_q]
    dxdt = jnp.hstack([v, ddqdt])
    return dxdt

def integrator(x, u):
        k1 = continous_dynamics(x, u)
        k2 = continous_dynamics(x + dt / 2 * k1, u)
        k3 = continous_dynamics(x + dt / 2 * k2, u)
        k4 = continous_dynamics(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


manipulator_sim = MyPointMassBoxManipulator(dt=dt, 
                                            box_target_state=x_box_target, 
                                            R=R, Q_box=Q_box, RN1=RN1, RN2=RN2,
                                            Q_f=Q_f, RN1_f=RN1_f, RN2_f=RN2_f,
                                            integrator='moreau',
                                            box_height=box_height,
                                            box_width=box_width,
                                            ball_radius=ball_radius,
                                            m_box=m_box,
                                            m_ball=m_ball,
                                            mu=mu) # mu=0.0 for box-floor to slide

q_0 = jnp.array([- 10*manipulator.ball_radius - manipulator.box_width/2, manipulator.box_height / 2,
                 manipulator.ball_radius + manipulator.box_width/2, manipulator.box_height / 2,
                 0.0,  manipulator.box_height / 2])
v_0 = jnp.zeros(6)
x_0 = jnp.hstack([q_0, v_0])

T_sim = 8.0
tspan_sim = jnp.arange(0, T_sim + box_MPC.dt, box_MPC.dt)
N_sim = len(tspan_sim) - 1
x_current = x_0

# Initialize storage arrays
X = jnp.zeros((manipulator.n_x, N_sim + 1))
U = jnp.zeros((manipulator.n_u, N_sim))
Lambdas = jnp.zeros((4, N_sim))  # <-- Initialize storage for lambda
uk_box = jnp.array([0.0, 0.0])
uk_val = np.array([0.0, 0.0, 0.0, 0.0])
ddqdt_val = np.zeros(6)
lambda_val = np.zeros(4)
X = X.at[:, 0].set(x_current)

print("Starting simulation loop...")

for k in range(N_sim):
    # Extract state components
    q = x_current[0:6]
    v = x_current[6:12]
    q_box = q[4:]
    v_box = v[4:]
    x_box = jnp.hstack([q_box, v_box])

    # Get MPC reference
    if k % control_ratio == 0:
        g_N = manipulator_sim._gap_function(x_current[:manipulator_sim.n_q])
        if g_N[0] <= 0.0 and g_N[1] <= 0.0:
            X_box_bar, U_box_bar, ddqdt_box, cost = box_MPC.optimize_trajectory(x_0=x_box)
            uk_box = U_box_bar[:, 0]

        # Solve Low-Level Controller
            ddqdt_val, uk_val, lambda_val = casadi_controller.solve(
                u_box_ref_val=uk_box, 
                ddq_box_ref_val=ddqdt_box[:, 0],
                A_val=A_np,
                b_val=b_np
        )
    
    # Convert to JAX arrays
    uk = jnp.array(uk_val)
    ddqdt = jnp.array(ddqdt_val)
    _lambda = jnp.array(lambda_val)

    # Store Data
    U = U.at[:, k].set(uk)
    Lambdas = Lambdas.at[:, k].set(_lambda) # <-- Save lambda trajectory

    # Integrate dynamics
    # x_next = integrator(x_current, uk)
    x_next = manipulator_sim.f_fcn(x_current, uk)
    x_current = x_next
    X = X.at[:, k+1].set(x_current)

print("Simulation complete.")
# --- Plotting Section ---
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# 1. Plot Positions (q)
# q is the first 6 elements of state X
for i in range(6):
    axs[0].plot(tspan_sim, X[i, :], label=f'$q_{{{i}}}$')
axs[0].set_ylabel('Positions $q$ [m]')
axs[0].set_title('State Trajectories')
axs[0].legend(loc='upper right', ncol=2)
axs[0].grid(True, alpha=0.5)

# 2. Plot Controls (u)
# U has length N_sim, so we use tspan_sim[:-1]
for i in range(4):
    axs[1].plot(tspan_sim[:-1], U[i, :], label=f'$u_{{{i}}}$')
axs[1].set_ylabel('Controls $u$ [N]')
axs[1].legend(loc='upper right', ncol=2)
axs[1].grid(True, alpha=0.5)

# 3. Plot Constraint Forces (lambda)
# Lambdas has length N_sim
for i in range(4):
    axs[2].plot(tspan_sim[:-1], Lambdas[i, :], label=f'$\lambda_{{{i}}}$')
axs[2].set_ylabel('Constraint Forces $\lambda$ [N]')
axs[2].set_xlabel('Time [s]')
axs[2].legend(loc='upper right', ncol=2)
axs[2].grid(True, alpha=0.5)

plt.tight_layout()
plt.show()

# --- Animation ---
anim = AnimationPointMassBox(manipulator, X, tspan_sim, dt)
anim.animate(fullscreen=True, save_video=False, filename='box_transport.mp4')