import time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import casadi as ca

# --- Custom Imports ---
from class_files.systems.system_base import System
from class_files.systems.point_mass_box_manipulator_sys import MyPointMassBoxManipulator
from class_files.box_MPC import box_MPC
from class_files.animations.animation_point_mass_box import AnimationPointMassBox
from casadi_low_level import CasadiLowLevelController

# ==========================================
# 1. Constants & Weights
# ==========================================
# Time
dt = 0.001
dt_control = 0.01
control_ratio = int(dt_control / dt)
T_horizon = 1.0
T_sim = 7.0

# Dimensions & Mass
box_width = 0.5
box_height = 0.3
ball_radius = 0.05
m_box = 0.5
m_ball = 1.0
x_box_target = jnp.array([1.0, 5 * box_height / 2, 0.0, 0.0])

# Friction
reg_friction = jnp.array([1e-2, 1e-2, 1e-2])
mu = jnp.array([0.5, 0.5, 0.1])

# System Weights
R = jnp.diag(jnp.array([1.0, 1.0, 1.0, 1.0]))
Q_box = jnp.diag(jnp.array([1.0, 1.0, 1.0, 1.0]))
Q_f = jnp.diag(jnp.array([1.0, 1.0, 1.0, 1.0]))
RN1, RN2 = 1.0, 1.0
RN1_f, RN2_f = 1.0, 1.0

# MPC Weights
Q_mpc = jnp.diag(jnp.array([10.0, 100.0, 10.0, 100.0]))
R_mpc = jnp.diag(jnp.array([0.1, 0.1])) * 1e1
Q_f_mpc = jnp.diag(jnp.array([100.0, 100.0, 100.0, 100.0]))

# Low Level Controller Weights
Q_box_acc = jnp.diag(jnp.array([10.0, 100.0]))
R_box_force = jnp.diag(jnp.array([0.1, 0.1]))
R_tau = jnp.diag(jnp.array([1.0, 1.0, 1.0, 1.0])) * 1e-1
epsilon = 1e-3

# ==========================================
# 2. System Initialization
# ==========================================

# -- Manipulator for Model Definitions --
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

# -- Manipulator for Simulation (Time Stepping) --
manipulator_sim = MyPointMassBoxManipulator(
    dt=dt,
    box_target_state=x_box_target,
    R=R,
    Q_box=Q_box,
    RN1=RN1,
    RN2=RN2,
    Q_f=Q_f,
    RN1_f=RN1_f,
    RN2_f=RN2_f,
    integrator="contact_euler",
    box_height=box_height,
    box_width=box_width,
    ball_radius=ball_radius,
    m_box=m_box,
    m_ball=m_ball,
    g=9.81,
    mu=mu,
    reg_friction=reg_friction,
)

# ==========================================
# 3. Controller Setup
# ==========================================

# --- Box MPC ---
box_MPC_controller = box_MPC(
    point_mass_box_sys=manipulator,
    T_horizon=T_horizon,
    Q=Q_mpc,
    R=R_mpc,
    Q_f=Q_f_mpc,
    ctrl_dt=dt_control
)

# --- Low Level Controller Helper Matrices ---
# Pre-calculate matrices at q=0 for the Casadi controller
q0_dummy = jnp.zeros(manipulator.n_q)
v0_dummy = jnp.zeros(manipulator.n_v)
u0_dummy = jnp.zeros(manipulator.n_u)

S = jnp.eye(6, 4)
M = manipulator._mass_matrix(q0_dummy)
W = manipulator._contact_jacobian(q0_dummy)[:, 0:4]
h = manipulator._generalized_forces(q0_dummy, v0_dummy, u0_dummy)

A_static = jnp.block([[M, -S, -W], [W.T, jnp.zeros((4, 8))]])
b_static = jnp.hstack([h, jnp.zeros(4)])
C_static = jnp.hstack([jnp.zeros((2, 4)), jnp.eye(2, 2)])

# Convert to numpy for Casadi
A_np = np.array(A_static)
b_np = np.array(b_static)

# --- Casadi Low Level Controller ---
casadi_controller = CasadiLowLevelController(
    manipulator=manipulator,
    box_MPC=box_MPC_controller,
    Q_box_acc=Q_box_acc,
    R_box_force=R_box_force,
    R_tau=R_tau,
    C=C_static,
    epsilon=epsilon
)

# ==========================================
# 4. Simulation Execution
# ==========================================
def run_simulation():
    # Initial State
    q_0 = jnp.array([
        -2.0 * manipulator.ball_radius - manipulator.box_width / 2,
        manipulator.box_height / 2,
        manipulator.ball_radius * 3.0 + manipulator.box_width / 2,
        manipulator.box_height / 2,
        0.0,
        manipulator.box_height / 2
    ])
    v_0 = jnp.zeros(6)
    x_current = jnp.hstack([q_0, v_0])

    # Storage Setup
    tspan_sim = jnp.arange(0, T_sim + box_MPC_controller.dt, box_MPC_controller.dt)
    N_sim = len(tspan_sim) - 1

    X = jnp.zeros((manipulator.n_x, N_sim + 1))
    U = jnp.zeros((manipulator.n_u, N_sim))
    Lambdas = jnp.zeros((4, N_sim))
    
    X = X.at[:, 0].set(x_current)

    # Loop Variables
    uk_box = jnp.array([0.0, 0.0])
    uk_val = np.array([0.0, 0.0, 0.0, 0.0])
    ddqdt_val = np.zeros(6)
    lambda_val = np.zeros(4)

    print("Starting simulation loop...")

    for k in range(N_sim):
        # Extract state components
        q = x_current[0:6]
        v = x_current[6:12]
        x_box = jnp.hstack([q[4:], v[4:]])

        # Control Logic
        if k % control_ratio == 0:
            # Check gap function
            g_N = manipulator_sim._gap_function(x_current[:manipulator_sim.n_q])
            
            if g_N[0] <= 0.0 and g_N[1] <= 0.0:
                    # 1. Solve MPC
                _, U_box_bar, ddqdt_box, _ = box_MPC_controller.optimize_trajectory(x_0=x_box)
                uk_box = U_box_bar[:, 0]

                # 2. Solve Low-Level Controller
                ddqdt_val, uk_val, lambda_val = casadi_controller.solve(
                    u_box_ref_val=uk_box, 
                    ddq_box_ref_val=ddqdt_box[:, 0],
                    A_val=A_np, 
                    b_val=b_np
                )
            # else:
            #     uk_val = manipulator_sim._PD_controller(q, v)

        # Store History
        uk = jnp.array(uk_val)
        _lambda = jnp.array(lambda_val)
        
        U = U.at[:, k].set(uk)
        Lambdas = Lambdas.at[:, k].set(_lambda)

        # Integrate Dynamics
        x_next = manipulator_sim.f_fcn(x_current, uk)
        x_current = x_next
        X = X.at[:, k+1].set(x_current)

    print("Simulation complete.")
    return tspan_sim, X, U, Lambdas

# ==========================================
# 5. Plotting & Animation
# ==========================================
if __name__ == "__main__":
    tspan, X, U, Lambdas = run_simulation()

    # --- Plotting ---
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # 1. Position
    for i in range(6):
        axs[0].plot(tspan, X[i, :], label=f'$q_{{{i}}}$')
    axs[0].set_ylabel('Positions $q$ [m]')
    axs[0].set_title('State Trajectories')
    axs[0].legend(loc='upper right', ncol=2)
    axs[0].grid(True, alpha=0.5)

    # 2. Controls
    for i in range(4):
        axs[1].plot(tspan[:-1], U[i, :], label=f'$u_{{{i}}}$')
    axs[1].set_ylabel('Controls $u$ [N]')
    axs[1].legend(loc='upper right', ncol=2)
    axs[1].grid(True, alpha=0.5)

    # 3. Forces
    for i in range(4):
        axs[2].plot(tspan[:-1], Lambdas[i, :], label=f'$\lambda_{{{i}}}$')
    axs[2].set_ylabel('Constraint Forces $\lambda$ [N]')
    axs[2].set_xlabel('Time [s]')
    axs[2].legend(loc='upper right', ncol=2)
    axs[2].grid(True, alpha=0.5)

    plt.tight_layout()
    plt.show()

    # --- Animation ---
    anim = AnimationPointMassBox(manipulator, X, tspan, dt)
    anim.animate(fullscreen=True, save_video=False, filename='box_transport.mp4')