import time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import casadi as ca

# --- Custom Imports ---
from class_files.systems.system_base import System
from class_files.systems.surface_box_manipulator_sys import MySurfaceBoxManipulator
from class_files.box_3DoF_MPC import SurfaceBoxMPC
from class_files.animations.animation_surface_box import AnimationSurfaceBox
from casadi_low_level_control import CasadiLowLevelController

# ==========================================
# 1. Constants & Weights
# ==========================================
# Time
dt = 0.001
dt_control = 0.01
control_ratio = int(dt_control / dt)
T_horizon = 1.0
T_sim = 10.0

# Dimensions
w_box = 0.5
h_box = 0.5
w_EE = 0.1
h_EE = 0.3

# Target: Box lifted to y=1.0, upright (phi=0)
# State: [x, y, phi, vx, vy, vphi]
x_box_target = jnp.array([1.0, 0.6, 45*jnp.pi/180.0, 0.0, 0.0, 0.0])

# Weights
# u is size 6
R = jnp.diag(1e-2 * jnp.ones(6)) 

# Box tracking (x, y, phi, vx, vy, vphi)
Q_box = jnp.diag(jnp.array([10.0, 10.0, 1.0, 1.0, 1.0, 1.0]))
Q_f = Q_box * 10.0

# Gap weights (6 contacts)
RN_list = [10.0] * 6
RN_f_list = [100.0] * 6

# Friction (6 contacts) - high friction for grasp, low for ground slide?
# Order: [U1, L1, U2, L2, GL, GR]
# mu = jnp.array([4.0, 4.0, 4.0, 4.0, 1.0, 1.0])/5.0
mu = jnp.array([0.8, 0.8, 0.8, 0.8, 0.2, 0.2])

# m_box = 1.0
# m_EE = 1.0
m_box = 1.0
m_EE = 1.0
# --- Instantiate System ---
manipulator = MySurfaceBoxManipulator(
    dt=dt,
    box_target_state=x_box_target,
    R=R,
    Q_box=Q_box,
    RN_list=RN_list,
    Q_f=Q_f,
    RN_f_list=RN_f_list,
    integrator="contact_euler",
    w_box=w_box,
    h_box=h_box,
    w_EE=w_EE,
    h_EE=h_EE,
    m_box=m_box,
    m_EE=m_EE,
    mu=mu,
)

# --- Initial State ---
# EE1 (Left)
q_ee1 = jnp.array([-0.6, 0.25, 1*30*jnp.pi/180])
# EE2 (Right)
q_ee2 = jnp.array([0.9, 0.25, 1*30*jnp.pi/180])
# Box (Center, on ground)
# h_box/2 is the y-center when on ground
q_box = jnp.array([0.0, 1*h_box/2, 30*jnp.pi/180]) 

q_0 = jnp.concatenate([q_ee1, q_ee2, q_box])
v_0 = jnp.zeros(9)
x_0 = jnp.concatenate([q_0, v_0])
# Target: Box lifted to y=1.0, upright (phi=0)
# State: [x, y, phi, vx, vy, vphi]

# MPC Weights
Q_mpc = jnp.diag(jnp.array([10.0, 100.0, 40.0, 10.0, 10.0, 100.0]))
R_mpc = jnp.diag(jnp.array([1.0, 1.0, 1.0*1e1]))*1e1
Q_f_mpc = Q_mpc * 10.0

# Low Level Controller Weights
Q_box_acc = jnp.diag(jnp.array([10.0, 100.0, 10.0]))
R_box_force = jnp.diag(jnp.array([0.1, 0.1, 0.1]))*1        
R_tau = jnp.diag(jnp.array([1.0, 1.0,
                            1.0, 1.0, 
                            1.0, 1.0])) * 0
epsilon = 1e-4

# ==========================================
# 2. System Initialization
# ==========================================


# -- Manipulator for Simulation (Time Stepping) --
manipulator_sim = MySurfaceBoxManipulator(
    dt=dt,
    box_target_state=x_box_target,
    R=R,
    Q_box=Q_box,
    RN_list=RN_list,
    Q_f=Q_f,
    RN_f_list=RN_f_list,
    integrator="contact_euler",
    w_box=w_box,
    h_box=h_box,
    w_EE=w_EE,
    h_EE=h_EE,
    m_box=m_box,
    m_EE=m_EE,
    mu=mu,
)

# ==========================================
# 3. Controller Setup
# ==========================================

# --- Box MPC ---
box_MPC_controller = SurfaceBoxMPC(
    surface_box_sys=manipulator,
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

S = jnp.eye(9, 6)
M = manipulator._mass_matrix(q0_dummy)
W = manipulator._contact_jacobian(q0_dummy)[:, 0:8]
h = manipulator._generalized_forces(q0_dummy, v0_dummy, u0_dummy)

A_static = jnp.block([[M, -S, -W], [W.T, jnp.zeros((8, 14))]])
b_static = jnp.hstack([h, jnp.zeros(8)])
C_static = jnp.hstack([jnp.zeros((3, 6)), jnp.eye(3, 3)])

# Convert to numpy for Casadi
A_np = np.array(A_static)
b_np = np.array(b_static)

# --- Casadi Low Level Controller ---
casadi_controller = CasadiLowLevelController(
    manipulator=manipulator,
    box_3DoF_MPC=box_MPC_controller,
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
        -2.0 * manipulator.w_EE/2 - manipulator.w_box / 2,
        manipulator.h_box / 2,
        0.0 * jnp.pi/180,
        3.0 * manipulator.w_EE/2 + manipulator.w_box / 2,
        manipulator.h_box / 2,
        0.0 * jnp.pi/180,
        0.0,
        3*manipulator.h_box / 2,
        2.0 * jnp.pi/180
    ])
    v_0 = jnp.zeros(9)
    x_current = jnp.hstack([q_0, v_0])

    # Storage Setup
    tspan_sim = jnp.arange(0, T_sim + box_MPC_controller.dt, box_MPC_controller.dt)
    N_sim = len(tspan_sim) - 1

    X = jnp.zeros((manipulator.n_x, N_sim + 1))
    U = jnp.zeros((manipulator.n_u, N_sim))
    Lambdas = jnp.zeros((8, N_sim))
    
    X = X.at[:, 0].set(x_current)

    # Loop Variables
    uk_box = jnp.array([0.0, 0.0, 0.0])
    uk_val = np.array([0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0])
    ddqdt_val = np.zeros(9)
    lambda_val = np.zeros(8)

    print("Starting simulation loop...")

    for k in range(N_sim):
        # Extract state components
        q = x_current[0:manipulator.n_q]
        v = x_current[manipulator.n_q:]
        x_box = jnp.hstack([q[6:], v[6:]])

        # Control Logic
        if k % control_ratio == 0:
            # Check gap function
            g_N = manipulator_sim._gap_function(x_current[:manipulator_sim.n_q])
            
            # if g_N[0] <= 0.0 and g_N[1] <= 0.0 and g_N[2] <= 0.0 and g_N[3] <= 0.0:
            # check if any of the EEs are in contact simultaneously on each side, i.e., at least one contact per EE
            if (g_N[0] <= 0.0 or g_N[1] <= 0.0) and (g_N[2] <= 0.0 or g_N[3] <= 0.0):
            # if g_N[0] <= 0.0 and g_N[2] <= 0.0 :
                    # 1. Solve MPC
                _, U_box_bar, ddqdt_box, _ = box_MPC_controller.optimize_trajectory(x_0=x_box)
                uk_box = U_box_bar[:, 0]

                # 2. Solve Low-Level Controller
                M = manipulator._mass_matrix(q)
                W = manipulator._contact_jacobian(q)[:, 0:8]
                h = manipulator._generalized_forces(q, v, u0_dummy)
                W_dot_T_v = manipulator._contact_jacobian_dot_transpose_dqdt(q,v)[0:8]
                A_static = jnp.block([[M, -S, -W], [W.T, jnp.zeros((8, 14))]])
                b_static = jnp.hstack([h, W_dot_T_v])
                C_static = jnp.hstack([jnp.zeros((3, 6)), jnp.eye(3, 3)])

                # Convert to numpy for Casadi
                A_np = np.array(A_static)
                b_np = np.array(b_static)

                ddqdt_val, uk_val, lambda_val = casadi_controller.solve(
                    u_box_ref_val=uk_box, 
                    ddq_box_ref_val=ddqdt_box[:, 0],
                    A_val=A_np, 
                    b_val=b_np,
                    v=np.array(v)
                )
            #     uk_val += manipulator_sim._PD_controller(q, v)[0:6]
            # else:
            #     uk_val = manipulator_sim._PD_controller(q, v)[0:6]

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

    # 1. Position (9 DOF)
    # q0-q2: EE1, q3-q5: EE2, q6-q8: Box
    labels_q = ['EE1_x', 'EE1_y', 'EE1_phi', 
                'EE2_x', 'EE2_y', 'EE2_phi', 
                'Box_x', 'Box_y', 'Box_phi']
    
    for i in range(9):
        axs[0].plot(tspan, X[i, :], label=f'${labels_q[i]}$')
    axs[0].set_ylabel('Positions $q$ [m] / [rad]')
    axs[0].set_title('State Trajectories')
    # Use smaller font or fewer columns if legend is too crowded
    axs[0].legend(loc='upper right', ncol=3, fontsize='small')
    axs[0].grid(True, alpha=0.5)

    # 2. Controls (6 Inputs)
    # u0-u2: EE1, u3-u5: EE2
    labels_u = ['EE1_Fx', 'EE1_Fy', 'EE1_Tau', 
                'EE2_Fx', 'EE2_Fy', 'EE2_Tau']

    for i in range(6):
        axs[1].plot(tspan[:-1], U[i, :], label=f'${labels_u[i]}$')
    axs[1].set_ylabel('Controls $u$ [N] / [Nm]')
    axs[1].legend(loc='upper right', ncol=3, fontsize='small')
    axs[1].grid(True, alpha=0.5)

    # 3. Forces (12 Constraints)
    # 0,1: Upper1 | 2,3: Lower1 | 4,5: Upper2 | 6,7: Lower2 | 8,9: GndL | 10,11: GndR
    # Even indices = Tangential, Odd indices = Normal
    for i in range(12):
        # Optional: Plot only normal forces (odd indices) to reduce clutter
        # if i % 2 != 0: 
        axs[2].plot(tspan[:-1], Lambdas[i, :], label=f'$\lambda_{{{i}}}$')
    axs[2].set_ylabel('Constraint Forces $\lambda$ [N]')
    axs[2].set_xlabel('Time [s]')
    axs[2].legend(loc='upper right', ncol=4, fontsize='x-small')
    axs[2].grid(True, alpha=0.5)

    plt.tight_layout()
    plt.show()

    # --- Animation ---
    anim = AnimationSurfaceBox(manipulator, X, tspan, dt)
    anim.animate(fullscreen=True, save_video=True, filename='box_transport.mp4')