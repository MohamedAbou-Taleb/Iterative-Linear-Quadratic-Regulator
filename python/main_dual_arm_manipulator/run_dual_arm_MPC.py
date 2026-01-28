import time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import casadi as ca

# --- Custom Imports ---
from class_files.systems.dual_arm_manipulator_sys import MyDualArmManipulator
from class_files.box_3DoF_MPC import SurfaceBoxMPC
from casadi_low_level_control_dual_arm import CasadiLowLevelControllerDualArm

# ==========================================
# 1. Constants & Weights
# ==========================================
# Time
dt = 0.001
dt_control = 0.01
control_ratio = int(dt_control / dt)
T_horizon = 1.0
T_sim = 4.0

# Dimensions (Must match system definition)
w_box = 0.4
h_box = 0.4

# MPC Target: Box lifted to y=0.8, upright (phi=0)
# State: [x, y, phi, vx, vy, vphi]

# MPC Weights
Q_mpc = jnp.diag(jnp.array([100.0, 100.0, 400.0, 30.0, 30.0, 30.0]))           
R_mpc = jnp.diag(jnp.array([1.0, 1.0, 1.0*1e0]))*1e-1
Q_f_mpc = Q_mpc * 10.0

# Low Level Controller Weights
Q_box_acc = jnp.diag(jnp.array([10.0, 100.0, 10.0]))
R_box_force = jnp.diag(jnp.array([0.1, 0.1, 0.1]))*1        
R_tau = jnp.diag(jnp.array([1.0, 1.0,
                            1.0, 1.0, 
                            1.0, 1.0])) * 0
epsilon = 1e-4

# --- Manipulator Physics Weights ---
# R: Regularization for system simulation (not used by controller)
R_sys = jnp.diag(1e-4 * jnp.ones(6))
Q_box_sys = jnp.diag(jnp.array([10.0, 10.0, 1.0, 1.0, 1.0, 1.0]))
RN_list = [100.0] * 6
RN_f_list = [1000.0] * 6
# mu = jnp.array([0.6] * 6) # Moderate friction
mu = jnp.array([1.0, 1.0, 1.0, 1.0, 0.2, 0.2])*1.0 # High friction for dual arm
# Initial Target (Placeholder)
x_box_target_init = jnp.array([0.15, 0.6, 5.0 * jnp.pi/180, 0.0, 0.0, 0.0])

# Weights
R = jnp.diag(1e-4 * jnp.ones(6)) 
Q_box = jnp.diag(jnp.array([10.0, 10.0, 1.0, 1.0, 1.0, 1.0]))
Q_f = Q_box * 10.0
m_EE = 0.5
theta_EE = 0.05
m_box = 1.0
theta_box = 0.1

# --- Instantiate System ---
# Base positions adjusted to ensure workspace reachability
manipulator = MyDualArmManipulator(
        dt=dt,
        box_target_state=x_box_target_init,
        R=R,
        Q_box=Q_box,
        RN_list=RN_list,
        Q_f=Q_f,
        RN_f_list=RN_f_list,
        integrator="moreau",
        w_box=w_box*1.0,
        h_box=h_box,
        mu=mu,
        m_EE=m_EE,
        theta_EE=theta_EE,
        m_box=m_box,
        theta_box=theta_box,
        # --- Specify Base Positions Here ---
        x_base_L = -0.8,  # Move Left Arm further left
        y_base_L = 0.7,   # Raise Left Arm base
        x_base_R = 0.8,   # Move Right Arm further right
        y_base_R = 0.7    # Raise Right Arm base
    )

manipulator_sim = MyDualArmManipulator(
        dt=dt,
        box_target_state=x_box_target_init,
        R=R,
        Q_box=Q_box,
        RN_list=RN_list,
        Q_f=Q_f,
        RN_f_list=RN_f_list,
        integrator="moreau",
        w_box=w_box,
        h_box=h_box,
        mu=mu,
        m_EE=m_EE,
        theta_EE=theta_EE,
        m_box=m_box,
        theta_box=theta_box,
        # --- Specify Base Positions Here ---
        x_base_L = -0.8,  # Move Left Arm further left
        y_base_L = 0.7,   # Raise Left Arm base
        x_base_R = 0.8,   # Move Right Arm further right
        y_base_R = 0.7    # Raise Right Arm base
    )

# --- 2. Initial State ---
# Left Arm: Shoulder=45deg, Elbow=-90deg
q_L = jnp.array([-120*jnp.pi/180, 90*jnp.pi/180, 0*jnp.pi/180]) 

# Right Arm: Shoulder=135deg, Elbow=90deg
q_R = jnp.array([( 70 - 180 )*jnp.pi/180, -140*jnp.pi/180, 180*jnp.pi/180]) 

# Box: Starts slightly in the air
q_box = jnp.array([0.0, 1*manipulator.h_box/2, 0.0]) 

q_0 = jnp.concatenate([q_L, q_R, q_box])
v_0 = jnp.zeros(9)
x_0 = jnp.concatenate([q_0, v_0])


# ==========================================
# 2. Controller Setup
# ==========================================

# --- Box MPC ---
box_MPC_controller = SurfaceBoxMPC(
    surface_box_sys=manipulator, # The MPC class is generic enough if it just looks at box dims
    T_horizon=T_horizon,
    Q=Q_mpc,
    R=R_mpc,
    Q_f=Q_f_mpc,
    ctrl_dt=dt_control
)

# --- Low Level Controller Helper Matrices ---
q0_dummy = jnp.zeros(manipulator.n_q)
v0_dummy = jnp.zeros(manipulator.n_v)
u0_dummy = jnp.zeros(manipulator.n_u)

# Selection Matrix S (9x6)
# Maps 6 joint torques to 9 generalized coordinates.
# Arms are actuated (Indices 0-5), Box is unactuated (Indices 6-8).
S = jnp.block([
    [jnp.eye(6)],
    [jnp.zeros((3, 6))]
])

# Contact Jacobian Slice
# We only care about the first 8 columns (4 contacts * 2 dirs) for manipulation
# These correspond to the Arm-Box contacts.
W_dummy = manipulator._contact_jacobian(q0_dummy)[:, 0:8]
h_dummy = manipulator._generalized_forces(q0_dummy, v0_dummy, u0_dummy)

M_dummy = manipulator._mass_matrix(q0_dummy)

# Static Matrices Structure for CasADi Init
A_static = jnp.block([[M_dummy, -S, -W_dummy], [W_dummy.T, jnp.zeros((8, 14))]])
b_static = jnp.hstack([h_dummy, jnp.zeros(8)])
C_static = jnp.hstack([jnp.zeros((3, 6)), jnp.eye(3, 3)]) # Extracts box acc

# Convert to numpy for Casadi
A_np = np.array(A_static)
b_np = np.array(b_static)

# --- Casadi Low Level Controller ---
casadi_controller = CasadiLowLevelControllerDualArm(
    manipulator=manipulator,
    box_3DoF_MPC=box_MPC_controller,
    Q_box_acc=Q_box_acc,
    R_box_force=R_box_force,
    R_tau=R_tau,
    C=C_static,
    epsilon=epsilon,
    tau_max=np.array([50.0, 30.0, 10.0, 50.0, 30.0, 10.0]),
    lambda_N_min = 0.0
    # tau_max=500.0 # Nm
)

# ==========================================
# 3. Simulation Execution
# ==========================================
def run_simulation():
    # Setup Storage
    tspan_sim = jnp.arange(0, T_sim + box_MPC_controller.dt, box_MPC_controller.dt)
    N_sim = len(tspan_sim) - 1

    X = jnp.zeros((manipulator.n_x, N_sim + 1))
    U = jnp.zeros((manipulator.n_u, N_sim))
    Lambdas = jnp.zeros((8, N_sim))
    
    x_current = x_0
    X = X.at[:, 0].set(x_current)

    # Loop Variables
    uk_box = jnp.array([0.0, 0.0, 0.0]) # Box Wrench Ref
    uk_val = np.zeros(6) # Joint Torques
    ddqdt_val = np.zeros(9)
    lambda_val = np.zeros(8)
    
    # Initialize dynamic target
    current_target = x_box_target_init
    
    print(f"Starting simulation... Initial Target Y={current_target[1]}")

    for k in range(N_sim):
        # Extract state components
        q = x_current[0:manipulator.n_q]
        v = x_current[manipulator.n_q:]
        x_box = jnp.hstack([q[6:], v[6:]])
        u_PD = manipulator._PD_controller(q, v)
        
        # --- Dynamic Target Logic ---
        if tspan_sim[k] >= 3.0 and tspan_sim[k] < 5.0:
            # Change target to: x=0, y=0.8, phi=0 (Upright lift)
            current_target = jnp.array([0.0, 1.4, 0.0, 0.0, 0.0, 0.0])
            # Optional: Update system prop if needed elsewhere, though MPC uses the passed var
            manipulator.box_target_state = current_target 
        elif tspan_sim[k] >= 5.0:
            current_target = jnp.array([-0.2, 0.3, 0.0, 0.0, 0.0, 0.0])
            # Optional: Update system prop if needed elsewhere, though MPC uses the passed var
            manipulator.box_target_state = current_target 
        else:
             current_target = x_box_target_init

        # Control Logic
        if k % control_ratio == 0:
            # 1. Check if Arms are close enough to apply force
            g_N = manipulator._gap_function(x_current[:manipulator.n_q])
            # g_N = manipulator_sim._gap_function(x_current[:manipulator.n_q])
            
            # Check if any of the EEs are in contact simultaneously on each side
            if (g_N[0] <= 0.0 or g_N[1] <= 0.0) and (g_N[2] <= 0.0 or g_N[3] <= 0.0):
                
                # 2. Solve High Level MPC (Box Trajectory)
                # --- CHANGE: Passed x_target_current to MPC ---
                _, U_box_bar, ddqdt_box, _ = box_MPC_controller.optimize_trajectory(
                    x_0=x_box, 
                    x_target_current=current_target
                )
                
                uk_box = U_box_bar[:, 0]
                # print(f"Step {k}: MPC Box Wrench Ref: {uk_box}")
                
                # 3. Update Dynamics Matrices at current state
                M = manipulator._mass_matrix(q)
                W = manipulator._contact_jacobian(q)[:, 0:8]
                h = manipulator._generalized_forces(q, v, u0_dummy) # u=0 for h calculation
                W_dot_T_v = manipulator._contact_jacobian_dot_transpose_dqdt(q, v)[0:8]
                
                # Formulate Linear System for CasADi
                # [M  -S  -W ] [ddq]   [h]
                # [W'  0   0 ] [tau] = [W_dot_T_v]
                #              [lam]
                A_dyn = jnp.block([[M, -S, -W], [W.T, jnp.zeros((8, 14))]])
                b_dyn = jnp.hstack([h + u_PD, -W_dot_T_v])

                A_np = np.array(A_dyn)
                b_np = np.array(b_dyn)

                # 4. Solve Low Level (Torque Allocation)
                ddqdt_val, uk_val, lambda_val = casadi_controller.solve(
                    u_box_ref_val=uk_box, 
                    ddq_box_ref_val=ddqdt_box[:, 0],
                    A_val=A_np, 
                    b_val=b_np,
                    v=np.array(v),
                    u_prev_val=uk_val
                )
                
                box_wrench = W[6:, :] @ lambda_val
                # Debug prints (optional)
                # d_wrench = box_wrench - uk_box
                # print(f"Step {k}: Achieved Box Wrench: {box_wrench}")

        # Store History
        uk = jnp.array(uk_val) + u_PD[0:6] # Add PD component
        _lambda = jnp.array(lambda_val)
        
        U = U.at[:, k].set(uk)
        Lambdas = Lambdas.at[:, k].set(_lambda)

        # Integrate Dynamics
        x_next = manipulator_sim.f_fcn(x_current, uk)
        x_current = x_next
        X = X.at[:, k+1].set(x_current)

    print("Simulation complete.")
    return tspan_sim, X, U, Lambdas, x_box_target_init

# ==========================================
# 4. Plotting & Animation
# ==========================================
if __name__ == "__main__":
    tspan, X, U, Lambdas, initial_target = run_simulation()

    # --- Plotting ---
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # 1. Box State
    axs[0].plot(tspan, X[6, :], label='Box X')
    axs[0].plot(tspan, X[7, :], label='Box Y', linewidth=2)
    axs[0].plot(tspan, X[8, :], label='Box Phi')
    
    # Plot Initial Target Lines
    axs[0].axhline(initial_target[0], color='b', linestyle='--', alpha=0.3, label='Init Target X')
    axs[0].axhline(initial_target[1], color='k', linestyle='--', alpha=0.3, label='Init Target Y')
    
    # Plot Final Target Lines (active after t=3.0)
    axs[0].axhline(0.0, color='b', linestyle='-', alpha=0.5, label='Final Target X')
    axs[0].axhline(0.8, color='k', linestyle='-', alpha=0.5, label='Final Target Y')
    
    axs[0].axvline(x=3.0, color='r', linestyle=':', label='Target Switch')
    
    axs[0].set_ylabel('Box Pos [m/rad]')
    axs[0].set_title('Box Trajectory')
    axs[0].legend()
    axs[0].grid(True)

    # 2. Joint Torques
    labels_u = ['L_S', 'L_E', 'L_W', 'R_S', 'R_E', 'R_W']
    for i in range(6):
        axs[1].plot(tspan[:-1], U[i, :], label=labels_u[i])
    axs[1].set_ylabel('Torque [Nm]')
    axs[1].set_title('Joint Torques')
    axs[1].legend(ncol=6, fontsize='small')
    axs[1].grid(True)

    # 3. Contact Forces
    force_labels = ['Up1', 'Low1', 'Up2', 'Low2']
    for i in range(4):
        idx = i*2 + 1
        axs[2].plot(tspan[:-1], Lambdas[idx, :], label=f'{force_labels[i]}_N')
    axs[2].set_ylabel('Normal Force [N]')
    axs[2].set_xlabel('Time [s]')
    axs[2].grid(True)
    
    # Plot tangent forces
    for i in range(4):
        idx = i*2
        axs[2].plot(tspan[:-1], Lambdas[idx, :], linestyle='--', alpha=0.5, label=f'{force_labels[i]}_T')
    axs[2].legend()
    
    plt.tight_layout()
    plt.show()

    # --- Animation ---
    try:
        from class_files.animations.animation_dual_arm_manipulator import AnimationDualArmBox
        print("\nPreparing Animation...")
        anim = AnimationDualArmBox(manipulator_sim, X, tspan, dt)
        anim.animate(save_video=False, filename='dual_arm_mpc.mp4')
    except ImportError:
        print("Animation class not found.")