import time
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import casadi as ca
import scipy.linalg as la

# --- Custom Imports ---
from class_files.systems.dual_arm_manipulator_sys import MyDualArmManipulator
from class_files.box_3DoF_MPC import SurfaceBoxMPC
from casadi_low_level_control_dual_arm import CasadiLowLevelControllerDualArm

# ==========================================
# 1. Constants & Weights
# ==========================================
# Time
dt = 0.001
dt_control = 0.01 * 1  # Control at 100Hz
control_ratio = int(dt_control / dt)
T_horizon = dt
T_sim = 8.0

# --- NEW: Switching Times ---
T_switch_1 = 3.0
T_switch_2 = 5.0
# Dimensions (Must match system definition)
w_box = 0.4
h_box = 0.4

# MPC Target: Box lifted to y=0.8, upright (phi=0)
# State: [x, y, phi, vx, vy, vphi]

# MPC Weights
# Q_mpc = jnp.diag(jnp.array([100.0, 100.0, 400.0, 30.0, 30.0, 30.0]))           
# R_mpc = jnp.diag(jnp.array([1.0, 1.0, 1.0*1e0]))*1
Q_mpc = jnp.diag(jnp.array([100.0, 100.0, 100.0, 30.0, 30.0, 30.0]))           
R_mpc = jnp.diag(jnp.array([1.0, 1.0, 1.0*1e0]))*10
# Q_f_mpc = Q_mpc * 10.0
A = jnp.array([[0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 1],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0]])
B = jnp.array([[0, 0, 0],
               [0, 0, 0],
               [0, 0, 0],
               [1, 0, 0],
               [0, 1, 0],
               [0, 0, 1]])
A_d = jnp.eye(6) + A * dt + 0.5 * (A @ A) * dt**2 + (1/6) * (A @ A @ A) * dt**3 + (1/24) * (A @ A @ A @ A) * dt**4
B_d = B * dt + 0.5 * (A @ B) * dt**2 + (1/6) * (A @ A @ B) * dt**3 + (1/24) * (A @ A @ A @ B) * dt**4
# Discrete-time LQR solution for terminal weight
Q_f_mpc = la.solve_discrete_are(A_d, B_d, Q_mpc, R_mpc, None, None)
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
        mu=mu*1.0,
        m_EE=m_EE,
        theta_EE=theta_EE,
        m_box=m_box*1.0,
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

target_pose_L = jnp.array([ q_box[0] - 0.3, q_box[1] + 0.2, 0.0])  # x, y, phi


# target_pose_L = jnp.array([ q_box[0] - 0.6, q_box[1] + 0.4, 0.0])  # x, y, phi

target_pose_R = jnp.array([ q_box[0] + 0.3, q_box[1] - 0.1, 0.0])  # x, y, phi
q_0, conv = manipulator.inverse_kinematics_arms(target_pose_L, target_pose_R, q_0, max_iter=50, tol=1e-4)
if not conv:
    print("Warning: Initial IK did not converge.")

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
tau_max = np.array([50.0, 30.0, 10.0, 50.0, 30.0, 10.0])*1.0
# --- Casadi Low Level Controller ---
casadi_controller = CasadiLowLevelControllerDualArm(
    manipulator=manipulator,
    box_3DoF_MPC=box_MPC_controller,
    Q_box_acc=Q_box_acc,
    R_box_force=R_box_force,
    R_tau=R_tau,
    C=C_static,
    epsilon=epsilon,
    tau_max=tau_max,
    lambda_N_min = 1.0,
    w_smooth=500.0*0.0
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

    X_noisy = jnp.zeros_like(X) # For storing noisy measurements (optional)
    X_noisy = X_noisy.at[:, 0].set(x_current) # Initialize with true state (no noise at t=0)
    X_filtered = jnp.zeros_like(X) # For storing filtered state estimates (optional)
    X_filtered = X_filtered.at[:, 0].set(x_current) # Initialize with true state (no noise at t=0)

    # Loop Variables
    uk_box = jnp.array([0.0, 0.0, 0.0]) # Box Wrench Ref
    uk_val = np.zeros(6) # Joint Torques
    ddqdt_val = np.zeros(9)
    lambda_val = np.zeros(8)
    
    # Initialize dynamic target
    current_target = x_box_target_init
    key = random.PRNGKey(0)

    # Define Covariance Matrices (Variance of 0.01^2 = 0.0001)
    # cov_q = jnp.eye(manipulator.n_q) * ((3*jnp.pi/180)**2)
    # cov_v = jnp.eye(manipulator.n_v) * ((3*jnp.pi/180)**2)
    # --- Budget Sensor Noise Profile ---
    # Arms (High-res encoders but cheap processing)
    sigma_q_arm = jnp.deg2rad(0.1)   # 0.1 degree
    sigma_v_arm = 0.08              # Noticeable velocity noise

    # Box (Vision-based tracking at ~30-60Hz)
    sigma_q_box_xy = 0.015          # 1.5 cm jitter
    sigma_q_box_phi = jnp.deg2rad(3.0) # 3 degrees jitter (vision is noisy here)
    # sigma_q_box_xy = 0.003          # 1.5 cm jitter
    # sigma_q_box_phi = jnp.deg2rad(3.0) # 3 degrees jitter (vision is noisy here)

    # Box Velocity (Calculated via finite difference - VERY noisy)
    sigma_v_box_xy = 0.2            # 20 cm/s noise
    sigma_v_box_phi = 0.2           # High angular velocity noise

    # Constructing the matrices
    q_vars = jnp.array([sigma_q_arm]*6 + [sigma_q_box_xy, sigma_q_box_xy, sigma_q_box_phi])**2
    v_vars = jnp.array([sigma_v_arm]*6 + [sigma_v_box_xy, sigma_v_box_xy, sigma_v_box_phi])**2

    cov_q = jnp.diag(q_vars)
    cov_v = jnp.diag(v_vars)

    alpha_q_filter = 0.4/0.4  # Smoothing factor for low-pass filter (optional)
    alpha_v_filter = 0.4/0.4
    q_filtered = x_current[0:manipulator.n_q]
    v_filtered = x_current[manipulator.n_q:]
    print(f"Starting simulation... Initial Target Y={current_target[1]}")
    # --- Before the loop ---
    dt_box_measurement = 0.05  # Box measurements every 50ms (20Hz)
    dt_arm_measurement = 0.01  # Arm measurements every 10ms (100Hz)
    # measurement_ratio_arm = int(dt_arm_measurement / dt)  # Number of sim steps per arm measurement
    # measurement_ratio_box = int(dt_box_measurement / dt)  # Number of sim steps per box measurement
    measurement_ratio_box = int(50/50)  # 100Hz control / 20Hz vision
    measurement_ratio_arm = int(10/10)  # Arm measurements at every sim step (100Hz)

    # measurement_ratio_box = 1  # 100Hz control / 20Hz vision

    last_q_box_meas = x_current[6:9]
    last_v_box_meas = x_current[15:18]

    add_noise_to_box = False
    add_noise_to_arms = False

    target_2 = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    target_3 = jnp.array([-0.2, 0.3, 0.0, 0.0, 0.0, 0.0])

    for k in range(N_sim):
        key, subkey = random.split(key)
        q = x_current[0:9]
        v = x_current[9:18]
        
        if k % measurement_ratio_arm == 0:
            # 1. Arm Measurements (100Hz - Always new)
            
            if add_noise_to_arms:
                q_arm_noise = random.multivariate_normal(subkey, jnp.zeros(6), cov_q[:6, :6])
                v_arm_noise = random.multivariate_normal(subkey, jnp.zeros(6), cov_v[:6, :6])
            else:
                q_arm_noise = jnp.zeros(6)
                v_arm_noise = jnp.zeros(6)
            q_arm_meas = q[:6] + q_arm_noise
            v_arm_meas = v[:6] + v_arm_noise

        # 2. Box Measurements (20Hz - Conditional)
        if k % measurement_ratio_box == 0:
            # New vision frame arrives
            if add_noise_to_box:
                q_box_noise = random.multivariate_normal(subkey, jnp.zeros(3), cov_q[6:9, 6:9])
                v_box_noise = random.multivariate_normal(subkey, jnp.zeros(3), cov_v[6:9, 6:9])
            else:
                q_box_noise = jnp.zeros(3)
                v_box_noise = jnp.zeros(3)
            last_q_box_meas = q[6:9] + q_box_noise
            last_v_box_meas = v[6:9] + v_box_noise
            
            # UPDATE filter with real data
            q_filtered = q_filtered.at[6:9].set(alpha_q_filter * last_q_box_meas + (1 - alpha_q_filter) * q_filtered[6:9])
            v_filtered = v_filtered.at[6:9].set(alpha_v_filter * last_v_box_meas + (1 - alpha_v_filter) * v_filtered[6:9])
        else:
            # NO vision frame: PREDICT using MPC model
            x_box_prev = jnp.concatenate([q_filtered[6:9], v_filtered[6:9]])
            x_box_pred = A_d @ x_box_prev + B_d @ uk_box
            
            q_filtered = q_filtered.at[6:9].set(x_box_pred[:3])
            v_filtered = v_filtered.at[6:9].set(x_box_pred[3:])

        # 3. Always update Arm Filter
        q_filtered = q_filtered.at[:6].set(alpha_q_filter * q_arm_meas + (1 - alpha_q_filter) * q_filtered[:6])
        v_filtered = v_filtered.at[:6].set(alpha_v_filter * v_arm_meas + (1 - alpha_v_filter) * v_filtered[:6])

        # 4. Reconstruct q_measured and v_measured for X_noisy
        # This reflects the "stair-step" nature of the box sensor
        q_measured = jnp.concatenate([q_arm_meas, last_q_box_meas])
        v_measured = jnp.concatenate([v_arm_meas, last_v_box_meas])

        # 5. Save everything
        X_noisy = X_noisy.at[:, k].set(jnp.concatenate([q_measured, v_measured]))
        X_filtered = X_filtered.at[:, k].set(jnp.concatenate([q_filtered, v_filtered]))
        # x_box = jnp.hstack([q[6:], v[6:]])
        # x_box = jnp.hstack([q_measured[6:9], v_measured[6:9]])
        x_box = jnp.hstack([q_filtered[6:9], v_filtered[6:9]])
        # u_PD = manipulator._PD_controller(q, v)
        # u_PD_val = np.array(u_PD[0:6])
        # --- Dynamic Target Logic ---
        if tspan_sim[k] >= T_switch_1 and tspan_sim[k] < T_switch_2:
            # Change target to: x=0, y=0.8, phi=0 (Upright lift)
            current_target = target_2
            # Optional: Update system prop if needed elsewhere, though MPC uses the passed var
            manipulator.box_target_state = current_target 
        elif tspan_sim[k] >= T_switch_2:
            current_target = target_3
            # Optional: Update system prop if needed elsewhere, though MPC uses the passed var
            manipulator.box_target_state = current_target 
        else:
             current_target = x_box_target_init

        # Control Logic
        if k % control_ratio == 0:
            # u_PD = manipulator._PD_controller(q, v)
            # u_PD = manipulator._PD_controller(q_measured, v_measured)
            u_PD = manipulator._PD_controller(q_filtered, v_filtered)
            u_PD_val = np.array(u_PD[0:6])
            # 1. Check if Arms are close enough to apply force
            # g_N = manipulator._gap_function(x_current[:manipulator.n_q])
            g_N = manipulator_sim._gap_function(x_current[:manipulator.n_q])
            # g_N = manipulator_sim._gap_function(q_measured) # Use measured state for gap function   
            
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
                # M = manipulator._mass_matrix(q)
                # W = manipulator._contact_jacobian(q)[:, 0:8]
                # h = manipulator._generalized_forces(q, v, u0_dummy) # u=0 for h calculation
                # W_dot_T_v = manipulator._contact_jacobian_dot_transpose_dqdt(q, v)[0:8]

                M = manipulator._mass_matrix(q_filtered)
                W = manipulator._contact_jacobian(q_filtered)[:, 0:8]
                h = manipulator._generalized_forces(q_filtered, v_filtered, u0_dummy) # u=0 for h calculation
                W_dot_T_v = manipulator._contact_jacobian_dot_transpose_dqdt(q_filtered, v_filtered)[0:8]


                
                
                # Formulate Linear System for CasADi
                # [M  -S  -W ] [ddq]   [h]
                # [W'  0   0 ] [tau] = [W_dot_T_v]
                #              [lam]
                A_dyn = jnp.block([[M, -S, -W], [W.T, jnp.zeros((8, 14))]])
                b_dyn = jnp.hstack([h + u_PD, -W_dot_T_v])
                # b_dyn = jnp.hstack([h, -W_dot_T_v])


                A_np = np.array(A_dyn)
                b_np = np.array(b_dyn)

                # 4. Solve Low Level (Torque Allocation)
                # ddqdt_val, uk_val, lambda_val = casadi_controller.solve(
                #     u_box_ref_val=uk_box, 
                #     ddq_box_ref_val=ddqdt_box[:, 0],
                #     A_val=A_np, 
                #     b_val=b_np,
                #     v=np.array(v),
                #     u_prev_val=uk_val,
                #     u_PD_val=u_PD_val
                # )

                ddqdt_val, uk_val, lambda_val = casadi_controller.solve(
                    u_box_ref_val=uk_box, 
                    ddq_box_ref_val=ddqdt_box[:, 0],
                    A_val=A_np, 
                    b_val=b_np,
                    v=np.array(v_filtered),
                    u_prev_val=uk_val,
                    u_PD_val=u_PD_val
                )

                # print the torque
                # print(f"Step {k}: Low Level Joint Torques: {uk_val + u_PD[0:6]}")
                # box_wrench = W[6:, :] @ lambda_val
                # Debug prints (optional)
                # d_wrench = box_wrench - uk_box
                # print(f"Step {k}: Achieved Box Wrench: {box_wrench}")
            #     uk = jnp.array(uk_val)
            # else:
            #     uk = jnp.array(u_PD_val)

        # Store History
        uk = jnp.array(uk_val) + u_PD_val[0:6] # Add PD component
        # saturate torques (safety)
        uk = jnp.clip(uk, -tau_max, tau_max)

        # smooth the control input
        # alpha = 0.1
        alpha = 1.0

        # if k % control_ratio == 0 and k > 0:
        uk = alpha * uk + (1 - alpha) * U[:, k-1]


        _lambda = jnp.array(lambda_val)
        
        U = U.at[:, k].set(uk)
        Lambdas = Lambdas.at[:, k].set(_lambda)

        # Integrate Dynamics
        x_next = manipulator_sim.f_fcn(x_current, uk)
        x_current = x_next
        X = X.at[:, k+1].set(x_current)

    print("Simulation complete.")
    targets = [x_box_target_init, target_2, target_3]
    switching_times = [T_switch_1, T_switch_2]
    return tspan_sim, X, U, Lambdas, x_box_target_init, X_noisy, X_filtered, targets, switching_times

# ==========================================
# 4. Plotting & Animation
# ==========================================
if __name__ == "__main__":
    tspan, X, U, Lambdas, initial_target, X_noisy, X_filtered, targets, switching_times = run_simulation()
    initial_target, target_2, target_3 = targets
    T_switch_1, T_switch_2 = switching_times
    # --- Construct Reference Trajectories for Plotting ---
    ref_x_traj = np.zeros_like(tspan)
    ref_y_traj = np.zeros_like(tspan)
    ref_phi_traj = np.zeros_like(tspan)

    # 1. Initial Target (t < T_switch_1)
    mask1 = tspan < T_switch_1
    ref_x_traj[mask1] = initial_target[0]
    ref_y_traj[mask1] = initial_target[1]
    ref_phi_traj[mask1] = initial_target[2]

    # 2. Second Target (T_switch_1 <= t < T_switch_2)
    mask2 = (tspan >= T_switch_1) & (tspan < T_switch_2)
    ref_x_traj[mask2] = target_2[0]
    ref_y_traj[mask2] = target_2[1]
    ref_phi_traj[mask2] = target_2[2]

    # 3. Third Target (t >= T_switch_2)
    mask3 = tspan >= T_switch_2
    ref_x_traj[mask3] = target_3[0]
    ref_y_traj[mask3] = target_3[1]
    ref_phi_traj[mask3] = target_3[2]

    # --- Plotting ---
    fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    # 1. Box State vs Reference
    axs[0].plot(tspan, X[6, :], color='tab:blue', label='Box X')
    axs[0].plot(tspan, ref_x_traj, color='tab:blue', linestyle='--', alpha=0.7, label='Ref X')
    
    axs[0].plot(tspan, X[7, :], color='tab:orange', linewidth=2, label='Box Y')
    axs[0].plot(tspan, ref_y_traj, color='tab:orange', linestyle='--', alpha=0.7, label='Ref Y')
    
    axs[0].plot(tspan, X[8, :], color='tab:green', label='Box Phi')
    axs[0].plot(tspan, ref_phi_traj, color='tab:green', linestyle='--', alpha=0.7, label='Ref Phi')

    # Mark the switching times using variables
    axs[0].axvline(x=T_switch_1, color='k', linestyle=':', alpha=0.3)
    axs[0].axvline(x=T_switch_2, color='k', linestyle=':', alpha=0.3)
    
    axs[0].set_ylabel('Box State')
    axs[0].set_title('Box Trajectory Tracking')
    axs[0].legend(loc='upper left', ncol=3)
    axs[0].grid(True)

    # 2. Joint Torques
    labels_u = ['tau_1', 'tau_2', 'tau_3', 'tau_4', 'tau_5', 'tau_6']
    colors = plt.cm.tab10(np.linspace(0, 1, 6))
    for i in range(6):
        axs[1].plot(tspan[:-1], U[i, :], label=labels_u[i], color=colors[i])
    axs[1].set_ylabel('Torque [Nm]')
    axs[1].set_title('Joint Torques')
    axs[1].legend(ncol=6, fontsize='small', loc='lower right')
    axs[1].grid(True)

    # 3. Contact Forces
    force_labels = ['Up1', 'Low1', 'Up2', 'Low2']
    colors_f = plt.cm.Set1(np.linspace(0, 1, 4))
    
    for i in range(4):
        idx_n = i*2 + 1
        idx_t = i*2    
        axs[2].plot(tspan[:-1], Lambdas[idx_n, :], color=colors_f[i], label=f'{force_labels[i]}_N')
        axs[2].plot(tspan[:-1], Lambdas[idx_t, :], color=colors_f[i], linestyle='--', alpha=0.4)

    axs[2].set_ylabel('Contact Forces [N]')
    axs[2].set_xlabel('Time [s]')
    axs[2].set_title('Contact Forces (Solid: Normal, Dashed: Tangent)')
    axs[2].legend(ncol=4, fontsize='small')
    axs[2].grid(True)
    
    plt.tight_layout()
    # plt.show()

    # create a plot that compares the real trajectory vs the noisy measurements for all states
    # compare the filtered state as well to show the effect of the low-pass filter
    fig2, axs2 = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    state_labels = ['X Position', 'Y Position', 'Phi (Orientation)']
    for i in range(3):
        axs2[i].plot(tspan, X[6+i, :], label='True State', color='tab:blue')
        axs2[i].plot(tspan, X_noisy[6+i, :], label='Noisy Measurement', color='tab:orange', alpha=0.7)
        axs2[i].plot(tspan, X_filtered[6+i, :], label='Filtered Estimate', color='tab:green', alpha=0.7)
        axs2[i].set_ylabel(state_labels[i])
        axs2[i].legend()
        axs2[i].grid(True)
    axs2[2].set_xlabel('Time [s]')
    plt.tight_layout()

    # For the box velocities as well
    fig2b, axs2b = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    velocity_labels = ['X Velocity', 'Y Velocity', 'Angular Velocity']
    for i in range(3):
        axs2b[i].plot(tspan, X[9+i, :], label='True Velocity', color='tab:green')
        axs2b[i].plot(tspan, X_noisy[9+i, :], label='Noisy Measurement', color='tab:red', alpha=0.7)
        axs2b[i].plot(tspan, X_filtered[9+i, :], label='Filtered Estimate', color='tab:purple', alpha=0.7)
        axs2b[i].set_ylabel(velocity_labels[i])
        axs2b[i].legend()
        axs2b[i].grid(True)
    axs2b[2].set_xlabel('Time [s]')
    plt.tight_layout()


    # create a plot that compares the real trajectory vs the noisy measurements for the first 6 states (joint angles) and the next 6 states (joint velocities)
    fig3, axs3 = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
    for i in range(6):
        axs3[0].plot(tspan, X[i, :], label=f'Joint {i+1} Angle', color=plt.cm.tab10(i))
        axs3[0].plot(tspan, X_noisy[i, :], label=f'Joint {i+1} Angle Noisy', color=plt.cm.tab10(i), alpha=0.7)
        # filtered state for joint angles
        axs3[0].plot(tspan, X_filtered[i, :], label=f'Joint {i+1} Angle Filtered', color=plt.cm.tab10(i), linestyle='--', alpha=0.7)
        axs3[1].plot(tspan, X[manipulator.n_q + i, :], label=f'Joint {i+1} Velocity', color=plt.cm.tab10(i))
        axs3[1].plot(tspan, X_noisy[manipulator.n_q + i, :], label=f'Joint {i+1} Velocity Noisy', color=plt.cm.tab10(i), alpha=0.7)
        # filtered state for joint velocities
        axs3[1].plot(tspan, X_filtered[manipulator.n_q + i, :], label=f'Joint {i+1} Velocity Filtered', color=plt.cm.tab10(i), linestyle='--', alpha=0.7)
        axs3[0].grid(True)
        axs3[1].grid(True)
    axs3[0].set_ylabel('Joint Angles [rad]')
    axs3[1].set_ylabel('Joint Velocities [rad/s]')
    axs3[1].set_xlabel('Time [s]')
    axs3[0].legend(loc='upper right', fontsize='small')
    axs3[1].legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.show()

    # --- Animation ---
    try:
        from class_files.animations.animation_dual_arm_manipulator import AnimationDualArmBox
        print("\nPreparing Animation...")
        anim = AnimationDualArmBox(manipulator_sim, X, tspan, dt)
        anim.animate(save_video=False, filename='dual_arm_mpc_box_with_noise.mp4')
    except ImportError:
        print("Animation class not found.")