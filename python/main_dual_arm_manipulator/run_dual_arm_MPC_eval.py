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

from pathlib import Path
# ==========================================
# 1. Constants & Weights
# ==========================================
# Time
dt = 0.001
dt_control = 0.01 * 1  # Control at 100Hz
control_ratio = int(dt_control / dt)
T_horizon = dt
T_sim = 6.0 # Increased to allow for convergence-based switching

# --- NEW: Switching Targets and Thresholds ---
target_2 = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
target_3 = jnp.array([-0.2, 0.3, 0.0, 0.0, 0.0, 0.0])
convergence_threshold = 0.05 # 6D norm threshold (pos + vel)
# convergence_threshold = 0.01 # 6D norm threshold (pos + vel)

# Dimensions (Must match system definition)
w_box = 0.4
h_box = 0.4

# MPC Weights
Q_mpc = jnp.diag(jnp.array([100.0, 100.0, 100.0, 30.0, 30.0, 30.0]))           
# R_mpc = jnp.diag(jnp.array([1.0, 1.0, 1.0*1e0]))*1e-5
# R_mpc = jnp.diag(jnp.array([1.0, 1.0, 1.0*1e0]))*1e-3
R_mpc = jnp.diag(jnp.array([1.0, 1.0, 1.0*1e0]))*1e-2



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

Q_f_mpc = la.solve_discrete_are(A_d, B_d, Q_mpc, R_mpc, None, None)

# Low Level Controller Weights
Q_box_acc = jnp.diag(jnp.array([10.0, 100.0, 10.0]))
R_box_force = jnp.diag(jnp.array([0.1, 0.1, 0.1]))*1        
R_tau = jnp.diag(jnp.array([1.0, 1.0,
                            1.0, 1.0, 
                            1.0, 1.0])) * 0
epsilon = 1e-4

mu = jnp.array([1.0, 1.0, 1.0, 1.0, 0.2, 0.2])*1.0 
# x_box_target_init = jnp.array([0.15, 0.6, 5.0 * jnp.pi/180, 0.0, 0.0, 0.0])
x_box_target_init = jnp.array([0.15, 0.0, 5.0 * jnp.pi/180, 0.0, 0.0, 0.0])


R = jnp.diag(1e-4 * jnp.ones(6)) 
Q_box = jnp.diag(jnp.array([10.0, 10.0, 1.0, 1.0, 1.0, 1.0]))
Q_f = Q_box * 10.0
m_EE, theta_EE, m_box, theta_box = 0.5, 0.05, 1.0, 0.1

manipulator = MyDualArmManipulator(
        dt=dt, box_target_state=x_box_target_init, R=R, Q_box=Q_box, RN_list=[100.0]*6,
        Q_f=Q_f, RN_f_list=[1000.0]*6, integrator="moreau", w_box=w_box, h_box=h_box,
        mu=mu, m_EE=m_EE, theta_EE=theta_EE, m_box=m_box, theta_box=theta_box,
        x_base_L = -0.8, y_base_L = 0.7, x_base_R = 0.8, y_base_R = 0.7
    )

manipulator_sim = MyDualArmManipulator(
        dt=dt, box_target_state=x_box_target_init, R=R, Q_box=Q_box, RN_list=[100.0]*6,
        Q_f=Q_f, RN_f_list=[1000.0]*6, integrator="moreau", w_box=w_box, h_box=h_box,
        mu=mu, m_EE=m_EE, theta_EE=theta_EE, m_box=m_box*1.1, theta_box=theta_box,
        x_base_L = -0.8, y_base_L = 0.7, x_base_R = 0.8, y_base_R = 0.7
    )

# --- Initial State ---
q_L = jnp.array([-120*jnp.pi/180, 90*jnp.pi/180, 0*jnp.pi/180]) 
q_R = jnp.array([( 70 - 180 )*jnp.pi/180, -140*jnp.pi/180, 180*jnp.pi/180]) 
q_box = jnp.array([0.0, manipulator.h_box/2, 0.0]) 
q_0 = jnp.concatenate([q_L, q_R, q_box])
v_0 = jnp.zeros(9)

target_pose_L = jnp.array([ q_box[0] - 0.3, q_box[1] + 0.1, 0.0])
target_pose_R = jnp.array([ q_box[0] + 0.3, q_box[1] + 0.1, 0.0])
q_0, conv = manipulator.inverse_kinematics_arms(target_pose_L, target_pose_R, q_0, max_iter=50, tol=1e-4)
x_0 = jnp.concatenate([q_0, v_0])

box_MPC_controller = SurfaceBoxMPC(
    surface_box_sys=manipulator, T_horizon=T_horizon, Q=Q_mpc, R=R_mpc, Q_f=Q_f_mpc, ctrl_dt=dt_control
)

S = jnp.block([[jnp.eye(6)], [jnp.zeros((3, 6))]])
C_static = jnp.hstack([jnp.zeros((3, 6)), jnp.eye(3, 3)]) 
tau_max = np.array([50.0, 30.0, 10.0, 50.0, 30.0, 10.0])

w_smooth = 0.0
casadi_controller = CasadiLowLevelControllerDualArm(
    manipulator=manipulator, box_3DoF_MPC=box_MPC_controller, Q_box_acc=Q_box_acc,
    R_box_force=R_box_force, R_tau=R_tau, C=C_static, epsilon=epsilon,
    tau_max=tau_max, lambda_N_min = 1.0, w_smooth=w_smooth
)

# ==========================================
# 3. Simulation Execution
# ==========================================
def run_simulation():
    tspan_sim = jnp.arange(0, T_sim + box_MPC_controller.dt, box_MPC_controller.dt)
    N_sim = len(tspan_sim) - 1

    X = jnp.zeros((manipulator.n_x, N_sim + 1))
    U = jnp.zeros((manipulator.n_u, N_sim))
    Lambdas = jnp.zeros((8, N_sim))
    X_ref_history = jnp.zeros((6, N_sim + 1)) # To store the dynamic reference
    
    x_current = x_0
    X = X.at[:, 0].set(x_current)

    X_noisy = jnp.zeros_like(X); X_noisy = X_noisy.at[:, 0].set(x_current)
    X_filtered = jnp.zeros_like(X); X_filtered = X_filtered.at[:, 0].set(x_current)

    uk_box = jnp.array([0.0, 0.0, 0.0]) 
    uk_val = np.zeros(6); ddqdt_val = np.zeros(9); lambda_val = np.zeros(8)
    
    # --- Switching Logic Setup ---
    target_list = [x_box_target_init, target_2, target_3]
    current_target_idx = 0
    actual_switching_times = []
    task_finished_time = None

    key = random.PRNGKey(0)
    sigma_q_arm, sigma_v_arm = jnp.deg2rad(0.1), 0.08
    sigma_q_box_xy, sigma_q_box_phi = 0.015, jnp.deg2rad(3.0)
    sigma_v_box_xy, sigma_v_box_phi = 0.2, 0.2

    q_vars = jnp.array([sigma_q_arm]*6 + [sigma_q_box_xy, sigma_q_box_xy, sigma_q_box_phi])**2
    v_vars = jnp.array([sigma_v_arm]*6 + [sigma_v_box_xy, sigma_v_box_xy, sigma_v_box_phi])**2
    cov_q, cov_v = jnp.diag(q_vars), jnp.diag(v_vars)

    # alpha_q_filter, alpha_v_filter = 1.0*0.6, 1.0*0.6
    alpha_q_filter, alpha_v_filter = 1.0, 1.0

    q_filtered, v_filtered = x_current[0:9], x_current[9:18]
    
    measurement_ratio_box, measurement_ratio_arm = 1, 1
    last_q_box_meas, last_v_box_meas = x_current[6:9], x_current[15:18]
    add_noise_to_box, add_noise_to_arms = False, False

    for k in range(N_sim):
        key, subkey = random.split(key)
        q, v = x_current[0:9], x_current[9:18]
        
        # Original Measurement & Filtering Logic
        if k % measurement_ratio_arm == 0:
            q_arm_noise = random.multivariate_normal(subkey, jnp.zeros(6), cov_q[:6, :6]) if add_noise_to_arms else jnp.zeros(6)
            v_arm_noise = random.multivariate_normal(subkey, jnp.zeros(6), cov_v[:6, :6]) if add_noise_to_arms else jnp.zeros(6)
            q_arm_meas, v_arm_meas = q[:6] + q_arm_noise, v[:6] + v_arm_noise

        if k % measurement_ratio_box == 0:
            q_box_noise = random.multivariate_normal(subkey, jnp.zeros(3), cov_q[6:9, 6:9]) if add_noise_to_box else jnp.zeros(3)
            v_box_noise = random.multivariate_normal(subkey, jnp.zeros(3), cov_v[6:9, 6:9]) if add_noise_to_box else jnp.zeros(3)
            last_q_box_meas, last_v_box_meas = q[6:9] + q_box_noise, v[6:9] + v_box_noise
            q_filtered = q_filtered.at[6:9].set(alpha_q_filter * last_q_box_meas + (1 - alpha_q_filter) * q_filtered[6:9])
            v_filtered = v_filtered.at[6:9].set(alpha_v_filter * last_v_box_meas + (1 - alpha_v_filter) * v_filtered[6:9])
        else:
            x_box_prev = jnp.concatenate([q_filtered[6:9], v_filtered[6:9]])
            x_box_pred = A_d @ x_box_prev + B_d @ uk_box
            q_filtered, v_filtered = q_filtered.at[6:9].set(x_box_pred[:3]), v_filtered.at[6:9].set(x_box_pred[3:])

        q_filtered = q_filtered.at[:6].set(alpha_q_filter * q_arm_meas + (1 - alpha_q_filter) * q_filtered[:6])
        v_filtered = v_filtered.at[:6].set(alpha_v_filter * v_arm_meas + (1 - alpha_v_filter) * v_filtered[:6])

        X_noisy = X_noisy.at[:, k].set(jnp.concatenate([q_arm_meas, last_q_box_meas, v_arm_meas, last_v_box_meas]))
        X_filtered = X_filtered.at[:, k].set(jnp.concatenate([q_filtered, v_filtered]))

        # --- MODIFIED: State-Dependent Reference Logic ---
        current_target = target_list[current_target_idx]
        
        # Use TRUE state (X) for 6D check: q_box is X[6:9], v_box is X[15:18]
        true_box_state_6d = jnp.concatenate([x_current[6:9], x_current[15:18]])
        error_norm = jnp.linalg.norm(true_box_state_6d - current_target)

        if error_norm < convergence_threshold:
            if current_target_idx < len(target_list) - 1:
                actual_switching_times.append(tspan_sim[k])
                current_target_idx += 1
                current_target = target_list[current_target_idx]
                print(f"Target {current_target_idx} reached at {tspan_sim[k]:.3f}s. Switching...")
            elif task_finished_time is None:
                task_finished_time = tspan_sim[k]
                print(f"TASK FINISHED at {task_finished_time:.3f}s.")

        X_ref_history = X_ref_history.at[:, k].set(current_target)
        manipulator.box_target_state = current_target 

        # --- Control Logic (Unchanged) ---
        if k % control_ratio == 0:
            u_PD = manipulator._PD_controller(q_filtered, v_filtered)
            u_PD_val = np.array(u_PD[0:6])
            g_N = manipulator_sim._gap_function(x_current[:9])
            
            if (g_N[0] <= 0.0 or g_N[1] <= 0.0) and (g_N[2] <= 0.0 or g_N[3] <= 0.0):
                x_box_for_mpc = jnp.hstack([q_filtered[6:9], v_filtered[6:9]])
                _, U_box_bar, ddqdt_box, _ = box_MPC_controller.optimize_trajectory(
                    x_0=x_box_for_mpc, x_target_current=current_target
                )
                uk_box = U_box_bar[:, 0]
                
                M, W = manipulator._mass_matrix(q_filtered), manipulator._contact_jacobian(q_filtered)[:, 0:8]
                h = manipulator._generalized_forces(q_filtered, v_filtered, jnp.zeros(6))
                W_dot_T_v = manipulator._contact_jacobian_dot_transpose_dqdt(q_filtered, v_filtered)[0:8]

                A_dyn = jnp.block([[M, -S, -W], [W.T, jnp.zeros((8, 14))]])
                b_dyn = jnp.hstack([h + u_PD, -W_dot_T_v])

                ddqdt_val, uk_val, lambda_val = casadi_controller.solve(
                    uk_box, ddqdt_box[:, 0], np.array(A_dyn), np.array(b_dyn), 
                    np.array(v_filtered), uk_val, u_PD_val
                )

        uk = jnp.clip(jnp.array(uk_val) + u_PD_val[0:6], -tau_max, tau_max)
        # alpha_u = 0.1
        alpha_u = 1.0
        uk = alpha_u * uk + (1 - alpha_u) * U[:, k-1] if k > 0 else uk
        U, Lambdas = U.at[:, k].set(uk), Lambdas.at[:, k].set(jnp.array(lambda_val))
        x_current = manipulator_sim.f_fcn(x_current, uk)
        X = X.at[:, k+1].set(x_current)

    return tspan_sim, X, U, Lambdas, X_ref_history, X_noisy, X_filtered, actual_switching_times, task_finished_time

# ==========================================
# 4. Plotting
# ==========================================
if __name__ == "__main__":
    tspan, X, U, Lambdas, X_ref, X_noisy, X_filtered, switch_times, final_time = run_simulation()

    # save data to a file
    # create header and stack data together
    # header = "t, r_obox_x, r_obox_y, r_obox_phi"
    # define header for t, X, X_noisy, X_filtered, U, Lambdas
# 1. Define the Header (Matches your structure)
    header = "t, q1, q2, q3, q4, q5, q6, q_box_x, q_box_y, q_box_phi, v1, v2, v3, v4, v5, v6, v_box_x, v_box_y, v_box_phi"
    header += ", " + ", ".join([f"q{i}_noisy" for i in range(1, 7)]) + ", " + ", ".join([f"q_box_{axis}_noisy" for axis in ['x', 'y', 'phi']])
    header += ", " + ", ".join([f"v{i}_noisy" for i in range(1, 7)]) + ", " + ", ".join([f"v_box_{axis}_noisy" for axis in ['x', 'y', 'phi']])
    header += ", " + ", ".join([f"q{i}_filtered" for i in range(1, 7)]) + ", " + ", ".join([f"q_box_{axis}_filtered" for axis in ['x', 'y', 'phi']])
    header += ", " + ", ".join([f"v{i}_filtered" for i in range(1, 7)]) + ", " + ", ".join([f"v_box_{axis}_filtered" for axis in ['x', 'y', 'phi']])
    header += ", " + ", ".join([f"tau_{i}" for i in range(1, 7)])
    header += ", " + ", ".join([f"lambda_{i}" for i in range(1, 9)])
    # header += ", " + ", ".join([f"switch_time_{i}" for i in range(1, len(switch_times)+1)])
    # header += ", final_time"
    # add reference state to the header
    header += ", " + ", ".join([f"ref_{state}" for state in ['box_x', 'box_y', 'box_phi', 'box_vx', 'box_vy', 'box_vphi']])

    # 1. Align the dimensions
    # We use [:-1] to remove the very last state/time entry so that 
    # everything matches the length of the control inputs (U)
    t_plot = tspan[:-1].reshape(-1, 1)
    X_plot = X[:, :-1].T
    X_noisy_plot = X_noisy[:, :-1].T
    X_filtered_plot = X_filtered[:, :-1].T
    U_plot = U.T
    Lambdas_plot = Lambdas.T
    # reference history (also aligned)
    X_ref_plot = X_ref[:, :-1].T

    # downsample the data
    t_plot = t_plot[::10]
    X_plot = X_plot[::10, :]
    X_noisy_plot = X_noisy_plot[::10, :]
    X_filtered_plot = X_filtered_plot[::10, :]
    U_plot = U_plot[::10, :]
    Lambdas_plot = Lambdas_plot[::10, :]
    # switch_times_plot = switch_times_plot[:, ::10] if switch_times_plot.size > 0 else switch_times_plot
    # final_time_plot = final_time_plot[:, ::10] if final_time_plot.size > 0 else final_time_plot
    X_ref_plot = X_ref_plot[::10, :]

    # 2. Stack Simulation Data
    sim_data = np.hstack([
        t_plot, 
        X_plot, 
        X_noisy_plot, 
        X_filtered_plot, 
        U_plot, 
        Lambdas_plot,
        # switch_times_plot, # do something about the switching times and final time to fit into the structure
        # final_time_plot,
        X_ref_plot
    ])

    # 3. Handle Event Data (Switching and Final Times)
    num_rows = sim_data.shape[0]
    # Ensure final_time is treated as a list element
    event_list = switch_times + [final_time]
    num_events = len(event_list)
    
    # Create a NaN matrix and place events in the first row
    event_padding = np.full((num_rows, num_events), np.nan)
    event_padding[0, :] = event_list

    # 4. Combine and Save
    full_data = np.hstack([sim_data, event_padding])
    
    path = Path(Path(__file__).parent, "dual_arm_nominal.csv")
    np.savetxt(
        path,
        full_data,
        delimiter=", ",
        header=header,
        comments="",
    )
    print(f"Successfully saved {full_data.shape[0]} rows to {path}")
    print(f"Data saved to {path}")
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    # Box Positions
    axs[0].plot(tspan, X[6, :], label='Box X', color='tab:blue')
    axs[0].plot(tspan, X_ref[0, :], '--', color='tab:blue', alpha=0.5)
    axs[0].plot(tspan, X[7, :], label='Box Y', color='tab:orange')
    axs[0].plot(tspan, X_ref[1, :], '--', color='tab:orange', alpha=0.5)
    for st in switch_times: axs[0].axvline(x=st, color='red', linestyle=':', alpha=0.6)
    if final_time: axs[0].axvline(x=final_time, color='green', linewidth=2, label='Task Finished')
    axs[0].legend(); axs[0].grid(True); axs[0].set_title("Original Script + State-Dependent Switching")

    # Box Velocities
    axs[1].plot(tspan, X[15, :], label='Vx', color='tab:green')
    axs[1].plot(tspan, X_ref[3, :], '--', color='tab:green', alpha=0.5)
    axs[1].plot(tspan, X[16, :], label='Vy', color='tab:red')
    axs[1].plot(tspan, X_ref[4, :], '--', color='tab:red', alpha=0.5)
    axs[1].legend(); axs[1].grid(True)

    # Torques
    for i in range(6): axs[2].plot(tspan[:-1], U[i, :], label=f'tau_{i+1}')
    axs[2].set_ylabel('Torque [Nm]'); axs[2].grid(True)

    plt.tight_layout()
    plt.show()
        # --- Animation ---
    try:
        from class_files.animations.animation_dual_arm_manipulator import AnimationDualArmBox
        print("\nPreparing Animation...")
        anim = AnimationDualArmBox(manipulator_sim, X, tspan, dt)
        anim.animate(save_video=False, filename='dual_arm_mpc_box_heavier_than_expected.mp4')
    except ImportError:
        print("Animation class not found.")