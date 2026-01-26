import sympy as sp
from sympy.printing.numpy import NumPyPrinter
import os

def generate_dual_arm_box_dynamics():
    print("Initializing Symbolic Derivation for Dual-Arm System...")

    # --- 1. Define Symbols ---
    # Generalized coordinates: 9 DoF
    # q0-q2: Left Arm (Shoulder, Elbow, Wrist) - Relative Angles
    # q3-q5: Right Arm (Shoulder, Elbow, Wrist) - Relative Angles
    # q6-q8: Box (x, y, alpha)
    q_list = list(sp.symbols('q0:9'))
    dq_list = list(sp.symbols('dq0:9'))
    tau_list = list(sp.symbols('tau0:6')) # 6 Actuators

    q = sp.Matrix(q_list)
    dq = sp.Matrix(dq_list)
    tau = sp.Matrix(tau_list)
    
    # System Parameters
    # Arm Geometry & Mass
    l1, l2, lc1, lc2 = sp.symbols('l1 l2 lc1 lc2')
    m1, m2, m_EE = sp.symbols('m1 m2 m_EE')
    theta1, theta2, theta_EE = sp.symbols('theta1 theta2 theta_EE')
    
    # Box Parameters
    w_box, h_box = sp.symbols('w_box h_box')
    m_box, theta_box = sp.symbols('m_box theta_box')
    
    # EE Geometry
    w_EE, h_EE = sp.symbols('w_EE h_EE')
    
    # Environment
    g = sp.symbols('g')
    
    # Base Positions
    x_base_L, y_base_L = sp.symbols('x_base_L y_base_L')
    x_base_R, y_base_R = sp.symbols('x_base_R y_base_R')

    # Unpack state for calculation
    # Left Arm
    q_L1, q_L2, q_L3 = q[0], q[1], q[2]
    # Right Arm
    q_R1, q_R2, q_R3 = q[3], q[4], q[5]
    # Box
    x_box, y_box, alpha_box = q[6], q[7], q[8]

    # --- 2. Kinematics (Forward Kinematics) ---
    def A_IB_fcn(angle):
        return sp.Matrix([
            [sp.cos(angle), -sp.sin(angle)],
            [sp.sin(angle),  sp.cos(angle)]
        ])

    # -- Left Arm (Relative -> Absolute) --
    alpha_L1 = q_L1
    alpha_L2 = q_L1 + q_L2
    alpha_L3 = q_L1 + q_L2 + q_L3
    
    pos_base_L = sp.Matrix([x_base_L, y_base_L])
    pos_cm_L1 = pos_base_L + A_IB_fcn(alpha_L1) * sp.Matrix([lc1, 0])
    pos_joint_L2 = pos_base_L + A_IB_fcn(alpha_L1) * sp.Matrix([l1, 0])
    pos_cm_L2 = pos_joint_L2 + A_IB_fcn(alpha_L2) * sp.Matrix([lc2, 0])
    pos_wrist_L = pos_joint_L2 + A_IB_fcn(alpha_L2) * sp.Matrix([l2, 0])
    
    # EE1 Pose (Tip)
    pos_EE1 = pos_wrist_L # Assuming wrist is EE center
    alpha_EE1 = alpha_L3

    # -- Right Arm (Relative -> Absolute) --
    alpha_R1 = q_R1
    alpha_R2 = q_R1 + q_R2
    alpha_R3 = q_R1 + q_R2 + q_R3

    pos_base_R = sp.Matrix([x_base_R, y_base_R])
    pos_cm_R1 = pos_base_R + A_IB_fcn(alpha_R1) * sp.Matrix([lc1, 0])
    pos_joint_R2 = pos_base_R + A_IB_fcn(alpha_R1) * sp.Matrix([l1, 0])
    pos_cm_R2 = pos_joint_R2 + A_IB_fcn(alpha_R2) * sp.Matrix([lc2, 0])
    pos_wrist_R = pos_joint_R2 + A_IB_fcn(alpha_R2) * sp.Matrix([l2, 0])

    # EE2 Pose (Tip)
    pos_EE2 = pos_wrist_R
    alpha_EE2 = alpha_R3

    # -- Box --
    pos_box = sp.Matrix([x_box, y_box])

    # --- 3. Dynamics (Lagrangian & Forces) ---
    print("Calculating Dynamics...")
    
    # Velocities (Linear and Angular)
    v_cm_L1 = pos_cm_L1.jacobian(q) * dq
    v_cm_L2 = pos_cm_L2.jacobian(q) * dq
    v_EE1   = pos_EE1.jacobian(q) * dq
    J_EE1 = pos_EE1.jacobian(q)
    
    v_cm_R1 = pos_cm_R1.jacobian(q) * dq
    v_cm_R2 = pos_cm_R2.jacobian(q) * dq
    v_EE2   = pos_EE2.jacobian(q) * dq
    J_EE2 = pos_EE2.jacobian(q)
    
    v_box   = pos_box.jacobian(q) * dq

    omega_L1 = sp.Matrix([alpha_L1]).jacobian(q) * dq
    omega_L2 = sp.Matrix([alpha_L2]).jacobian(q) * dq
    omega_EE1 = sp.Matrix([alpha_EE1]).jacobian(q) * dq

    omega_R1 = sp.Matrix([alpha_R1]).jacobian(q) * dq
    omega_R2 = sp.Matrix([alpha_R2]).jacobian(q) * dq
    omega_EE2 = sp.Matrix([alpha_EE2]).jacobian(q) * dq

    omega_box = sp.Matrix([alpha_box]).jacobian(q) * dq

    # Kinetic Energy (T)
    def get_KE(m, v, theta, w):
        return 0.5 * m * (v.T * v)[0] + 0.5 * theta * (w.T * w)[0]

    T_L = get_KE(m1, v_cm_L1, theta1, omega_L1) + \
          get_KE(m2, v_cm_L2, theta2, omega_L2) + \
          get_KE(m_EE, v_EE1, theta_EE, omega_EE1)

    T_R = get_KE(m1, v_cm_R1, theta1, omega_R1) + \
          get_KE(m2, v_cm_R2, theta2, omega_R2) + \
          get_KE(m_EE, v_EE2, theta_EE, omega_EE2)

    T_box = get_KE(m_box, v_box, theta_box, omega_box)
    T_total = T_L + T_R + T_box

    # Mass Matrix M
    print("  - Mass Matrix M...")
    grads = sp.Matrix([T_total]).jacobian(dq) 
    M = grads.jacobian(dq)

    # Potential Energy (V)
    print("  - Potential Energy V...")
    def get_PE(m, pos):
        return m * g * pos[1] 

    V_total = get_PE(m1, pos_cm_L1) + get_PE(m2, pos_cm_L2) + get_PE(m_EE, pos_EE1) + \
              get_PE(m1, pos_cm_R1) + get_PE(m2, pos_cm_R2) + get_PE(m_EE, pos_EE2) + \
              get_PE(m_box, pos_box)

    # Generalized Forces
    gen_force = -sp.Matrix([V_total]).jacobian(q).T

    print("  - Convective Terms...")
    def add_conv_term(F_accum, pos_sym, m_sym, v_sym):
        J = pos_sym.jacobian(q)
        a_conv = v_sym.jacobian(q) * dq 
        return F_accum - J.T * (m_sym * a_conv)

    gen_force = add_conv_term(gen_force, pos_cm_L1, m1, v_cm_L1)
    gen_force = add_conv_term(gen_force, pos_cm_L2, m2, v_cm_L2)
    gen_force = add_conv_term(gen_force, pos_EE1, m_EE, v_EE1)
    gen_force = add_conv_term(gen_force, pos_cm_R1, m1, v_cm_R1)
    gen_force = add_conv_term(gen_force, pos_cm_R2, m2, v_cm_R2)
    gen_force = add_conv_term(gen_force, pos_EE2, m_EE, v_EE2)
    gen_force = add_conv_term(gen_force, pos_box, m_box, v_box)

    # Actuation
    print("  - Actuation...")
    B = sp.zeros(9, 6)
    B[0:6, 0:6] = sp.eye(6)
    gen_force = gen_force + B * tau

    # --- 4. Contact Kinematics (W Matrix) ---
    print("Calculating Contact Constraints...")
    
    A_IB_EE1 = A_IB_fcn(alpha_EE1)
    A_IB_EE2 = A_IB_fcn(alpha_EE2)
    A_IB_box = A_IB_fcn(alpha_box)

    I_e_x_B_box = A_IB_box.col(0)
    I_e_y_B_box = A_IB_box.col(1)

    # Box Corners
    I_r_Obox_left  = pos_box - w_box/2 * I_e_x_B_box
    I_r_Obox_right = pos_box + w_box/2 * I_e_x_B_box
    I_r_Obox_bottom_left  = I_r_Obox_left  - h_box/2 * I_e_y_B_box
    I_r_Obox_bottom_right = I_r_Obox_right - h_box/2 * I_e_y_B_box

    # EE Contact Points (SP)
    I_r_SP_upper1 = A_IB_EE1 * sp.Matrix([w_EE/2, h_EE/2])
    I_r_SP_lower1 = A_IB_EE1 * sp.Matrix([w_EE/2, -h_EE/2])
    I_r_OP_upper1 = pos_EE1 + I_r_SP_upper1
    I_r_OP_lower1 = pos_EE1 + I_r_SP_lower1

    I_r_SP_upper2 = A_IB_EE2 * sp.Matrix([-w_EE/2, h_EE/2])
    I_r_SP_lower2 = A_IB_EE2 * sp.Matrix([-w_EE/2, -h_EE/2])
    I_r_OP_upper2 = pos_EE2 + I_r_SP_upper2
    I_r_OP_lower2 = pos_EE2 + I_r_SP_lower2

    # Relative Vectors (Gap calc)
    I_r_P_upper1_box_left = I_r_Obox_left - I_r_OP_upper1
    I_r_P_lower1_box_left = I_r_Obox_left - I_r_OP_lower1
    I_r_P_upper2_box_right = I_r_Obox_right - I_r_OP_upper2
    I_r_P_lower2_box_right = I_r_Obox_right - I_r_OP_lower2

    # Normals
    I_n_box_left = -I_e_x_B_box
    I_n_box_right = I_e_x_B_box
    I_n_ground = sp.Matrix([0, 1])

    # Gap Functions (g_N)
    g_N_upper1 = (I_n_box_left.T * (-I_r_P_upper1_box_left))[0]
    g_N_lower1 = (I_n_box_left.T * (-I_r_P_lower1_box_left))[0]
    g_N_upper2 = (I_n_box_right.T * (-I_r_P_upper2_box_right))[0]
    g_N_lower2 = (I_n_box_right.T * (-I_r_P_lower2_box_right))[0]
    g_N_ground_left = (I_n_ground.T * I_r_Obox_bottom_left)[0]
    g_N_ground_right = (I_n_ground.T * I_r_Obox_bottom_right)[0]

    g_N_vec = sp.Matrix([g_N_upper1, g_N_lower1, g_N_upper2, g_N_lower2, g_N_ground_left, g_N_ground_right])

    # Normal Jacobians (w_N)
    w_N_upper1 = sp.Matrix([g_N_upper1]).jacobian(q).T
    w_N_lower1 = sp.Matrix([g_N_lower1]).jacobian(q).T
    w_N_upper2 = sp.Matrix([g_N_upper2]).jacobian(q).T
    w_N_lower2 = sp.Matrix([g_N_lower2]).jacobian(q).T
    w_N_ground_left = sp.Matrix([g_N_ground_left]).jacobian(q).T
    w_N_ground_right = sp.Matrix([g_N_ground_right]).jacobian(q).T

    # Tangents
    I_t_left = I_e_y_B_box
    I_t_right = I_e_y_B_box
    I_t_ground = sp.Matrix([1, 0])

    # Contact Velocities
    I_v_box_left = I_r_Obox_left.jacobian(q) * dq
    I_v_box_right = I_r_Obox_right.jacobian(q) * dq
    I_v_P_upper1 = I_r_OP_upper1.jacobian(q) * dq
    I_v_P_lower1 = I_r_OP_lower1.jacobian(q) * dq
    I_v_P_upper2 = I_r_OP_upper2.jacobian(q) * dq
    I_v_P_lower2 = I_r_OP_lower2.jacobian(q) * dq
    I_v_box_bottom_left = I_r_Obox_bottom_left.jacobian(q) * dq
    I_v_box_bottom_right = I_r_Obox_bottom_right.jacobian(q) * dq

    # Tangent Velocity Functions (gamma_T)
    gamma_T_upper1 = (I_t_left.T * (I_v_box_left - I_v_P_upper1))[0]
    gamma_T_lower1 = (I_t_left.T * (I_v_box_left - I_v_P_lower1))[0]
    gamma_T_upper2 = (I_t_right.T * (I_v_box_right - I_v_P_upper2))[0]
    gamma_T_lower2 = (I_t_right.T * (I_v_box_right - I_v_P_lower2))[0]
    gamma_T_ground_left = (I_t_ground.T * I_v_box_bottom_left)[0]
    gamma_T_ground_right = (I_t_ground.T * I_v_box_bottom_right)[0]

    gamma_T_vec = sp.Matrix([
        gamma_T_upper1, gamma_T_lower1, gamma_T_upper2, gamma_T_lower2,
        gamma_T_ground_left, gamma_T_ground_right
    ])

    # Tangent Jacobians (w_T)
    w_T_upper1 = sp.Matrix([gamma_T_upper1]).jacobian(dq).T
    w_T_lower1 = sp.Matrix([gamma_T_lower1]).jacobian(dq).T
    w_T_upper2 = sp.Matrix([gamma_T_upper2]).jacobian(dq).T
    w_T_lower2 = sp.Matrix([gamma_T_lower2]).jacobian(dq).T
    w_T_ground_left = sp.Matrix([gamma_T_ground_left]).jacobian(dq).T
    w_T_ground_right = sp.Matrix([gamma_T_ground_right]).jacobian(dq).T

    # Assemble W
    W = sp.Matrix.hstack(
        w_T_upper1, w_N_upper1, 
        w_T_lower1, w_N_lower1,
        w_T_upper2, w_N_upper2, 
        w_T_lower2, w_N_lower2,
        w_T_ground_left, w_N_ground_left,
        w_T_ground_right, w_N_ground_right
    )

    W_dot_term = (W.T * dq).jacobian(q) * dq

    # --- 5. Simplification & Code Generation ---
    print("Simplifying expressions (heavy computation)...")
    M = sp.simplify(M)
    gen_force = sp.simplify(gen_force)
    W = sp.simplify(W)
    g_N_vec = sp.simplify(g_N_vec)
    gamma_T_vec = sp.simplify(gamma_T_vec)
    W_dot_term = sp.simplify(W_dot_term)
    J_EE1 = sp.simplify(J_EE1)
    J_EE2 = sp.simplify(J_EE2)

    # --- 6. Forward Kinematics for Animation ---
    # Dictionary of key points to export
    fk_exports = {
        # Left Arm
        "get_pos_base_L": pos_base_L,
        "get_pos_joint_L2": pos_joint_L2,
        "get_pos_wrist_L": pos_wrist_L,
        "get_pos_EE1": pos_EE1,
        # Right Arm
        "get_pos_base_R": pos_base_R,
        "get_pos_joint_R2": pos_joint_R2,
        "get_pos_wrist_R": pos_wrist_R,
        "get_pos_EE2": pos_EE2,
        # Box
        "get_pos_box": pos_box,
        # End effector Jacobians
        "get_J_EE1": J_EE1,
        "get_J_EE2": J_EE2
    }

    print("Generating Code...")
    printer = NumPyPrinter()

    def export_func(name, expr, args):
        if hasattr(expr, 'shape') and (expr.shape[0] > 1 and expr.shape[1] == 1):
            expr_to_print = [x for x in expr]
            code_content = printer.doprint(expr_to_print)
            code = f"numpy.array({code_content})"
        else:
            code = printer.doprint(expr)

        unpack_q = "    q0, q1, q2, q3, q4, q5, q6, q7, q8 = q\n"
        unpack_dq = "    dq0, dq1, dq2, dq3, dq4, dq5, dq6, dq7, dq8 = dq\n"
        unpack_tau = "    tau0, tau1, tau2, tau3, tau4, tau5 = tau\n"
        
        return f"\ndef {name}({', '.join(args)}):\n{unpack_q}{unpack_dq}{unpack_tau}    return {code}\n"

    args_list = [
        "q", "dq", "tau",
        "l1", "l2", "lc1", "lc2", 
        "m1", "m2", "m_EE", 
        "theta1", "theta2", "theta_EE", 
        "w_box", "h_box", "m_box", "theta_box", 
        "w_EE", "h_EE", 
        "g", 
        "x_base_L", "y_base_L", "x_base_R", "y_base_R"
    ]

    output_dir = "class_files/systems/dynamics"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_filename = f"{output_dir}/dual_arm_box_dynamics_lib.py"

    with open(output_filename, "w") as f:
        f.write("import jax\n")
        f.write("import jax.numpy as jnp\n")
        f.write("import jax.numpy as numpy\n\n")
        
        f.write("# --- Auto-generated Dynamics for Dual Arm Box System --- \n")
        
        # Dynamics Functions
        f.write(export_func("get_W", W, args_list))
        f.write(export_func("get_W_dot_transpose_dqdt", W_dot_term, args_list))
        f.write(export_func("get_g_N", g_N_vec, args_list))
        f.write(export_func("get_gamma_T", gamma_T_vec, args_list))
        f.write(export_func("get_M", M, args_list))
        f.write(export_func("get_gen_force", gen_force, args_list))
        f.write(export_func("get_B_matrix", B, args_list))
        # write end effector Jacobians
        f.write(export_func("get_J_EE1", J_EE1, args_list))
        f.write(export_func("get_J_EE2", J_EE2, args_list))
        
        # Forward Kinematics Functions
        f.write("\n# --- Forward Kinematics for Animation --- \n")
        for name, expr in fk_exports.items():
            f.write(export_func(name, expr, args_list))

    print(f"Success! '{output_filename}' has been generated.")

if __name__ == "__main__":
    generate_dual_arm_box_dynamics()