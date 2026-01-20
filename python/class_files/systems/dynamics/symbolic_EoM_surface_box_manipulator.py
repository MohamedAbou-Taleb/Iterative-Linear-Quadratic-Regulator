import sympy as sp
from sympy.printing.numpy import NumPyPrinter

def generate_surface_box_dynamics():
    # --- 1. Define Symbols ---
    # Generalized coordinates: 9 DoF
    # q0-q2: EE1 (x, y, phi)
    # q3-q5: EE2 (x, y, phi)
    # q6-q8: Box (x, y, phi)
    q_list = list(sp.symbols('q0:9'))
    dq_list = list(sp.symbols('dq0:9'))
    
    q = sp.Matrix(q_list)
    dq = sp.Matrix(dq_list)
    
    # Parameters
    w_box, h_box = sp.symbols('w_box h_box')
    w_EE, h_EE = sp.symbols('w_EE h_EE')
    m_box, m_EE, theta_box, theta_EE = sp.symbols('m_box m_EE theta_box theta_EE')
    g = sp.symbols('g')

    # Unpack state for clarity (matching MATLAB variable names)
    x_EE1, y_EE1, phi_EE1 = q[0], q[1], q[2]
    x_EE2, y_EE2, phi_EE2 = q[3], q[4], q[5]
    x_box, y_box, phi_box = q[6], q[7], q[8]

    # --- 2. Geometry & Kinematics ---
    def A_IB_fcn(phi):
        return sp.Matrix([
            [sp.cos(phi), -sp.sin(phi)],
            [sp.sin(phi),  sp.cos(phi)]
        ])

    A_IB_EE1 = A_IB_fcn(phi_EE1)
    A_IB_EE2 = A_IB_fcn(phi_EE2)
    A_IB_box = A_IB_fcn(phi_box)

    I_r_OEE1 = sp.Matrix([x_EE1, y_EE1])
    I_r_OEE2 = sp.Matrix([x_EE2, y_EE2])
    I_r_Obox = sp.Matrix([x_box, y_box])

    # Basis vectors
    I_e_x_B_EE1 = A_IB_EE1.col(0)
    I_e_y_B_EE1 = A_IB_EE1.col(1)
    I_e_x_B_EE2 = A_IB_EE2.col(0)
    I_e_y_B_EE2 = A_IB_EE2.col(1)
    I_e_x_B_box = A_IB_box.col(0)
    I_e_y_B_box = A_IB_box.col(1)

    # Box Geometry
    I_r_Obox_left  = I_r_Obox - w_box/2 * I_e_x_B_box
    I_r_Obox_right = I_r_Obox + w_box/2 * I_e_x_B_box
    
    I_r_Obox_bottom_left  = I_r_Obox_left  - h_box/2 * I_e_y_B_box
    I_r_Obox_bottom_right = I_r_Obox_right - h_box/2 * I_e_y_B_box

    # EE1 Surface Points (SP)
    B_EE1_r_SP_upper1 = sp.Matrix([w_EE/2, h_EE/2])
    I_r_SP_upper1 = A_IB_EE1 * B_EE1_r_SP_upper1
    
    B_EE1_r_SP_lower1 = sp.Matrix([w_EE/2, -h_EE/2])
    I_r_SP_lower1 = A_IB_EE1 * B_EE1_r_SP_lower1

    # EE2 Surface Points (SP)
    B_EE2_r_SP_upper2 = sp.Matrix([-w_EE/2, h_EE/2])
    I_r_SP_upper2 = A_IB_EE2 * B_EE2_r_SP_upper2
    
    B_EE2_r_SP_lower2 = sp.Matrix([-w_EE/2, -h_EE/2])
    I_r_SP_lower2 = A_IB_EE2 * B_EE2_r_SP_lower2

    # Absolute locations of EE contact points
    I_r_OP_upper1 = I_r_OEE1 + I_r_SP_upper1
    I_r_OP_lower1 = I_r_OEE1 + I_r_SP_lower1
    I_r_OP_upper2 = I_r_OEE2 + I_r_SP_upper2
    I_r_OP_lower2 = I_r_OEE2 + I_r_SP_lower2

    # Relative vectors (EE point to Box Center/Side)
    I_r_P_upper1_box_left = I_r_Obox_left - I_r_OP_upper1
    I_r_P_lower1_box_left = I_r_Obox_left - I_r_OP_lower1
    I_r_P_upper2_box_right = I_r_Obox_right - I_r_OP_upper2
    I_r_P_lower2_box_right = I_r_Obox_right - I_r_OP_lower2

    # --- 3. Normals and Gaps ---
    # Box normals (outward facing from the box center)
    I_n_box_left = -I_e_x_B_box
    I_n_box_right = I_e_x_B_box
    
    I_n_ground_left = sp.Matrix([0, 1])
    I_n_ground_right = sp.Matrix([0, 1])

    # Gap Functions (g_N)
    # Note: MATLAB uses I_n' * (-I_r_P...).
    # -I_r_P... = -(Box - EE) = EE - Box.
    # So Gap = Normal . (EE_Point - Box_Surface_Point)
    g_N_upper1 = (I_n_box_left.T * (-I_r_P_upper1_box_left))[0]
    g_N_lower1 = (I_n_box_left.T * (-I_r_P_lower1_box_left))[0]
    g_N_upper2 = (I_n_box_right.T * (-I_r_P_upper2_box_right))[0]
    g_N_lower2 = (I_n_box_right.T * (-I_r_P_lower2_box_right))[0]

    g_N_ground_left = (I_n_ground_left.T * I_r_Obox_bottom_left)[0]
    g_N_ground_right = (I_n_ground_right.T * I_r_Obox_bottom_right)[0]

    # Collect gaps
    g_N_vec = sp.Matrix([
        g_N_upper1, g_N_lower1, 
        g_N_upper2, g_N_lower2, 
        g_N_ground_left, g_N_ground_right
    ])

    # Normal Jacobians (w_N)
    w_N_upper1 = sp.Matrix([g_N_upper1]).jacobian(q).T
    w_N_lower1 = sp.Matrix([g_N_lower1]).jacobian(q).T
    w_N_upper2 = sp.Matrix([g_N_upper2]).jacobian(q).T
    w_N_lower2 = sp.Matrix([g_N_lower2]).jacobian(q).T
    w_N_ground_left = sp.Matrix([g_N_ground_left]).jacobian(q).T
    w_N_ground_right = sp.Matrix([g_N_ground_right]).jacobian(q).T

    # --- 4. Tangents and Velocities ---
    # Tangent vectors
    I_t_left = I_e_y_B_box
    I_t_right = I_e_y_B_box
    I_t_ground = sp.Matrix([1, 0])

    # Velocities
    # Note: SymPy jacobian(q) * dq gives the time derivative d/dt
    I_v_box_left = I_r_Obox_left.jacobian(q) * dq
    I_v_box_right = I_r_Obox_right.jacobian(q) * dq

    I_v_P_upper1 = I_r_OP_upper1.jacobian(q) * dq
    I_v_P_lower1 = I_r_OP_lower1.jacobian(q) * dq
    I_v_P_upper2 = I_r_OP_upper2.jacobian(q) * dq
    I_v_P_lower2 = I_r_OP_lower2.jacobian(q) * dq

    I_v_box_bottom_left = I_r_Obox_bottom_left.jacobian(q) * dq
    I_v_box_bottom_right = I_r_Obox_bottom_right.jacobian(q) * dq

    # Tangential Velocity Functions (gamma_T)
    # gamma = Tangent . (V_Box - V_EE)
    gamma_T_upper1 = (I_t_left.T * (I_v_box_left - I_v_P_upper1))[0]
    gamma_T_lower1 = (I_t_left.T * (I_v_box_left - I_v_P_lower1))[0]
    gamma_T_upper2 = (I_t_right.T * (I_v_box_right - I_v_P_upper2))[0]
    gamma_T_lower2 = (I_t_right.T * (I_v_box_right - I_v_P_lower2))[0]

    gamma_T_ground_left = (I_t_ground.T * I_v_box_bottom_left)[0]
    gamma_T_ground_right = (I_t_ground.T * I_v_box_bottom_right)[0]

    gamma_T_vec = sp.Matrix([
        gamma_T_upper1, gamma_T_lower1,
        gamma_T_upper2, gamma_T_lower2,
        gamma_T_ground_left, gamma_T_ground_right
    ])

    # Tangential Jacobians (w_T) - derivative w.r.t dq
    w_T_upper1 = sp.Matrix([gamma_T_upper1]).jacobian(dq).T
    w_T_lower1 = sp.Matrix([gamma_T_lower1]).jacobian(dq).T
    w_T_upper2 = sp.Matrix([gamma_T_upper2]).jacobian(dq).T
    w_T_lower2 = sp.Matrix([gamma_T_lower2]).jacobian(dq).T
    w_T_ground_left = sp.Matrix([gamma_T_ground_left]).jacobian(dq).T
    w_T_ground_right = sp.Matrix([gamma_T_ground_right]).jacobian(dq).T

    # --- 5. Assemble Matrices ---
    # W Matrix: Interleave T and N for each contact point
    # Order: [T_u1, N_u1, T_l1, N_l1, T_u2, N_u2, T_l2, N_l2, T_gl, N_gl, T_gr, N_gr]
    W = sp.Matrix.hstack(
        w_T_upper1, w_N_upper1, 
        w_T_lower1, w_N_lower1,
        w_T_upper2, w_N_upper2, 
        w_T_lower2, w_N_lower2,
        w_T_ground_left, w_N_ground_left,
        w_T_ground_right, w_N_ground_right
    )
    # --- New Term: W_dot_transpose_dqdt ---
    # W_dot_transpose_dqdt = jacobian(W'*dq, q)
    W_dot_term = (W.T * dq).jacobian(q) @ dq

    # Mass Matrix M
    # Diag: m_EE, m_EE, theta_EE (for EE1), m_EE, m_EE, theta_EE (for EE2), m_box, m_box, theta_box (for Box)
    M = sp.diag(
        m_EE, m_EE, theta_EE,
        m_EE, m_EE, theta_EE,
        m_box, m_box, theta_box
    )

    # Generalized Forces (Gravity on box only)
    # q indices: 0-2 (EE1), 3-5 (EE2), 6 (x_box), 7 (y_box), 8 (phi_box)
    gen_force = sp.Matrix([0, 0, 0, 0, 0, 0, 0, -m_box*g, 0])

    # --- 6. Code Generation ---
    print("Simplifying expressions (this may take a moment)...")
    W = sp.simplify(W)
    W_dot_term = sp.simplify(W_dot_term)
    g_N_vec = sp.simplify(g_N_vec)
    gamma_T_vec = sp.simplify(gamma_T_vec)
    M = sp.simplify(M)
    gen_force = sp.simplify(gen_force)

    print("Generating code...")
    printer = NumPyPrinter()

    def export_func(name, expr, args):
        # Flatten logic for 1D vectors (Nx1 matrices)
        if hasattr(expr, 'shape') and expr.shape[1] == 1:
            expr_to_print = [x for x in expr]
            code_content = printer.doprint(expr_to_print)
            code = f"numpy.array({code_content})"
        else:
            code = printer.doprint(expr)

        # Unpacking logic for 9 DoF
        unpack_q = "    q0, q1, q2, q3, q4, q5, q6, q7, q8 = q\n"
        unpack_dq = "    dq0, dq1, dq2, dq3, dq4, dq5, dq6, dq7, dq8 = dq\n"
        
        return f"\ndef {name}({', '.join(args)}):\n{unpack_q}{unpack_dq}    return {code}\n"

    args_list = "q, dq, w_box, h_box, w_EE, h_EE, m_box, m_EE, theta_box, theta_EE, g"
    
    # Define output filename
    output_filename = "class_files/systems/dynamics/surface_box_dynamics_lib.py"

    with open(output_filename, "w") as f:
        f.write("import jax\n")
        f.write("import jax.numpy as jnp\n")
        f.write("import jax.numpy as numpy\n\n")
        
        f.write("# --- Auto-generated Dynamics from SymPy for Surface Box Manipulator --- \n")
        f.write(export_func("get_W", W, [args_list]))
        f.write(export_func("get_W_dot_transpose_dqdt", W_dot_term, [args_list]))
        f.write(export_func("get_g_N", g_N_vec, [args_list]))
        f.write(export_func("get_gamma_T", gamma_T_vec, [args_list]))
        f.write(export_func("get_M", M, [args_list]))
        f.write(export_func("get_gen_force", gen_force, [args_list]))

    print(f"Success! '{output_filename}' has been generated.")

if __name__ == "__main__":
    generate_surface_box_dynamics()