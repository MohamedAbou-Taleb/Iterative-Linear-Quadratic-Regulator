import sympy as sp
from sympy.printing.numpy import NumPyPrinter

def generate_walking_dynamics():
    # --- 1. Define Symbols ---
    # 7 DoF: x_MB, y_MB, theta_B, q_hip1, q_knee1, q_hip2, q_knee2
    q_list = list(sp.symbols('q0:7'))
    dq_list = list(sp.symbols('dq0:7'))
    
    q = sp.Matrix(q_list)
    dq = sp.Matrix(dq_list)
    
    # Parameters
    m_B, m_upper, m_lower = sp.symbols('m_B m_upper m_lower')
    theta_B = sp.symbols('theta_B') # Base Inertia
    theta_upper, theta_lower = sp.symbols('theta_upper theta_lower')
    l_upper, l_lower = sp.symbols('l_upper l_lower')
    g = sp.symbols('g')

    # Mapping for clarity:
    # q[0]=x, q[1]=y, q[2]=theta_B (Base Pitch)
    # q[3]=Hip1, q[4]=Knee1 (Relative)
    # q[5]=Hip2, q[6]=Knee2 (Relative)

    # --- 2. Geometry & Kinematics ---
    I_r_Omb = sp.Matrix([q[0], q[1]])
    theta_base = q[2]

    # Absolute Angles (Base + Relative)
    abs_u1 = theta_base + q[3]
    abs_l1 = theta_base + q[3] + q[4]
    abs_u2 = theta_base + q[5]
    abs_l2 = theta_base + q[5] + q[6]

    # Helper for rotation vectors
    def get_vec(angle, length):
        return sp.Matrix([length * sp.sin(angle), -length * sp.cos(angle)])

    # Joints (Hips attached to Base Frame)
    I_r_Ojoint1 = I_r_Omb 
    I_r_Ojoint2 = I_r_Omb 

    # COMs
    I_r_Oupper1 = I_r_Ojoint1 + get_vec(abs_u1, l_upper/2)
    I_r_Olower1 = I_r_Ojoint1 + get_vec(abs_u1, l_upper) + get_vec(abs_l1, l_lower/2)
    
    I_r_Oupper2 = I_r_Ojoint2 + get_vec(abs_u2, l_upper/2)
    I_r_Olower2 = I_r_Ojoint2 + get_vec(abs_u2, l_upper) + get_vec(abs_l2, l_lower/2)

    # Feet
    I_r_Ofoot1 = I_r_Ojoint1 + get_vec(abs_u1, l_upper) + get_vec(abs_l1, l_lower)
    I_r_Ofoot2 = I_r_Ojoint2 + get_vec(abs_u2, l_upper) + get_vec(abs_l2, l_lower)

    # --- 3. Velocities & Jacobians ---
    # Angular Velocities (Vectors for cross products)
    d_abs_B = dq[2]
    d_abs_u1 = dq[2] + dq[3]
    d_abs_l1 = dq[2] + dq[3] + dq[4]
    d_abs_u2 = dq[2] + dq[5]
    d_abs_l2 = dq[2] + dq[5] + dq[6]

    Omega_B = sp.Matrix([0, 0, d_abs_B])
    Omega_upper1 = sp.Matrix([0, 0, d_abs_u1])
    Omega_lower1 = sp.Matrix([0, 0, d_abs_l1])
    Omega_upper2 = sp.Matrix([0, 0, d_abs_u2])
    Omega_lower2 = sp.Matrix([0, 0, d_abs_l2])

    # Linear Jacobians
    J_mb = I_r_Omb.jacobian(q)
    J_s_upper1 = I_r_Oupper1.jacobian(q)
    J_s_lower1 = I_r_Olower1.jacobian(q)
    J_s_upper2 = I_r_Oupper2.jacobian(q)
    J_s_lower2 = I_r_Olower2.jacobian(q)

    # Rotational Jacobians (dOmega/ddq)
    J_R_B = Omega_B.jacobian(dq)
    J_R_upper1 = Omega_upper1.jacobian(dq)
    J_R_lower1 = Omega_lower1.jacobian(dq)
    J_R_upper2 = Omega_upper2.jacobian(dq)
    J_R_lower2 = Omega_lower2.jacobian(dq)

    # --- 4. Mass Matrix (M) ---
    # YOUR METHOD: Project Inertias using Jacobians
    Theta_B_mat = sp.diag(0, 0, theta_B)
    Theta_upper_mat = sp.diag(0, 0, theta_upper)
    Theta_lower_mat = sp.diag(0, 0, theta_lower)

    M = (J_mb.T * m_B * J_mb + J_R_B.T * Theta_B_mat * J_R_B +
         J_s_upper1.T * m_upper * J_s_upper1 + J_R_upper1.T * Theta_upper_mat * J_R_upper1 +
         J_s_lower1.T * m_lower * J_s_lower1 + J_R_lower1.T * Theta_lower_mat * J_R_lower1 +
         J_s_upper2.T * m_upper * J_s_upper2 + J_R_upper2.T * Theta_upper_mat * J_R_upper2 +
         J_s_lower2.T * m_lower * J_s_lower2 + J_R_lower2.T * Theta_lower_mat * J_R_lower2)

    M = sp.simplify(M)

    # --- 5. Coriolis and Gravity Forces (f_cg) ---
    # Coriolis: J^T * d(v)/dq * dq
    def get_coriolis_term(J, v_lin):
        return J.T * v_lin.jacobian(q) * dq

    # FIXED: Added mass multipliers to prevent "rocketing"
    f_c = -(m_upper * get_coriolis_term(J_s_upper1, J_s_upper1 * dq) +
            m_lower * get_coriolis_term(J_s_lower1, J_s_lower1 * dq) +
            m_upper * get_coriolis_term(J_s_upper2, J_s_upper2 * dq) +
            m_lower * get_coriolis_term(J_s_lower2, J_s_lower2 * dq))
            # Note: Rotational coriolis is 0 in 2D planar case

    # Gravity term
    I_g_vec = sp.Matrix([0, -g])
    f_g = (J_mb.T       * m_B     * I_g_vec + 
           J_s_upper1.T * m_upper * I_g_vec +
           J_s_upper2.T * m_upper * I_g_vec +
           J_s_lower1.T * m_lower * I_g_vec +
           J_s_lower2.T * m_lower * I_g_vec)

    f_cg = sp.simplify(f_c + f_g)

    # --- 6. Actuation Matrix (B) ---
    # 7-DoF: q = [x, y, theta, hip1, knee1, hip2, knee2]
    # Inputs apply to relative joints q3, q4, q5, q6
    B = sp.Matrix.zeros(7, 4)
    B[3, 0] = 1 # Hip 1
    B[4, 1] = 1 # Knee 1
    B[5, 2] = 1 # Hip 2
    B[6, 3] = 1 # Knee 2

    # --- 7. Contact Dynamics ---
    g_N_vec = sp.Matrix([I_r_Ofoot1[1], I_r_Ofoot2[1]]) # y positions

    w_N1 = sp.Matrix([I_r_Ofoot1[1]]).jacobian(q).T
    w_N2 = sp.Matrix([I_r_Ofoot2[1]]).jacobian(q).T

    v_foot1 = I_r_Ofoot1.jacobian(q) * dq
    v_foot2 = I_r_Ofoot2.jacobian(q) * dq
    
    gamma_T_vec = sp.Matrix([v_foot1[0], v_foot2[0]])

    w_T1 = sp.Matrix([v_foot1[0]]).jacobian(dq).T
    w_T2 = sp.Matrix([v_foot2[0]]).jacobian(dq).T

    W = sp.Matrix.hstack(w_T1, w_N1, w_T2, w_N2)

    # --- 8. Code Generation ---
    print("Simplifying expressions...")
    W = sp.simplify(W)
    g_N_vec = sp.simplify(g_N_vec)
    gamma_T_vec = sp.simplify(gamma_T_vec)
    f_cg = sp.simplify(f_cg)

    printer = NumPyPrinter()

    def export_func(name, expr, args):
        if hasattr(expr, 'shape') and expr.shape[1] == 1:
            expr_to_print = [x for x in expr]
            code_content = printer.doprint(expr_to_print)
            code = f"numpy.array({code_content})"
        else:
            code = printer.doprint(expr)
        unpack_q = "    q0, q1, q2, q3, q4, q5, q6 = q\n"
        unpack_dq = "    dq0, dq1, dq2, dq3, dq4, dq5, dq6 = dq\n"
        return f"\ndef {name}({', '.join(args)}):\n{unpack_q}{unpack_dq}    return {code}\n"

    args_list = "q, dq, m_B, theta_B, m_upper, m_lower, theta_upper, theta_lower, l_upper, l_lower, g"

    print("Generating file...")
    with open("class_files/systems/dynamics/walking_7DoF_dynamics_lib.py", "w") as f:
        f.write("import jax.numpy as numpy\n\n")
        f.write("# --- Auto-generated Dynamics for Walking 7DoF --- \n")
        f.write(export_func("get_W", W, [args_list]))
        f.write(export_func("get_g_N", g_N_vec, [args_list]))
        f.write(export_func("get_gamma_T", gamma_T_vec, [args_list]))
        f.write(export_func("get_M", M, [args_list]))
        f.write(export_func("get_f_cg", f_cg, [args_list])) 
        f.write(export_func("get_B", B, [args_list]))

    print("Success.")

if __name__ == "__main__":
    generate_walking_dynamics()