import sympy as sp
from sympy.printing.numpy import NumPyPrinter

def generate_walking_dynamics():
    # --- 1. Define Symbols ---
    q_list = list(sp.symbols('q0:6'))   # 6 DoF: x, y, q_upper1, q_lower1, q_upper2, q_lower2
    dq_list = list(sp.symbols('dq0:6'))
    
    q = sp.Matrix(q_list)
    dq = sp.Matrix(dq_list)
    
    # Parameters
    m_B, m_upper, m_lower = sp.symbols('m_B m_upper m_lower')
    theta_upper, theta_lower = sp.symbols('theta_upper theta_lower')
    l_upper, l_lower = sp.symbols('l_upper l_lower')
    g = sp.symbols('g')

    # Mapping to MATLAB variable names for clarity
    # q[0] = x_MB, q[1] = y_MB
    # q[2] = q3 (upper1), q[3] = q4 (lower1 relative)
    # q[4] = q5 (upper2), q[5] = q6 (lower2 relative)

    # --- 2. Geometry & Kinematics ---
    I_r_Omb = sp.Matrix([q[0], q[1]])

    # Joint 1 (Leg 1)
    I_r_Ojoint1 = I_r_Omb + sp.Matrix([
        l_upper * sp.sin(q[2]), 
        -l_upper * sp.cos(q[2])
    ])
    
    # Joint 2 (Leg 2)
    I_r_Ojoint2 = I_r_Omb + sp.Matrix([
        l_upper * sp.sin(q[4]), 
        -l_upper * sp.cos(q[4])
    ])

    # COMs (Upper)
    I_r_Oupper1 = I_r_Omb + sp.Matrix([
        l_upper/2 * sp.sin(q[2]), 
        -l_upper/2 * sp.cos(q[2])
    ])
    I_r_Oupper2 = I_r_Omb + sp.Matrix([
        l_upper/2 * sp.sin(q[4]), 
        -l_upper/2 * sp.cos(q[4])
    ])

    # COMs (Lower)
    # Note: MATLAB uses (q3 + q4) for leg 1 and (q5 + q6) for leg 2
    angle_leg1 = q[2] + q[3]
    angle_leg2 = q[4] + q[5]

    I_r_Olower1 = I_r_Ojoint1 + sp.Matrix([
        l_lower/2 * sp.sin(angle_leg1), 
        -l_lower/2 * sp.cos(angle_leg1)
    ])
    I_r_Olower2 = I_r_Ojoint2 + sp.Matrix([
        l_lower/2 * sp.sin(angle_leg2), 
        -l_lower/2 * sp.cos(angle_leg2)
    ])

    # Feet
    I_r_Ofoot1 = I_r_Ojoint1 + sp.Matrix([
        l_lower * sp.sin(angle_leg1), 
        -l_lower * sp.cos(angle_leg1)
    ])
    I_r_Ofoot2 = I_r_Ojoint2 + sp.Matrix([
        l_lower * sp.sin(angle_leg2), 
        -l_lower * sp.cos(angle_leg2)
    ])

    # --- 3. Velocities & Jacobians ---
    # Angular Velocities (3D vectors for cross product consistency with Inertia)
    # q[2] is omega_upper1, q[4] is omega_upper2
    Omega_upper1 = sp.Matrix([0, 0, dq[2]])
    Omega_upper2 = sp.Matrix([0, 0, dq[4]])
    Omega_lower1 = sp.Matrix([0, 0, dq[2] + dq[3]])
    Omega_lower2 = sp.Matrix([0, 0, dq[4] + dq[5]])

    # Linear Jacobians (COMs)
    J_mb = I_r_Omb.jacobian(q)
    J_s_upper1 = I_r_Oupper1.jacobian(q)
    J_s_upper2 = I_r_Oupper2.jacobian(q)
    J_s_lower1 = I_r_Olower1.jacobian(q)
    J_s_lower2 = I_r_Olower2.jacobian(q)

    # Rotational Jacobians
    J_R_upper1 = Omega_upper1.jacobian(dq)
    J_R_upper2 = Omega_upper2.jacobian(dq)
    J_R_lower1 = Omega_lower1.jacobian(dq)
    J_R_lower2 = Omega_lower2.jacobian(dq)

    # --- 4. Mass Matrix (M) ---
    Theta_upper_mat = sp.diag(0, 0, theta_upper)
    Theta_lower_mat = sp.diag(0, 0, theta_lower)

    # Note: The MATLAB script provided omits the Base Mass (m_B) in the M calculation. 
    # We follow the MATLAB script strictly here.
    M = (J_mb.T * m_B * J_mb +
         J_s_upper1.T * m_upper * J_s_upper1 + J_R_upper1.T * Theta_upper_mat * J_R_upper1 +
         J_s_upper2.T * m_upper * J_s_upper2 + J_R_upper2.T * Theta_upper_mat * J_R_upper2 +
         J_s_lower1.T * m_lower * J_s_lower1 + J_R_lower1.T * Theta_lower_mat * J_R_lower1 +
         J_s_lower2.T * m_lower * J_s_lower2 + J_R_lower2.T * Theta_lower_mat * J_R_lower2)

    M = sp.simplify(M)

    # --- 5. Coriolis and Gravity Forces (f_cg) ---
    # Coriolis term: f_c = - sum( J^T * dJ/dt * dq )
    # We compute dJ/dt * dq as jacobian(v, q) * dq
    
    def get_coriolis_term(J, v_lin):
        # dJ_dt_dq is equivalent to jacobian(J*dq, q) * dq in the MATLAB script
        # J*dq is the velocity v_lin
        return J.T * v_lin.jacobian(q) * dq

    f_c = -(m_upper*get_coriolis_term(J_s_upper1, J_s_upper1 * dq) +
            m_upper*get_coriolis_term(J_s_upper2, J_s_upper2 * dq) +
            m_lower*get_coriolis_term(J_s_lower1, J_s_lower1 * dq) +
            m_lower*get_coriolis_term(J_s_lower2, J_s_lower2 * dq))

    # Gravity term
    I_g_vec = sp.Matrix([0, -g])
    f_g = (J_mb.T       * m_B     * I_g_vec + 
           J_s_upper1.T * m_upper * I_g_vec +
           J_s_upper2.T * m_upper * I_g_vec +
           J_s_lower1.T * m_lower * I_g_vec +
           J_s_lower2.T * m_lower * I_g_vec)

    f_cg = sp.simplify(f_c + f_g)

    # --- 6. Actuation Matrix (B) ---
    # Maps 4 inputs [u1, u2, u3, u4] to 6 Generalized Coordinates
    # u1->Hip1, u2->Knee1, u3->Hip2, u4->Knee2
    # This corresponds to q[2], q[3], q[4], q[5]
    B = sp.Matrix.zeros(6, 4)
    B[2, 0] = 1
    B[3, 1] = 1
    B[4, 2] = 1
    B[5, 3] = 1

    # --- 7. Contact Dynamics ---
    I_n_1 = sp.Matrix([0, 1])
    I_n_2 = sp.Matrix([0, 1])

    g_N1 = (I_n_1.T * I_r_Ofoot1)[0]
    g_N2 = (I_n_2.T * I_r_Ofoot2)[0]
    g_N_vec = sp.Matrix([g_N1, g_N2])

    w_N1 = sp.Matrix([g_N1]).jacobian(q).T
    w_N2 = sp.Matrix([g_N2]).jacobian(q).T

    I_t_1 = sp.Matrix([1, 0])
    I_t_2 = sp.Matrix([1, 0])

    I_v_foot1 = I_r_Ofoot1.jacobian(q) * dq
    I_v_foot2 = I_r_Ofoot2.jacobian(q) * dq

    gamma_T1 = (I_t_1.T * I_v_foot1)[0]
    gamma_T2 = (I_t_2.T * I_v_foot2)[0]
    gamma_T_vec = sp.Matrix([gamma_T1, gamma_T2])

    w_T1 = sp.Matrix([gamma_T1]).jacobian(dq).T
    w_T2 = sp.Matrix([gamma_T2]).jacobian(dq).T

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

        unpack_q = "    q0, q1, q2, q3, q4, q5 = q\n"
        unpack_dq = "    dq0, dq1, dq2, dq3, dq4, dq5 = dq\n"
        
        return f"\ndef {name}({', '.join(args)}):\n{unpack_q}{unpack_dq}    return {code}\n"

    args_list = "q, dq, m_B, m_upper, m_lower, theta_upper, theta_lower, l_upper, l_lower, g"

    print("Generating file...")
    with open("class_files/systems/dynamics/walking_6DoF_dynamics_lib.py", "w") as f:
        f.write("import jax\n")
        f.write("import jax.numpy as jnp\n")
        f.write("import jax.numpy as numpy\n\n")
        
        f.write("# --- Auto-generated Dynamics from SymPy for Walking 6DoF --- \n")
        f.write(export_func("get_W", W, [args_list]))
        f.write(export_func("get_g_N", g_N_vec, [args_list]))
        f.write(export_func("get_gamma_T", gamma_T_vec, [args_list]))
        f.write(export_func("get_M", M, [args_list]))
        # get_f_cg returns the nonlinear effects (Coriolis + Gravity)
        f.write(export_func("get_f_cg", f_cg, [args_list])) 
        # get_B returns the actuation matrix (mapping 4 inputs to 6 DoF)
        f.write(export_func("get_B", B, [args_list]))

    print("Success! 'walking_6DoF_dynamics_lib.py' has been generated.")

if __name__ == "__main__":
    generate_walking_dynamics()