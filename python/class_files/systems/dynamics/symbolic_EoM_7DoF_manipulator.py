import sympy as sp
from sympy.printing.numpy import NumPyPrinter

def generate_jax_dynamics():
    # --- 1. Define Symbols ---
    q_list = list(sp.symbols('q0:7'))
    dq_list = list(sp.symbols('dq0:7'))
    
    q = sp.Matrix(q_list)
    dq = sp.Matrix(dq_list)
    
    # Parameters
    w, h = sp.symbols('w h')
    m_box, m_ball, theta_box, ball_radius = sp.symbols('m_box m_ball theta_box ball_radius')
    g = sp.symbols('g')

    # Unpack state
    phi = q[6] 
    
    # --- 2. Geometry & Kinematics ---
    A_IB = sp.Matrix([
        [sp.cos(phi), -sp.sin(phi)],
        [sp.sin(phi),  sp.cos(phi)]
    ])

    I_r_Oball1 = sp.Matrix([q[0], q[1]])
    I_r_Oball2 = sp.Matrix([q[2], q[3]])
    I_r_Obox   = sp.Matrix([q[4], q[5]])

    I_e_x_B = A_IB.col(0)
    I_e_y_B = A_IB.col(1)

    # Contact points
    I_r_P1ball1 = I_r_Oball1 - (I_r_Obox - w/2 * I_e_x_B)
    I_r_P2ball2 = I_r_Oball2 - (I_r_Obox + w/2 * I_e_x_B)
    B_r_P1ball1 = A_IB.T @ I_r_P1ball1
    B_r_P2ball2 = A_IB.T @ I_r_P2ball2
    I_r_OP3 = I_r_Obox - w/2 * I_e_x_B - h/2 * I_e_y_B
    I_r_OP4 = I_r_Obox + w/2 * I_e_x_B - h/2 * I_e_y_B

    # Normals
    B_n_1 = sp.Matrix([-1, 0])
    B_n_2 = sp.Matrix([1, 0])
    I_n_1 = A_IB * B_n_1
    I_n_2 = A_IB * B_n_2
    I_n_3 = sp.Matrix([0, 1])
    I_n_4 = sp.Matrix([0, 1])

    # --- 3. Gap Functions ---
    g_N1 = (I_r_P1ball1.T * I_n_1)[0] - ball_radius
    g_N2 = (I_r_P2ball2.T * I_n_2)[0] - ball_radius
    g_N3 = (I_r_OP3.T * I_n_3)[0]
    g_N4 = (I_r_OP4.T * I_n_4)[0]

    g_N_vec = sp.Matrix([g_N1, g_N2, g_N3, g_N4])

    w_N1 = sp.Matrix([g_N1]).jacobian(q).T
    w_N2 = sp.Matrix([g_N2]).jacobian(q).T
    w_N3 = sp.Matrix([g_N3]).jacobian(q).T
    w_N4 = sp.Matrix([g_N4]).jacobian(q).T

    # --- 4. Tangential Velocities ---
    I_t_1 = I_e_y_B
    I_t_2 = I_e_y_B
    I_t_3 = sp.Matrix([1, 0])
    I_t_4 = sp.Matrix([1, 0])

    I_v_P1 = (I_r_Obox - w/2 * I_e_x_B).jacobian(q) * dq
    I_v_P2 = (I_r_Obox + w/2 * I_e_x_B).jacobian(q) * dq
    I_v_P3 = I_r_OP3.jacobian(q) * dq
    I_v_P4 = I_r_OP4.jacobian(q) * dq
    
    I_v_ball1 = I_r_Oball1.jacobian(q) * dq
    I_v_ball2 = I_r_Oball2.jacobian(q) * dq

    gamma_T1 = (I_t_1.T * (I_v_ball1 - I_v_P1))[0]
    gamma_T2 = (I_t_2.T * (I_v_ball2 - I_v_P2))[0]
    gamma_T3 = (I_t_3.T * I_v_P3)[0]
    gamma_T4 = (I_t_4.T * I_v_P4)[0]

    gamma_T_vec = sp.Matrix([gamma_T1, gamma_T2, gamma_T3, gamma_T4])

    w_T1 = sp.Matrix([gamma_T1]).jacobian(dq).T
    w_T2 = sp.Matrix([gamma_T2]).jacobian(dq).T
    w_T3 = sp.Matrix([gamma_T3]).jacobian(dq).T
    w_T4 = sp.Matrix([gamma_T4]).jacobian(dq).T

    # --- 5. Assemble Matrices ---
    W = sp.Matrix.hstack(w_T1, w_N1, w_T2, w_N2, w_T3, w_N3, w_T4, w_N4)
    
    M = sp.diag(
        m_ball, m_ball, m_ball, m_ball, 
        m_box, m_box,                   
        theta_box                       
    )

    gen_force = sp.Matrix([0, 0, 0, 0, 0, -m_box*g, 0])

    # --- 6. Code Generation ---
    print("Simplifying expressions...")
    W = sp.simplify(W)
    g_N_vec = sp.simplify(g_N_vec)
    gamma_T_vec = sp.simplify(gamma_T_vec)

    printer = NumPyPrinter()

    def export_func(name, expr, args):
        # --- LOGIC TO FLATTEN VECTORS ---
        # If the SymPy object is a Matrix with 1 column (Nx1),
        # convert it to a Python list before printing.
        # This forces NumPyPrinter to output "[a, b, c]" instead of "[[a], [b], [c]]"
        if hasattr(expr, 'shape') and expr.shape[1] == 1:
            # Flatten to list
            expr_to_print = [x for x in expr]
            # Print the list content -> "[eqn1, eqn2...]"
            code_content = printer.doprint(expr_to_print)
            # Wrap in array -> "numpy.array([...])"
            code = f"numpy.array({code_content})"
        else:
            # It's a full matrix (like W or M), keep 2D structure
            code = printer.doprint(expr)

        # Unpacking logic (Crucial for execution)
        unpack_q = "    q0, q1, q2, q3, q4, q5, q6 = q\n"
        unpack_dq = "    dq0, dq1, dq2, dq3, dq4, dq5, dq6 = dq\n"
        
        return f"\ndef {name}({', '.join(args)}):\n{unpack_q}{unpack_dq}    return {code}\n"

    args_list = "q, dq, w, h, m_box, m_ball, ball_radius, theta_box, g"

    with open("class_files/systems/dynamics/point_box_7DoF_dynamics_lib.py", "w") as f:
        f.write("import jax\n")
        f.write("import jax.numpy as jnp\n")
        f.write("import jax.numpy as numpy\n\n")
        
        f.write("# --- Auto-generated Dynamics from SymPy --- \n")
        f.write(export_func("get_W", W, [args_list]))
        f.write(export_func("get_g_N", g_N_vec, [args_list]))
        f.write(export_func("get_gamma_T", gamma_T_vec, [args_list]))
        f.write(export_func("get_M", M, [args_list]))
        f.write(export_func("get_gen_force", gen_force, [args_list]))
        f.write(export_func("get_B_r_P1ball1", B_r_P1ball1, [args_list]))
        f.write(export_func("get_B_r_P2ball2", B_r_P2ball2, [args_list]))

    print("Success! 'dynamics_lib.py' has been generated with 1D vectors.")

if __name__ == "__main__":
    generate_jax_dynamics()