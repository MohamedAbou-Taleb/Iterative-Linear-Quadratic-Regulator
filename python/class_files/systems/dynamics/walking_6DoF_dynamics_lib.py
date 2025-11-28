import jax
import jax.numpy as jnp
import jax.numpy as numpy

# --- Auto-generated Dynamics from SymPy for Walking 6DoF --- 

def get_W(q, dq, m_B, m_upper, m_lower, theta_upper, theta_lower, l_upper, l_lower, g):
    q0, q1, q2, q3, q4, q5 = q
    dq0, dq1, dq2, dq3, dq4, dq5 = dq
    return numpy.array([[1, 0, 1, 0], [0, 1, 0, 1], [l_lower*numpy.cos(q2 + q3) + l_upper*numpy.cos(q2), l_lower*numpy.sin(q2 + q3) + l_upper*numpy.sin(q2), 0, 0], [l_lower*numpy.cos(q2 + q3), l_lower*numpy.sin(q2 + q3), 0, 0], [0, 0, l_lower*numpy.cos(q4 + q5) + l_upper*numpy.cos(q4), l_lower*numpy.sin(q4 + q5) + l_upper*numpy.sin(q4)], [0, 0, l_lower*numpy.cos(q4 + q5), l_lower*numpy.sin(q4 + q5)]])

def get_g_N(q, dq, m_B, m_upper, m_lower, theta_upper, theta_lower, l_upper, l_lower, g):
    q0, q1, q2, q3, q4, q5 = q
    dq0, dq1, dq2, dq3, dq4, dq5 = dq
    return numpy.array([-l_lower*numpy.cos(q2 + q3) - l_upper*numpy.cos(q2) + q1, -l_lower*numpy.cos(q4 + q5) - l_upper*numpy.cos(q4) + q1])

def get_gamma_T(q, dq, m_B, m_upper, m_lower, theta_upper, theta_lower, l_upper, l_lower, g):
    q0, q1, q2, q3, q4, q5 = q
    dq0, dq1, dq2, dq3, dq4, dq5 = dq
    return numpy.array([dq0 + dq2*(l_lower*numpy.cos(q2 + q3) + l_upper*numpy.cos(q2)) + dq3*l_lower*numpy.cos(q2 + q3), dq0 + dq4*(l_lower*numpy.cos(q4 + q5) + l_upper*numpy.cos(q4)) + dq5*l_lower*numpy.cos(q4 + q5)])

def get_M(q, dq, m_B, m_upper, m_lower, theta_upper, theta_lower, l_upper, l_lower, g):
    q0, q1, q2, q3, q4, q5 = q
    dq0, dq1, dq2, dq3, dq4, dq5 = dq
    return numpy.array([[m_B + 2*m_lower + 2*m_upper, 0, (1/2)*l_upper*m_upper*numpy.cos(q2) + (1/2)*m_lower*(l_lower*numpy.cos(q2 + q3) + 2*l_upper*numpy.cos(q2)), (1/2)*l_lower*m_lower*numpy.cos(q2 + q3), (1/2)*l_upper*m_upper*numpy.cos(q4) + (1/2)*m_lower*(l_lower*numpy.cos(q4 + q5) + 2*l_upper*numpy.cos(q4)), (1/2)*l_lower*m_lower*numpy.cos(q4 + q5)], [0, m_B + 2*m_lower + 2*m_upper, (1/2)*l_upper*m_upper*numpy.sin(q2) + (1/2)*m_lower*(l_lower*numpy.sin(q2 + q3) + 2*l_upper*numpy.sin(q2)), (1/2)*l_lower*m_lower*numpy.sin(q2 + q3), (1/2)*l_upper*m_upper*numpy.sin(q4) + (1/2)*m_lower*(l_lower*numpy.sin(q4 + q5) + 2*l_upper*numpy.sin(q4)), (1/2)*l_lower*m_lower*numpy.sin(q4 + q5)], [(1/2)*l_upper*m_upper*numpy.cos(q2) + (1/2)*m_lower*(l_lower*numpy.cos(q2 + q3) + 2*l_upper*numpy.cos(q2)), (1/2)*l_upper*m_upper*numpy.sin(q2) + (1/2)*m_lower*(l_lower*numpy.sin(q2 + q3) + 2*l_upper*numpy.sin(q2)), (1/4)*l_lower**2*m_lower + l_lower*l_upper*m_lower*numpy.cos(q3) + l_upper**2*m_lower + (1/4)*l_upper**2*m_upper + theta_lower + theta_upper, (1/4)*l_lower**2*m_lower + (1/2)*l_lower*l_upper*m_lower*numpy.cos(q3) + theta_lower, 0, 0], [(1/2)*l_lower*m_lower*numpy.cos(q2 + q3), (1/2)*l_lower*m_lower*numpy.sin(q2 + q3), (1/4)*l_lower**2*m_lower + (1/2)*l_lower*l_upper*m_lower*numpy.cos(q3) + theta_lower, (1/4)*l_lower**2*m_lower + theta_lower, 0, 0], [(1/2)*l_upper*m_upper*numpy.cos(q4) + (1/2)*m_lower*(l_lower*numpy.cos(q4 + q5) + 2*l_upper*numpy.cos(q4)), (1/2)*l_upper*m_upper*numpy.sin(q4) + (1/2)*m_lower*(l_lower*numpy.sin(q4 + q5) + 2*l_upper*numpy.sin(q4)), 0, 0, (1/4)*l_lower**2*m_lower + l_lower*l_upper*m_lower*numpy.cos(q5) + l_upper**2*m_lower + (1/4)*l_upper**2*m_upper + theta_lower + theta_upper, (1/4)*l_lower**2*m_lower + (1/2)*l_lower*l_upper*m_lower*numpy.cos(q5) + theta_lower], [(1/2)*l_lower*m_lower*numpy.cos(q4 + q5), (1/2)*l_lower*m_lower*numpy.sin(q4 + q5), 0, 0, (1/4)*l_lower**2*m_lower + (1/2)*l_lower*l_upper*m_lower*numpy.cos(q5) + theta_lower, (1/4)*l_lower**2*m_lower + theta_lower]])

def get_f_cg(q, dq, m_B, m_upper, m_lower, theta_upper, theta_lower, l_upper, l_lower, g):
    q0, q1, q2, q3, q4, q5 = q
    dq0, dq1, dq2, dq3, dq4, dq5 = dq
    return numpy.array([(1/2)*dq2**2*l_upper*m_upper*numpy.sin(q2) + (1/2)*dq4**2*l_upper*m_upper*numpy.sin(q4) + (1/2)*m_lower*(dq2*(dq2*(l_lower*numpy.sin(q2 + q3) + 2*l_upper*numpy.sin(q2)) + dq3*l_lower*numpy.sin(q2 + q3)) + dq3*l_lower*(dq2 + dq3)*numpy.sin(q2 + q3)) + (1/2)*m_lower*(dq4*(dq4*(l_lower*numpy.sin(q4 + q5) + 2*l_upper*numpy.sin(q4)) + dq5*l_lower*numpy.sin(q4 + q5)) + dq5*l_lower*(dq4 + dq5)*numpy.sin(q4 + q5)), -1/2*dq2**2*l_lower*m_lower*numpy.cos(q2 + q3) - dq2**2*l_upper*m_lower*numpy.cos(q2) - 1/2*dq2**2*l_upper*m_upper*numpy.cos(q2) - dq2*dq3*l_lower*m_lower*numpy.cos(q2 + q3) - 1/2*dq3**2*l_lower*m_lower*numpy.cos(q2 + q3) - 1/2*dq4**2*l_lower*m_lower*numpy.cos(q4 + q5) - dq4**2*l_upper*m_lower*numpy.cos(q4) - 1/2*dq4**2*l_upper*m_upper*numpy.cos(q4) - dq4*dq5*l_lower*m_lower*numpy.cos(q4 + q5) - 1/2*dq5**2*l_lower*m_lower*numpy.cos(q4 + q5) - g*m_B - 2*g*m_lower - 2*g*m_upper, dq2*dq3*l_lower*l_upper*m_lower*numpy.sin(q3) + (1/2)*dq3**2*l_lower*l_upper*m_lower*numpy.sin(q3) - 1/2*g*l_lower*m_lower*numpy.sin(q2 + q3) - g*l_upper*m_lower*numpy.sin(q2) - 1/2*g*l_upper*m_upper*numpy.sin(q2), -1/2*l_lower*m_lower*(dq2**2*l_upper*numpy.sin(q3) + g*numpy.sin(q2 + q3)), dq4*dq5*l_lower*l_upper*m_lower*numpy.sin(q5) + (1/2)*dq5**2*l_lower*l_upper*m_lower*numpy.sin(q5) - 1/2*g*l_lower*m_lower*numpy.sin(q4 + q5) - g*l_upper*m_lower*numpy.sin(q4) - 1/2*g*l_upper*m_upper*numpy.sin(q4), -1/2*l_lower*m_lower*(dq4**2*l_upper*numpy.sin(q5) + g*numpy.sin(q4 + q5))])

def get_B(q, dq, m_B, m_upper, m_lower, theta_upper, theta_lower, l_upper, l_lower, g):
    q0, q1, q2, q3, q4, q5 = q
    dq0, dq1, dq2, dq3, dq4, dq5 = dq
    return numpy.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
