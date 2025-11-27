import jax
import jax.numpy as jnp
import jax.numpy as numpy

# --- Auto-generated Dynamics from SymPy --- 

def get_W(q, dq, w, h, m_box, m_ball, ball_radius, theta_box, g):
    q0, q1, q2, q3, q4, q5, q6 = q
    dq0, dq1, dq2, dq3, dq4, dq5, dq6 = dq
    return numpy.array([[-numpy.sin(q6), -numpy.cos(q6), 0, 0, 0, 0, 0, 0], [numpy.cos(q6), -numpy.sin(q6), 0, 0, 0, 0, 0, 0], [0, 0, -numpy.sin(q6), numpy.cos(q6), 0, 0, 0, 0], [0, 0, numpy.cos(q6), numpy.sin(q6), 0, 0, 0, 0], [numpy.sin(q6), numpy.cos(q6), numpy.sin(q6), -numpy.cos(q6), 1, 0, 1, 0], [-numpy.cos(q6), numpy.sin(q6), -numpy.cos(q6), -numpy.sin(q6), 0, 1, 0, 1], [(1/2)*w, q0*numpy.sin(q6) - q1*numpy.cos(q6) - q4*numpy.sin(q6) + q5*numpy.cos(q6), -1/2*w, -q2*numpy.sin(q6) + q3*numpy.cos(q6) + q4*numpy.sin(q6) - q5*numpy.cos(q6), (1/2)*h*numpy.cos(q6) + (1/2)*w*numpy.sin(q6), (1/2)*h*numpy.sin(q6) - 1/2*w*numpy.cos(q6), (1/2)*h*numpy.cos(q6) - 1/2*w*numpy.sin(q6), (1/2)*h*numpy.sin(q6) + (1/2)*w*numpy.cos(q6)]])

def get_g_N(q, dq, w, h, m_box, m_ball, ball_radius, theta_box, g):
    q0, q1, q2, q3, q4, q5, q6 = q
    dq0, dq1, dq2, dq3, dq4, dq5, dq6 = dq
    return numpy.array([-ball_radius - q0*numpy.cos(q6) - q1*numpy.sin(q6) + q4*numpy.cos(q6) + q5*numpy.sin(q6) - 1/2*w, -ball_radius + q2*numpy.cos(q6) + q3*numpy.sin(q6) - q4*numpy.cos(q6) - q5*numpy.sin(q6) - 1/2*w, -1/2*h*numpy.cos(q6) + q5 - 1/2*w*numpy.sin(q6), -1/2*h*numpy.cos(q6) + q5 + (1/2)*w*numpy.sin(q6)])

def get_gamma_T(q, dq, w, h, m_box, m_ball, ball_radius, theta_box, g):
    q0, q1, q2, q3, q4, q5, q6 = q
    dq0, dq1, dq2, dq3, dq4, dq5, dq6 = dq
    return numpy.array([-dq0*numpy.sin(q6) + dq1*numpy.cos(q6) + dq4*numpy.sin(q6) - dq5*numpy.cos(q6) + (1/2)*dq6*w, -dq2*numpy.sin(q6) + dq3*numpy.cos(q6) + dq4*numpy.sin(q6) - dq5*numpy.cos(q6) - 1/2*dq6*w, dq4 + (1/2)*dq6*(h*numpy.cos(q6) + w*numpy.sin(q6)), dq4 + (1/2)*dq6*(h*numpy.cos(q6) - w*numpy.sin(q6))])

def get_M(q, dq, w, h, m_box, m_ball, ball_radius, theta_box, g):
    q0, q1, q2, q3, q4, q5, q6 = q
    dq0, dq1, dq2, dq3, dq4, dq5, dq6 = dq
    return numpy.array([[m_ball, 0, 0, 0, 0, 0, 0], [0, m_ball, 0, 0, 0, 0, 0], [0, 0, m_ball, 0, 0, 0, 0], [0, 0, 0, m_ball, 0, 0, 0], [0, 0, 0, 0, m_box, 0, 0], [0, 0, 0, 0, 0, m_box, 0], [0, 0, 0, 0, 0, 0, theta_box]])

def get_gen_force(q, dq, w, h, m_box, m_ball, ball_radius, theta_box, g):
    q0, q1, q2, q3, q4, q5, q6 = q
    dq0, dq1, dq2, dq3, dq4, dq5, dq6 = dq
    return numpy.array([0, 0, 0, 0, 0, -g*m_box, 0])

def get_B_r_P1ball1(q, dq, w, h, m_box, m_ball, ball_radius, theta_box, g):
    q0, q1, q2, q3, q4, q5, q6 = q
    dq0, dq1, dq2, dq3, dq4, dq5, dq6 = dq
    return numpy.array([(q0 - q4 + (1/2)*w*numpy.cos(q6))*numpy.cos(q6) + (q1 - q5 + (1/2)*w*numpy.sin(q6))*numpy.sin(q6), -(q0 - q4 + (1/2)*w*numpy.cos(q6))*numpy.sin(q6) + (q1 - q5 + (1/2)*w*numpy.sin(q6))*numpy.cos(q6)])

def get_B_r_P2ball2(q, dq, w, h, m_box, m_ball, ball_radius, theta_box, g):
    q0, q1, q2, q3, q4, q5, q6 = q
    dq0, dq1, dq2, dq3, dq4, dq5, dq6 = dq
    return numpy.array([(q2 - q4 - 1/2*w*numpy.cos(q6))*numpy.cos(q6) + (q3 - q5 - 1/2*w*numpy.sin(q6))*numpy.sin(q6), -(q2 - q4 - 1/2*w*numpy.cos(q6))*numpy.sin(q6) + (q3 - q5 - 1/2*w*numpy.sin(q6))*numpy.cos(q6)])
