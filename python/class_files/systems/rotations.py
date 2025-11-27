import numpy as np
from algebra import norm, cross3, ax2skew, ax2skew_a, LeviCivita3, ax2skew_squared
import jax.numpy as jnp

# for small angles we use first order approximations of the equations since
# most of the SO(3) and SE(3) equations get singular for psi -> 0.
# angle_singular = 1.0e-6
angle_singular = 0.0

eye3 = jnp.eye(3, dtype=float)


def Exp_SO3(psi: jnp.ndarray) -> jnp.ndarray:
    """SO(3) exponential function, see Crisfield1999 above (4.1) and 
    Park2005 (12).

    References
    ----------
    Crisfield1999: https://doi.org/10.1098/rspa.1999.0352 \\
    Park2005: https://doi.org/10.1109/TRO.2005.852253
    """
    angle = norm(psi)
    # if angle > angle_singular:
    # Park2005 (12)
    sa = jnp.sin(angle)
    ca = jnp.cos(angle)
    alpha = sa / angle
    beta2 = (1.0 - ca) / (angle * angle)
    # return eye3 + alpha * ax2skew(psi) + beta2 * ax2skew_squared(psi)
    # else:
    # first order approximation

    result = jnp.where(
        angle > angle_singular,
        eye3 + alpha * ax2skew(psi) + beta2 * ax2skew_squared(psi),
        eye3 + ax2skew(psi),
    )
    return result
    # return eye3 + ax2skew(psi)


def Exp_SO3_psi(psi: jnp.ndarray) -> jnp.ndarray:
    """Derivative of the axis-angle rotation found in Crisfield1999 above (4.1). 
    Derivations and final results are given in Gallego2015 (9).

    References
    ----------
    Crisfield1999: https://doi.org/10.1098/rspa.1999.0352 \\
    Gallego2015: https://doi.org/10.1007/s10851-014-0528-x
    """
    angle = norm(psi)

    A_psi = jnp.zeros((3, 3, 3), dtype=psi.dtype)
    if angle > angle_singular:
        angle2 = angle * angle
        sa = jnp.sin(angle)
        ca = jnp.cos(angle)
        alpha = sa / angle
        alpha_psik = (ca - alpha) / angle2
        beta = 2.0 * (1.0 - ca) / angle2
        beta2_psik = (alpha - beta) / angle2

        psi_tilde = ax2skew(psi)
        psi_tilde2 = ax2skew_squared(psi)

        ############################
        # alpha * psi_tilde (part I)
        ############################
        A_psi[0, 2, 1] = A_psi[1, 0, 2] = A_psi[2, 1, 0] = alpha
        A_psi[0, 1, 2] = A_psi[1, 2, 0] = A_psi[2, 0, 1] = -alpha

        #############################
        # alpha * psi_tilde (part II)
        #############################
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    A_psi[i, j, k] += psi_tilde[i, j] * psi[k] * alpha_psik

        ###############################
        # beta2 * psi_tilde @ psi_tilde
        ###############################
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    A_psi[i, j, k] += psi_tilde2[i, j] * psi[k] * beta2_psik
                    for l in range(3):
                        A_psi[i, j, k] += (
                            0.5
                            * beta
                            * (
                                LeviCivita3(k, l, i) * psi_tilde[l, j]
                                + psi_tilde[l, i] * LeviCivita3(k, l, j)
                            )
                        )
    else:
        ###################
        # alpha * psi_tilde
        ###################
        A_psi[0, 2, 1] = A_psi[1, 0, 2] = A_psi[2, 1, 0] = 1.0
        A_psi[0, 1, 2] = A_psi[1, 2, 0] = A_psi[2, 0, 1] = -1.0

    return A_psi


def Log_SO3(A: jnp.ndarray) -> jnp.ndarray:
    ca = 0.5 * (jnp.trace(A) - 1.0)
    ca = jnp.clip(ca, -1, 1)  # clip to [-1, 1] for arccos!
    angle = jnp.arccos(ca)

    # fmt: off
    psi = 0.5 * jnp.array([
        A[2, 1] - A[1, 2],
        A[0, 2] - A[2, 0],
        A[1, 0] - A[0, 1]
    ], dtype=A.dtype)
    # fmt: on

    # if angle > angle_singular and angle < jnp.pi:
    #     psi *= angle / jnp.sqrt(1.0 - ca * ca)
    #     dP_n = jnp.where(gap_val[i] > 0.0, 0.0, dP_n_contact)
    psi = jnp.where(
        (angle > angle_singular) & (angle < jnp.pi),
        psi * angle / jnp.sqrt(1.0 - ca * ca),
        psi,
    )
    # psi *= angle / jnp.sqrt(1.0 - ca * ca)
    return psi


def Log_SO3_A(A: jnp.ndarray) -> jnp.ndarray:
    """Derivative of the SO(3) Log map. See Blanco-Claraco2010 (10.11)

    References:
    ===========
    Blanco-Claraco2010: https://doi.org/10.48550/arXiv.2103.15980
    """
    ca = 0.5 * (jnp.trace(A) - 1.0)
    ca = jnp.clip(ca, -1, 1)  # clip to [-1, 1] for arccos!
    angle = jnp.arccos(ca)

    psi_A = jnp.zeros((3, 3, 3), dtype=A.dtype)
    if angle > angle_singular & angle < jnp.pi:
        sa = jnp.sin(angle)
        b = 0.5 * angle / sa

        # fmt: off
        a = (angle * ca - sa) / (4.0 * sa**3) * jnp.array([
            A[2, 1] - A[1, 2],
            A[0, 2] - A[2, 0],
            A[1, 0] - A[0, 1]
        ], dtype=A.dtype)
        # fmt: on

        psi_A[0, 0, 0] = psi_A[0, 1, 1] = psi_A[0, 2, 2] = a[0]
        psi_A[1, 0, 0] = psi_A[1, 1, 1] = psi_A[1, 2, 2] = a[1]
        psi_A[2, 0, 0] = psi_A[2, 1, 1] = psi_A[2, 2, 2] = a[2]

        psi_A[0, 2, 1] = psi_A[1, 0, 2] = psi_A[2, 1, 0] = b
        psi_A[0, 1, 2] = psi_A[1, 2, 0] = psi_A[2, 0, 1] = -b
    else:
        psi_A[0, 2, 1] = psi_A[1, 0, 2] = psi_A[2, 1, 0] = 0.5
        psi_A[0, 1, 2] = psi_A[1, 2, 0] = psi_A[2, 0, 1] = -0.5

    return psi_A


def T_SO3(psi: jnp.ndarray) -> jnp.ndarray:
    angle2 = psi @ psi
    angle = jnp.sqrt(angle2)
    if angle > angle_singular:
        # Park2005 (19), actually its the transposed!
        sa = jnp.sin(angle)
        ca = jnp.cos(angle)
        psi_tilde = ax2skew(psi)
        alpha = sa / angle
        beta2 = (1.0 - ca) / angle2
        return (
            eye3 - beta2 * psi_tilde + ((1.0 - alpha) / angle2) * ax2skew_squared(psi)
        )
    else:
        # first order approximation
        return eye3 - 0.5 * ax2skew(psi)


def T_SO3_psi(psi: jnp.ndarray) -> jnp.ndarray:
    T_SO3_psi = jnp.zeros((3, 3, 3), dtype=float)

    angle = norm(psi)
    if angle > angle_singular:
        sa = jnp.sin(angle)
        ca = jnp.cos(angle)
        alpha = sa / angle
        angle2 = angle * angle
        angle4 = angle2 * angle2
        beta2 = (1.0 - ca) / angle2
        beta2_psik = (2.0 * beta2 - alpha) / angle2
        c = (1.0 - alpha) / angle2
        c_psik = (3.0 * alpha - 2.0 - ca) / angle4

        psi_tilde = ax2skew(psi)
        psi_tilde2 = ax2skew_squared(psi)

        ####################
        # -beta2 * psi_tilde
        ####################
        T_SO3_psi[0, 1, 2] = T_SO3_psi[1, 2, 0] = T_SO3_psi[2, 0, 1] = beta2
        T_SO3_psi[0, 2, 1] = T_SO3_psi[1, 0, 2] = T_SO3_psi[2, 1, 0] = -beta2
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    T_SO3_psi[i, j, k] += psi_tilde[i, j] * psi[k] * beta2_psik

        ##################################################
        # ((1.0 - alpha) / angle2) * psi_tilde @ psi_tilde
        ##################################################
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    T_SO3_psi[i, j, k] += psi_tilde2[i, j] * psi[k] * c_psik
                    for l in range(3):
                        T_SO3_psi[i, j, k] += c * (
                            LeviCivita3(k, l, i) * psi_tilde[l, j]
                            + psi_tilde[l, i] * LeviCivita3(k, l, j)
                        )
    else:
        ####################
        # -beta2 * psi_tilde
        ####################
        T_SO3_psi[0, 1, 2] = T_SO3_psi[1, 2, 0] = T_SO3_psi[2, 0, 1] = 0.5
        T_SO3_psi[0, 2, 1] = T_SO3_psi[1, 0, 2] = T_SO3_psi[2, 1, 0] = -0.5

    return T_SO3_psi


def T_SO3_dot(psi: jnp.ndarray, psi_dot: jnp.ndarray) -> jnp.ndarray:
    """Derivative of tangent map w.r.t. scalar argument of rotation vector, see
    Ibrahimbegović1995 (71). Actually in Ibrahimbegovic1995 (28) T_s^{T}
    is shown!

    References
    ----------
    Ibrahimbegovic1995: https://doi.org/10.1002/nme.1620382107
    """
    angle = norm(psi)
    if angle > 0:
        sa = jnp.sin(angle)
        ca = jnp.cos(angle)
        c1 = (angle * ca - sa) / angle**3
        c2 = (angle * sa + 2 * ca - 2) / angle**4
        c3 = (3 * sa - 2 * angle - angle * ca) / angle**5
        c4 = (1 - ca) / angle**2
        c5 = (angle - sa) / angle**3

        return (
            c1 * jnp.outer(psi_dot, psi)
            + c2 * jnp.outer(cross3(psi, psi_dot), psi)
            + c3 * (psi @ psi_dot) * jnp.outer(psi, psi)
            - c4 * ax2skew(psi_dot)
            + c5 * (psi @ psi_dot) * eye3
            + c5 * jnp.outer(psi, psi_dot)
        ).T  #  transpose of Ibrahimbegović1995 (71)
    else:
        return jnp.zeros((3, 3), dtype=float)  # Cardona1988 after (46)


def T_SO3_inv(psi: jnp.ndarray) -> jnp.ndarray:
    angle2 = psi @ psi
    angle = jnp.sqrt(angle2)
    psi_tilde = ax2skew(psi)
    if angle > angle_singular:
        # Park2005 (19), actually its the transposed!
        gamma = 0.5 * angle / (jnp.tan(0.5 * angle))
        return eye3 + 0.5 * psi_tilde + ((1.0 - gamma) / angle2) * ax2skew_squared(psi)
    else:
        # first order approximation
        return eye3 + 0.5 * psi_tilde


def T_SO3_inv_psi(psi: jnp.ndarray) -> jnp.ndarray:
    T_SO3_inv_psi = jnp.zeros((3, 3, 3), dtype=psi.dtype)

    #################
    # 0.5 * psi_tilde
    #################
    T_SO3_inv_psi[0, 1, 2] = T_SO3_inv_psi[1, 2, 0] = T_SO3_inv_psi[2, 0, 1] = -0.5
    T_SO3_inv_psi[0, 2, 1] = T_SO3_inv_psi[1, 0, 2] = T_SO3_inv_psi[2, 1, 0] = 0.5

    angle = norm(psi)
    if angle > angle_singular:
        psi_tilde = ax2skew(psi)
        psi_tilde2 = ax2skew_squared(psi)
        cot = 1.0 / jnp.tan(0.5 * angle)
        gamma = 0.5 * angle * cot
        angle2 = angle * angle
        c = (1.0 - gamma) / angle2
        gamma_psi_k = gamma / angle2 * (1.0 - gamma) - 1.0 / 4.0
        c_psi_k = (-2.0 * c - gamma_psi_k) / angle2

        ###########################################################
        # ((1.0 - gamma) / (angle * angle)) * psi_tilde @ psi_tilde
        ###########################################################
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    T_SO3_inv_psi[i, j, k] += psi_tilde2[i, j] * psi[k] * c_psi_k
                    for l in range(3):
                        T_SO3_inv_psi[i, j, k] += c * (
                            LeviCivita3(i, k, l) * psi_tilde[l, j]
                            + psi_tilde[i, l] * LeviCivita3(l, k, j)
                        )

    return T_SO3_inv_psi


def SE3(A_IB: jnp.ndarray, r_OP: jnp.ndarray) -> jnp.ndarray:
    H = jnp.zeros((4, 4), dtype=jnp.common_type(A_IB, r_OP))
    H[:3, :3] = A_IB
    H[:3, 3] = r_OP
    H[3, 3] = 1.0
    return H


def SE3inv(H: jnp.ndarray) -> jnp.ndarray:
    A_IB = H[:3, :3]
    r_OP = H[:3, 3]
    return SE3(A_IB.T, -A_IB.T @ r_OP)


def Exp_SE3(h: jnp.ndarray) -> jnp.ndarray:
    r = h[:3]
    psi = h[3:]

    H = jnp.zeros((4, 4), dtype=h.dtype)
    H[:3, :3] = Exp_SO3(psi)
    H[:3, 3] = T_SO3(psi).T @ r
    H[3, 3] = 1.0

    return H


def Exp_SE3_h(h: jnp.ndarray) -> jnp.ndarray:
    r = h[:3]
    psi = h[3:]

    H_h = jnp.zeros((4, 4, 6), dtype=h.dtype)
    H_h[:3, :3, 3:] = Exp_SO3_psi(psi)
    H_h[:3, 3, 3:] = jnp.einsum("l,lik->ik", r, T_SO3_psi(psi))
    H_h[:3, 3, :3] = T_SO3(psi).T
    return H_h


def Log_SE3(H: jnp.ndarray) -> jnp.ndarray:
    A = H[:3, :3]
    r = H[:3, 3]
    psi = Log_SO3(A)
    h = jnp.concatenate((T_SO3_inv(psi).T @ r, psi))
    return h


def Log_SE3_H(H: jnp.ndarray) -> jnp.ndarray:
    A = H[:3, :3]
    r = H[:3, 3]
    psi = Log_SO3(A)
    psi_A = Log_SO3_A(A)
    h_H = jnp.zeros((6, 4, 4), dtype=H.dtype)
    h_H[:3, :3, :3] = jnp.einsum("l,lim,mjk", r, T_SO3_inv_psi(psi), psi_A)
    h_H[:3, :3, 3] = T_SO3_inv(psi).T
    h_H[3:, :3, :3] = psi_A
    return h_H


def U(a, b):
    a_tilde = ax2skew(a)

    b2 = b @ b
    if b2 > 0:
        abs_b = jnp.sqrt(b2)
        alpha = jnp.sin(abs_b) / abs_b
        beta = 2.0 * (1.0 - jnp.cos(abs_b)) / b2

        b_tilde = ax2skew(b)

        # Sonneville2014 (A.12); how is this related to Park2005 (20) and (21)?
        return (
            -0.5 * beta * a_tilde
            + (1.0 - alpha) * (a_tilde @ b_tilde + b_tilde @ a_tilde) / b2
            + ((b @ a) / b2)
            * (
                (beta - alpha) * b_tilde
                + (0.5 * beta - 3.0 * (1.0 - alpha) / b2) * b_tilde @ b_tilde
            )
        )
    else:
        return -0.5 * a_tilde  # Soneville2014


def T_SE3(h: jnp.ndarray) -> jnp.ndarray:
    r = h[:3]
    psi = h[3:]

    T = jnp.zeros((6, 6), dtype=h.dtype)
    T[:3, :3] = T[3:, 3:] = T_SO3(psi)
    T[:3, 3:] = U(r, psi)
    return T


class A_IB_basic:
    """Basic rotations in Euclidean space."""

    def __init__(self, phi: float):
        self.phi = phi
        self.sp = jnp.sin(phi)
        self.cp = jnp.cos(phi)

    @property
    def x(self) -> jnp.ndarray:
        """Rotation around x-axis."""
        # fmt: off
        return jnp.array([[1,       0,        0],
                         [0, self.cp, -self.sp],
                         [0, self.sp,  self.cp]])
        # fmt: on

    @property
    def dx(self) -> jnp.ndarray:
        """Derivative of Rotation around x-axis."""
        # fmt: off
        return jnp.array([[0,        0,        0],
                         [0, -self.sp, -self.cp],
                         [0,  self.cp, -self.sp]])
        # fmt: on

    @property
    def y(self) -> jnp.ndarray:
        """Rotation around y-axis."""
        # fmt: off
        return jnp.array([[ self.cp, 0, self.sp],
                         [       0, 1,       0],
                         [-self.sp, 0, self.cp]])
        # fmt: on

    @property
    def dy(self) -> jnp.ndarray:
        """Derivative of Rotation around y-axis."""
        # fmt: off
        return jnp.array([[-self.sp, 0,  self.cp],
                         [       0, 0,        0],
                         [-self.cp, 0, -self.sp]])
        # fmt: on

    @property
    def z(self) -> jnp.ndarray:
        """Rotation around z-axis."""
        # fmt: off
        return jnp.array([[self.cp, -self.sp, 0],
                         [self.sp,  self.cp, 0],
                         [      0,        0, 1]])
        # fmt: on

    @property
    def dz(self) -> jnp.ndarray:
        """Derivative of Rotation around z-axis."""
        # fmt: off
        return jnp.array([[-self.sp,  -self.cp, 0],
                         [  self.cp, -self.sp, 0],
                         [        0,        0, 0]])
        # fmt: on


def Spurrier(R: jnp.ndarray) -> jnp.ndarray:
    """
    Spurrier's algorithm to extract the unit quaternion from a given rotation
    matrix, see Spurrier19978, Simo1986 Table 12 and Crisfield1997 Section 16.10.

    References
    ----------
    Spurrier19978: https://arc.aiaa.org/doi/10.2514/3.57311 \\
    Simo1986: https://doi.org/10.1016/0045-7825(86)90079-4 \\
    Crisfield1997: http://inis.jinr.ru/sl/M_Mathematics/MN_Numerical%20methods/MNf_Finite%20elements/Crisfield%20M.A.%20Vol.2.%20Non-linear%20Finite%20Element%20Analysis%20of%20Solids%20and%20Structures..%20Advanced%20Topics%20(Wiley,1996)(ISBN%20047195649X)(509s).pdf
    """
    decision = jnp.zeros(4, dtype=float)
    decision[:3] = jnp.diag(R)
    decision[3] = jnp.trace(R)
    i = jnp.argmax(decision)

    quat = jnp.zeros(4, dtype=float)
    if i != 3:
        j = (i + 1) % 3
        k = (j + 1) % 3

        quat[i + 1] = jnp.sqrt(0.5 * R[i, i] + 0.25 * (1 - decision[3]))
        quat[0] = (R[k, j] - R[j, k]) / (4 * quat[i + 1])
        quat[j + 1] = (R[j, i] + R[i, j]) / (4 * quat[i + 1])
        quat[k + 1] = (R[k, i] + R[i, k]) / (4 * quat[i + 1])

    else:
        quat[0] = 0.5 * jnp.sqrt(1 + decision[3])
        quat[1] = (R[2, 1] - R[1, 2]) / (4 * quat[0])
        quat[2] = (R[0, 2] - R[2, 0]) / (4 * quat[0])
        quat[3] = (R[1, 0] - R[0, 1]) / (4 * quat[0])

    return quat


def quat2axis_angle(Q: jnp.ndarray) -> jnp.ndarray:
    """Extract the rotation vector psi for a given quaterion Q = [q0, q] in
    accordance with Wiki2021.

    References
    ----------
    Wiki2021: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Recovering_the_axis-angle_representation
    """
    q0, vq = Q[0], Q[1:]
    q = norm(vq)
    if q > 0:
        axis = vq / q
        angle = 2 * jnp.arctan2(q, q0)
        return angle * axis
    else:
        return jnp.zeros(3)


def smallest_rotation(
    J_a: jnp.ndarray, J_b: jnp.ndarray, normalize: bool = True
) -> jnp.ndarray:
    """Compute the transformation matrix A_JK in accordance with 
    {}_J a = A_JK {}_J b. This is sometimes referred to 'smallest rotation',
    see Crisield1996 Section 16.13. Both vectors are normalized if 
    normalize=True is used.

    This tranformation has a singularity for {}_J b = -{}_J a. This can be 
    overcome using a singular value decomposition that determines the rotation 
    axis, see Eigen3.

    References
    ----------
    Crisfield1996: http://inis.jinr.ru/sl/M_Mathematics/MN_Numerical%20methods/MNf_Finite%20elements/Crisfield%20M.A.%20Vol.2.%20Non-linear%20Finite%20Element%20Analysis%20of%20Solids%20and%20Structures..%20Advanced%20Topics%20(Wiley,1996)(ISBN%20047195649X)(509s).pdf \\
    Eigen3: https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Geometry/Quaternion.h#L633-669
    """
    if normalize:
        J_a = J_a / norm(J_a)
        J_b = J_b / norm(J_b)

    cos_psi = J_a @ J_b
    denom = 1.0 + cos_psi

    # TODO: What is a good singular value her?
    # if denom > 0:
    if denom > 1e-6:
        e = cross3(J_a, J_b)
        return cos_psi * eye3 + ax2skew(e) + jnp.outer(e, e) / denom
    else:
        M = jnp.vstack((J_a, J_b))
        _, _, Vh = jnp.linalg.svd(M)
        axis = Vh[2]
        cos_psi = jnp.clip(cos_psi, -1, 1)
        psi = jnp.arccos(cos_psi)
        return Exp_SO3(psi * axis)


def Exp_SO3_quat(P, normalize=True):
    """Exponential mapping defined by (unit) quaternion, see 
    Egeland2002 (6.163), Nuetzi2016 (3.31) and Rucker2018 (13).

    References:
    -----------
    Egeland2002: https://folk.ntnu.no/oe/Modeling%20and%20Simulation.pdf \\
    Nuetzi2016: https://www.research-collection.ethz.ch/handle/20.500.11850/117165 \\
    Rucker2018: https://ieeexplore.ieee.org/document/8392463
    """
    p0, p = P[0, None], P[1:]
    if normalize:
        # Nuetzi2016 (3.31) and Rucker2018 (13)
        P2 = P @ P
        return eye3 + (2 / P2) * (p0 * ax2skew(p) + ax2skew_squared(p))
    else:
        # returns always an orthogonal matrix, but not necessary normalized,
        # see Egeland2002 (6.163)
        return (p0**2 - p @ p) * eye3 + jnp.outer(p, 2 * p) + 2 * p0 * ax2skew(p)


def Exp_SO3_quat_p(P, normalize=True):
    """Derivative of Exp_SO3_quat with respect to P."""
    p0, p = P[0, None], P[1:]
    p_tilde = ax2skew(p)
    p_tilde_p = ax2skew_a()

    if normalize:
        P2 = P @ P
        A_P = jnp.einsum(
            "ij,k->ijk", p0 * p_tilde + ax2skew_squared(p), -(4 / (P2 * P2)) * P
        )
        s2 = 2 / P2
        A_P[:, :, 0] += s2 * p_tilde
        A_P[:, :, 1:] += (
            s2 * p0 * p_tilde_p
            + jnp.einsum("ijl,jk->ikl", p_tilde_p, s2 * p_tilde)
            + jnp.einsum("ij,jkl->ikl", s2 * p_tilde, p_tilde_p)
        )
    else:
        A_P = jnp.zeros((3, 3, 4), dtype=P.dtype)
        A_P[:, :, 0] = 2 * p0 * eye3 + 2 * ax2skew(p)
        A_P[:, :, 1:] = -jnp.multiply.outer(eye3, 2 * p) + 2 * p0 * ax2skew_a()
        A_P[0, :, 1:] += 2 * p[0] * eye3
        A_P[1, :, 1:] += 2 * p[1] * eye3
        A_P[2, :, 1:] += 2 * p[2] * eye3
        A_P[0, :, 1] += 2 * p
        A_P[1, :, 2] += 2 * p
        A_P[2, :, 3] += 2 * p

    return A_P


Log_SO3_quat = Spurrier


def T_SO3_quat(P, normalize=True):
    """Tangent map for unit quaternion. See Egeland2002 (6.327).

    References:
    -----------
    Egeland2002: https://folk.ntnu.no/oe/Modeling%20and%20Simulation.pdf
    """
    p0, p = P[0, None], P[1:]
    if normalize:
        return (2 / (P @ P)) * jnp.hstack((-p[:, None], p0 * eye3 - ax2skew(p)))
    else:
        return 2 * (P @ P) * jnp.hstack((-p[:, None], p0 * eye3 - ax2skew(p)))


def T_SO3_inv_quat(P, normalize=True):
    """Inverse tangent map for unit quaternion. See Egeland2002 (6.329) and
    (6.330), Nuetzi2016 (3.11) and (4.19) as well as Rucker2018 (21) 
    and (22).

    References:
    -----------
    Egeland2002: https://folk.ntnu.no/oe/Modeling%20and%20Simulation.pdf \\
    Nuetzi2016: https://www.research-collection.ethz.ch/handle/20.500.11850/117165 \\
    Rucker2018: https://ieeexplore.ieee.org/document/8392463
    """
    p0, p = P[0, None], P[1:]
    if normalize:
        return 0.5 * jnp.vstack((-p.T, p0 * eye3 + ax2skew(p)))
    else:
        return 1 / (2 * (P @ P) ** 2) * jnp.vstack((-p.T, p0 * eye3 + ax2skew(p)))


def T_SO3_quat_P(P, normalize=True):
    p0, p = P[0, None], P[1:]
    P2 = P @ P
    matrix = jnp.hstack((-p[:, None], p0 * eye3 - ax2skew(p)))
    if normalize:
        factor = 2 / P2
        factor_P = -4 * P / P2**2
    else:
        factor = 2 * P2
        factor_P = 4 * P

    T_P = jnp.multiply.outer(matrix, factor_P)
    T_P[:, 0, 1:] -= factor * eye3
    T_P[:, 1:, 0] += factor * eye3
    T_P[:, 1:, 1:] -= factor * ax2skew_a()

    return T_P


def T_SO3_inv_quat_P(P, normalize=True):
    if normalize:
        T_inv_P = jnp.zeros((4, 3, 4), dtype=float)
        T_inv_P[0, :, 1:] = -0.5 * eye3
        T_inv_P[1:, :, 0] = 0.5 * eye3
        T_inv_P[1:, :, 1:] = 0.5 * ax2skew_a()
    else:
        p0, p = P[0, None], P[1:]
        P2 = P @ P
        factor = 1 / (2 * P2**2)
        factor_P = -2 / (P2**3) * P
        matrix = jnp.vstack((-p.T, p0 * eye3 + ax2skew(p)))

        T_inv_P = jnp.multiply.outer(matrix, factor_P)
        T_inv_P[0, :, 1:] -= factor * eye3
        T_inv_P[1:, :, 0] += factor * eye3
        T_inv_P[1:, :, 1:] += factor * ax2skew_a()

    return T_inv_P


def quatprod(P, Q):
    """Quaternion product, see Egeland2002 (6.190).

    References:
    -----------
    Egeland2002: https://folk.ntnu.no/oe/Modeling%20and%20Simulation.pdf
    """
    p0, p = P[0, None], P[1:]
    q0, q = Q[0, None], Q[1:]
    z0 = p0 * q0 - p @ q
    z = p0 * q + q0 * p + cross3(p, q)
    return jnp.array([z0, *z])


def axis_angle2quat(axis, angle):
    n = axis / norm(axis)
    return jnp.concatenate([[jnp.cos(angle / 2)], jnp.sin(angle / 2) * n])
