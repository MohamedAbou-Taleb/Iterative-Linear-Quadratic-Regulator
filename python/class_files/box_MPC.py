import jax
import jax.numpy as jnp
import numpy as np
from typing import Union
import time 
from jax import jit, lax 
from jax import jacfwd, jacrev, grad, hessian
import matplotlib.pyplot as plt 
from class_files.iLQR_class import iLQR
# Robust import
try:
    from class_files.systems.system_base import System
except ImportError:
    from class_files.systems.system_base import System

class box_MPC(System):

    def __init__(self, 
                 point_mass_box_sys: System,
                 T_horizon: float = 1.0,
                 Q: jnp.ndarray = jnp.diag(jnp.array([1.0, 1.0, 1.0, 1.0])),
                 R: jnp.ndarray = jnp.diag(jnp.array([1.0, 1.0])),
                 Q_f: jnp.ndarray = jnp.diag(jnp.array([0.0, 0.0, 0.0, 0.0])),
                 ctrl_dt: float = None):
        self.T_horizon = T_horizon
        self.dt = point_mass_box_sys.dt
        self.Q = Q
        self.R = R
        self.Q_f = Q_f
        
        integrator = point_mass_box_sys.integrator_name
        if integrator == "rk4":
            self._f_fcn = self._rk4_integrator
            _f_x = jacfwd(self._f_fcn, argnums=0)
            _f_u = jacfwd(self._f_fcn, argnums=1)
        elif integrator == "midpoint":
            self._f_fcn = self._midpoint_integrator
            _f_x = jacfwd(self._f_fcn, argnums=0)
            _f_u = jacfwd(self._f_fcn, argnums=1)
        elif integrator == "euler":
            self._f_fcn = self._euler_integrator
            _f_x = jacfwd(self._f_fcn, argnums=0)
            _f_u = jacfwd(self._f_fcn, argnums=1)

        self.f_fcn = jit(self._f_fcn)
        self.f_x_fcn = jit(_f_x)
        self.f_u_fcn = jit(_f_u)
        self.m_box = point_mass_box_sys.m_box
        self.g = point_mass_box_sys.g
        self.x_target = point_mass_box_sys.box_target_state
        self.n_x = 4
        self.n_u = 2
        x_0 = jnp.zeros(self.n_x)
        self.tspan = jnp.arange(0, T_horizon + self.dt, self.dt)
        self.N = len(self.tspan) - 1
        if ctrl_dt is None:
            self.ctrl_dt = self.dt
        else:
            self.ctrl_dt = ctrl_dt
        self.tspan_ctrl = jnp.arange(0, T_horizon + self.ctrl_dt, self.ctrl_dt)
        self.N_ctrl = len(self.tspan_ctrl) - 1
        self.iLQR = iLQR(system=self, T=self.T_horizon, x_0=x_0, U_init=jnp.zeros((self.n_u, self.N_ctrl)), ctrl_dt=self.ctrl_dt)
        self.l_fcn = jit(self._l_fcn)
        self.l_x_fcn = jit(grad(self._l_fcn, argnums=0))
        self.l_u_fcn = jit(grad(self._l_fcn, argnums=1))
        self.l_xx_fcn = jit(hessian(self._l_fcn, argnums=0))
        self.l_uu_fcn = jit(hessian(self._l_fcn, argnums=1))
        self.l_ux_fcn = jit(jacfwd(grad(self._l_fcn, argnums=1), argnums=0))

        self.l_f_fcn = jit(self._l_f_fcn)
        self.l_f_x_fcn = jit(grad(self._l_f_fcn, argnums=0))
        self.l_f_xx_fcn = jit(hessian(self._l_f_fcn, argnums=0))


        A = self.f_x_fcn(self.x_target, jnp.array([0.0, 0.0]))
        B =self.f_u_fcn(self.x_target, jnp.array([0.0, 0.0]))
        bias = self.f_fcn(jnp.zeros(self.n_x), jnp.zeros(self.n_u))
        print("A matrix at target state:\n", A)
        print("B matrix at target state:\n", B)
        print("I-A matrix at target state:\n", jnp.eye(self.n_x) - A)
        self.u_target = jnp.linalg.pinv(B) @ ((jnp.eye(self.n_x) - A) @ self.x_target - bias)
        # self.u_target = jnp.linalg.solve(B.T @ B, B.T @ (jnp.eye(self.n_x) - A) @ self.x_target)

    def _euler_integrator(self, x, u):
            x_dot = self._f_cont_fcn(x, u)
            return x + x_dot * self.dt

    def _midpoint_integrator(self, x, u):
        k1 = self._f_cont_fcn(x, u)
        x_mid = x + (self.dt / 2.0) * k1
        k2 = self._f_cont_fcn(x_mid, u)
        return x + self.dt * k2

    def _rk4_integrator(self, x, u):
        k1 = self._f_cont_fcn(x, u)
        k2 = self._f_cont_fcn(x + self.dt / 2 * k1, u)
        k3 = self._f_cont_fcn(x + self.dt / 2 * k2, u)
        k4 = self._f_cont_fcn(x + self.dt * k3, u)
        return x + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    def _f_cont_fcn(self, x, u):
        q = x[0:2]
        v = x[2:4]
        vdot = 1/self.m_box * jnp.array([u[0], u[1] - self.m_box * self.g])
        x_dot = jnp.hstack([v, vdot])
        return x_dot
    
    def _contact_jacobian(self, q):
         return super()._contact_jacobian(q)
    
    def _gap_function(self, q):
         return super()._gap_function(q)
    
    def _generalized_forces(self, q, v, u):
         return super()._generalized_forces(q, v, u)
    
    def _l_f_fcn(self, x):
         err_x = x - self.x_target
         l_f = 0.5 * err_x.T @ self.Q_f @ err_x
         return l_f
    
    def _l_fcn(self, x, u):
         err_x = x - self.x_target
         err_u = u - self.u_target
         l = 0.5 * err_x.T @ self.Q @ err_x + 0.5 * err_u.T @ self.R @ err_u
         return l
    
    def _mass_matrix(self, q):
         return super()._mass_matrix(q)
    
    def optimize_trajectory(self, x_0: jnp.ndarray):
        self.iLQR.x_0 = x_0
        X_bar, U_bar, cost = self.iLQR.optimize_trajectory()
        ddqdt = 1/self.m_box * jnp.array([U_bar[0, :], U_bar[1, :] - self.m_box * self.g])
        return X_bar, U_bar, ddqdt, cost
    
    def u_box_of_lambda(self, _lambda):
        # lambda_T1, lambda_N1, lambda_T2, lambda_N2 = _lambda
        F_x = _lambda[1] - _lambda[3]
        F_y = _lambda[0] + _lambda[2]
        u_box = jnp.array([F_x, F_y])
        return u_box

if __name__ == "__main__":
    from systems.point_mass_box_manipulator_sys import MyPointMassBoxManipulator
    dt = 0.01
    box_width = 0.5
    box_height = 0.3
    ball_radius = 0.05
    x_box_target = jnp.array([1.0, 3*box_height / 2, 0.0, 0.0])

    R = jnp.diag(jnp.array([1.0, 1.0, 1.0, 1.0]))
    Q_box = jnp.diag(jnp.array([1.0, 1.0, 1.0, 1.0]))
    Q_f = jnp.diag(jnp.array([1.0, 1.0, 1.0, 1.0]))
    RN1 = 1.0
    RN2 = 1.0
    RN1_f = 1.0
    RN2_f = 1.0
    m_box = 0.5
    m_ball = 1
    reg_friction = jnp.array([1e-2, 1e-2, 1e-2])
    # --- Instantiate System ---
    manipulator = MyPointMassBoxManipulator(
        dt=dt,
        box_target_state=x_box_target,
        R=R,
        Q_box=Q_box,
        RN1=RN1,
        RN2=RN2,
        Q_f=Q_f,
        RN1_f=RN1_f,
        RN2_f=RN2_f,
        integrator="rk4",
        box_height=box_height,
        box_width=box_width,
        ball_radius=ball_radius,
        m_box=m_box,
        m_ball=m_ball,
        mu=jnp.array([0.3, 0.3, 0.0]),
        reg_friction=reg_friction,
    )

    Q = jnp.diag(jnp.array([10.0, 100.0, 1.0, 1.0]))
    R = jnp.diag(jnp.array([0.1, 0.1]))
    Q_f = jnp.diag(jnp.array([100.0, 100.0, 10.0, 10.0]))
    box_MPC = box_MPC(point_mass_box_sys=manipulator, T_horizon=1.0,
                      Q=Q, R=R, Q_f=Q_f)
    q0 = jnp.array([0.0, manipulator.box_height/2])
    v0 = jnp.array([0.0, 0.0])
    x0 = jnp.hstack([q0, v0])

    xPlus = box_MPC.f_fcn(box_MPC.x_target, box_MPC.u_target)
    print("u_target:", box_MPC.u_target)
    print("xPlus:", xPlus)
    dfdx = box_MPC.f_x_fcn(xPlus, jnp.array([0.0, box_MPC.m_box * box_MPC.g]))
    print("dfdx:", dfdx)
    # box_MPC.iLQR.x_0 = x0

    T_sim = 1.0
    tspan_sim = jnp.arange(0, T_sim + box_MPC.dt, box_MPC.dt)
    N_sim = len(tspan_sim) - 1
    x_current = x0
    X = jnp.zeros((box_MPC.n_x, N_sim + 1))
    U = jnp.zeros((box_MPC.n_u, N_sim))
    X = X.at[:, 0].set(x_current)
    for k in range(N_sim):
        X_bar, U_bar, ddqdt, cost = box_MPC.optimize_trajectory(x_0=x_current)
        # box_MPC.iLQR.x_0 = x_current
        # X_bar, U_bar, cost = box_MPC.iLQR.optimize_trajectory()
        uk = U_bar[:, 0]
        x_next = box_MPC.f_fcn(x_current, uk)
        x_current = x_next
        X = X.at[:, k+1].set(x_current)
        U = U.at[:, k].set(uk)

    X_bar = X
    U_bar = U
    tspan = tspan_sim
    
    

    # Plotting the results
    # tspan = box_MPC.tspan
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(tspan, X_bar[0, :], label='x position')
    plt.plot(tspan, X_bar[1, :], label='y position')
    plt.axhline(y=box_MPC.x_target[1], color='r', linestyle='--', label='Target y position')
    plt.title('Box Position Over Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.legend()
    plt.subplot(3,1,2)
    plt.plot(tspan, X_bar[2, :], label='x velocity')
    plt.plot(tspan, X_bar[3, :], label='y velocity')
    plt.title('Box Velocity Over Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.legend()
    plt.subplot(3,1,3)
    plt.step(tspan[:-1], U_bar[0, :], label='Force in x')
    plt.step(tspan[:-1], U_bar[1, :], label='Force in y')
    plt.title('Control Inputs Over Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Force [N]')
    plt.legend()
    plt.tight_layout()
    plt.show()
