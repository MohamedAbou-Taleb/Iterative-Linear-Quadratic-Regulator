import jax
import jax.numpy as jnp
import numpy as np
from typing import Union
import time 
from jax import jit, lax 
from jax import jacfwd, jacrev, grad, hessian
import matplotlib.pyplot as plt 
from class_files.iLQR_class import iLQR
from class_files.systems.surface_box_manipulator_sys import MySurfaceBoxManipulator
import jaxopt
jax.config.update("jax_enable_x64", True)


class NonsmoothLowLevelController:
    def __init__(self, manipulator, Q_box, R_box_percussion, R_tau, epsilon, Q_dynamics_defect):
        self.manipulator = manipulator
        self.Q_box = Q_box
        self.R_box_percussion = R_box_percussion
        self.R_tau = R_tau
        self.epsilon = epsilon
        self.Q_dynamics_defect = Q_dynamics_defect

        self.solver = jaxopt.LBFGS(fun=self.cost_fcn, maxiter=100, tol=1e-5)
        self.jit_run = jit(self.solver.run)

    def cost_fcn(self, decision_var, x, x_box_ref, P_box_ref):
        xPlus = decision_var[0:18]
        q = x[0:9]
        W = self.manipulator._contact_jacobian(q)
        W_box = W[6:, 0:8]
        P = decision_var[18:26]
        tau = decision_var[26:32]
        xPlusP_sim = self.manipulator.f_fcn(x_state=x, u_control=tau, return_percussion=True)
        xPlus_sim = xPlusP_sim[0:18]
        P_sim = xPlusP_sim[18:26]
        qPlus_sim = xPlus_sim[0:9]
        vPlus_sim = xPlus_sim[9:18]
        x_boxPlus_sim = jnp.concatenate([qPlus_sim[6:9], vPlus_sim[6:9]])
        d_x_box = x_boxPlus_sim - x_box_ref
        P_box = W_box @ P
        d_P_box = P_box - P_box_ref
        dynamics_defect = jnp.concatenate([xPlus - xPlus_sim, P - P_sim])
        cost = 0.5 * ( dynamics_defect.T @ self.Q_dynamics_defect @ dynamics_defect +
                       d_x_box.T @ self.Q_box @ d_x_box + d_P_box.T @ self.R_box_percussion @ d_P_box + self.epsilon * P_box.T @ P_box)  
        
        return cost
    
    def solve(self, init_params, x, x_box_ref, P_box_ref):
        result = self.jit_run(
            init_params=init_params,
            x=x,
            x_box_ref=x_box_ref,
            P_box_ref=P_box_ref)
        x_opt = result.params[0:18]
        P_opt = result.params[18:26]
        tau_opt = result.params[26:32]
        return tau_opt, result


if __name__ == "__main__":
    dt = 0.001
    dt_control = 0.01
    control_ratio = int(dt_control / dt)
    T_horizon = 1.0
    T_sim = 5.0

    # Dimensions
    w_box = 0.5
    h_box = 0.5
    w_EE = 0.1
    h_EE = 0.3

    # Target: Box lifted to y=1.0, upright (phi=0)
    # State: [x, y, phi, vx, vy, vphi]
    x_box_target = jnp.array([1.0, 0.6, 45*jnp.pi/180.0, 0.0, 0.0, 0.0])

    # Weights
    # u is size 6
    R = jnp.diag(1e-2 * jnp.ones(6)) *1e-2

    # Box tracking (x, y, phi, vx, vy, vphi)
    Q_box = jnp.diag(jnp.array([10.0, 10.0, 1.0, 1.0, 100.0, 1.0]))
    Q_f = Q_box * 10.0

    # Gap weights (6 contacts)
    RN_list = [10.0] * 6
    RN_f_list = [100.0] * 6

    # Friction (6 contacts) - high friction for grasp, low for ground slide?
    # Order: [U1, L1, U2, L2, GL, GR]
    # mu = jnp.array([4.0, 4.0, 4.0, 4.0, 1.0, 1.0])/5.0
    mu = jnp.array([0.8, 0.8, 0.8, 0.8, 0.2, 0.2])

    # m_box = 1.0
    # m_EE = 1.0
    m_box = 1.0
    m_EE = 1.0
    # --- Instantiate System ---
    manipulator = MySurfaceBoxManipulator(
        dt=dt,
        box_target_state=x_box_target,
        R=R,
        Q_box=Q_box,
        RN_list=RN_list,
        Q_f=Q_f,
        RN_f_list=RN_f_list,
        integrator="contact_euler",
        w_box=w_box,
        h_box=h_box,
        w_EE=w_EE,
        h_EE=h_EE,
        m_box=m_box,
        m_EE=m_EE,
        mu=mu,)

    # --- Initial State ---
    # EE1 (Left)
    q_ee1 = jnp.array([-0.6, 0.25, 1*30*jnp.pi/180])
    # EE2 (Right)
    q_ee2 = jnp.array([0.9, 0.25, 1*30*jnp.pi/180])
    # Box (Center, on ground)
    # h_box/2 is the y-center when on ground
    q_box = jnp.array([0.0, 1*h_box/2, 30*jnp.pi/180]) 

    q_0 = jnp.concatenate([q_ee1, q_ee2, q_box])
    v_0 = jnp.zeros(9)
    x_0 = jnp.concatenate([q_0, v_0])
    # Target: Box lifted to y=1.0, upright (phi=0)
    # State: [x, y, phi, vx, vy, vphi]

    # MPC Weights
    Q_mpc = jnp.diag(jnp.array([100.0, 100.0, 400.0, 30.0, 30.0, 30.0]))           
    R_mpc = jnp.diag(jnp.array([1.0, 1.0, 1.0*1e0]))*10
    Q_f_mpc = Q_mpc * 10.0

    # Low Level Controller Weights
    Q_box = jnp.diag(jnp.array([10.0, 100.0, 10.0, 1.0, 1.0, 1.0]))
    R_box_percussion = jnp.diag(jnp.array([0.1, 0.1, 0.1]))*1        
    R_tau = jnp.diag(jnp.array([1.0, 1.0,
                                1.0, 1.0, 
                                1.0, 1.0])) * 0
    epsilon = 1e-4
    Q_dynamics_defect = 1e0 * jnp.eye(26)

    low_level_controller = NonsmoothLowLevelController(
        manipulator=manipulator,
        Q_box=Q_box,
        R_box_percussion=R_box_percussion,
        R_tau=R_tau,
        epsilon=epsilon,
        Q_dynamics_defect=Q_dynamics_defect)

    x = x_0
    x_box_ref = jnp.concatenate([x_box_target[0:3], jnp.zeros(3)])
    P_box_ref = jnp.zeros(3)
    decision_var_init = jnp.zeros(32)
    # Test cost function
    cost = low_level_controller.cost_fcn(decision_var_init, x, x_box_ref, P_box_ref)
    print("Initial cost:", cost)

    # Test solver
    tau_opt, result = low_level_controller.solve(
        init_params=decision_var_init,
        x=x,
        x_box_ref=x_box_ref,
        P_box_ref=P_box_ref)
    x_opt = result.params[0:18]
    P_opt = result.params[18:26]
    tau_opt = result.params[26:32]
    print("Optimized cost:", result.state.value)
    # check success
    print("Solver success:", result.state.error)