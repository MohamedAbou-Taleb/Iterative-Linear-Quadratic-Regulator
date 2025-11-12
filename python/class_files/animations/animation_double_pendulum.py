import vtk
from time import sleep
import numpy as np

class AnimationDoublePendulum:
    
    def __init__(self, double_pendulum_sys, X_data, tspan, dt):
        self.double_pendulum_sys = double_pendulum_sys
        self.l1 = double_pendulum_sys.l1
        self.l2 = double_pendulum_sys.l2
        self.width1 = self.l1 * 0.05
        self.width2 = self.l2 * 0.05
        self.height1 = self.l1 * 0.05
        self.height2 = self.l2 * 0.05
        self.X_data = X_data
        self.q = self.X_data[0:2, :]
        self.tspan = tspan
        self.dt = dt
        self.N = len(tspan)

    def compute_positions(self, q):
        q1 = q[0, :]
        q2 = q[1, :]
        
        x1 = 0.5*self.l1 * np.sin(q1)
        y1 = -0.5*self.l1 * np.cos(q1)
        
        x2 = 2*x1 + 0.5*self.l2 * np.sin(q1 + q2)
        y2 = 2*y1 - 0.5*self.l2 * np.cos(q1+ q2)

        I_r_os_1 = np.array([x1, y1, np.zeros(len(x1))])
        I_r_os_2 = np.array([x2, y2, np.zeros(len(x2))])
        
        return I_r_os_1, I_r_os_2
    
    def compute_transformation_matrix(self, q_i):
        # stack into array for possible time-series q_i
        A_IB_i = np.array([[np.cos(q_i), -np.sin(q_i), 0.0],
                                [np.sin(q_i),  np.cos(q_i), 0.0],
                                [0.0,          0.0,         1.0]])
        return A_IB_i
    
    def compute_all_positions_and_orientations(self, q):
        I_r_os_1, I_r_os_2 = self.compute_positions(q)

        A_IB_1 = np.array([self.compute_transformation_matrix(q_i) for q_i in q[0, :]])
        A_IB_2 = np.array([self.compute_transformation_matrix(q_i[0] + q_i[1]) for q_i in q.T])

        return I_r_os_1, I_r_os_2, A_IB_1, A_IB_2
    
    def create_environment(self):
        self.pend_1 = vtk.vtkCubeSource()
        self.pend_1.SetXLength(self.width1)
        self.pend_1.SetYLength(self.l1)
        self.pend_1.SetZLength(self.height1)

        self.H_IB_1 = vtk.vtkMatrix4x4()
        _H_IB_1 = vtk.vtkMatrixToLinearTransform()
        _H_IB_1.SetInput(self.H_IB_1)
        tf_filter_1 = vtk.vtkTransformPolyDataFilter()
        tf_filter_1.SetInputConnection(self.pend_1.GetOutputPort())
        tf_filter_1.SetTransform(_H_IB_1)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tf_filter_1.GetOutputPort())
        self.actor_1 = vtk.vtkActor()
        self.actor_1.SetMapper(mapper)
        self.actor_1.GetProperty().SetColor([82/255, 108/255, 164/255])  # Blue color

        self.floor = vtk.vtkCubeSource()
        self.floor.SetXLength(1)
        self.floor.SetYLength(0.001)
        self.floor.SetZLength(1)

        self.H_IB_floor = vtk.vtkMatrix4x4()
        _H_IB_floor = vtk.vtkMatrixToLinearTransform()
        _H_IB_floor.SetInput(self.H_IB_floor)
        tf_filter_floor = vtk.vtkTransformPolyDataFilter()
        tf_filter_floor.SetInputConnection(self.floor.GetOutputPort())
        tf_filter_floor.SetTransform(_H_IB_floor)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tf_filter_floor.GetOutputPort())
        self.actor_floor = vtk.vtkActor()
        self.actor_floor.SetMapper(mapper)

        # Pendulum 2
        self.pend_2 = vtk.vtkCubeSource()
        self.pend_2.SetXLength(self.width2)
        self.pend_2.SetYLength(self.l2)
        self.pend_2.SetZLength(self.height2)
        self.H_IB_2 = vtk.vtkMatrix4x4()
        _H_IB_2 = vtk.vtkMatrixToLinearTransform()
        _H_IB_2.SetInput(self.H_IB_2)
        tf_filter_2 = vtk.vtkTransformPolyDataFilter()
        tf_filter_2.SetInputConnection(self.pend_2.GetOutputPort())
        tf_filter_2.SetTransform(_H_IB_2)
        mapper2 = vtk.vtkPolyDataMapper()
        mapper2.SetInputConnection(tf_filter_2.GetOutputPort())
        self.actor_2 = vtk.vtkActor()
        self.actor_2.SetMapper(mapper2)
        # Renderer
        self.renderer = vtk.vtkRenderer()
        self.renderer.AddActor(self.actor_1)
        self.renderer.AddActor(self.actor_2)
        self.renderer.AddActor(self.actor_floor)
        self.renderer.SetBackground(vtk.vtkNamedColors().GetColor3d("DarkGreen")) # White background
        # Render Window
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.MakeRenderWindowInteractor()   
        self.render_window.SetSize(800, 600)
        # Interactor
        self.interactor = self.render_window.GetInteractor()
        self.cam_widget = vtk.vtkCameraOrientationWidget()
        self.cam_widget.SetParentRenderer(self.renderer)
        self.cam_widget.On()
        self.camera = self.renderer.GetActiveCamera()
        self.camera.SetPosition(0, 1, 10)
    
    def animate(self):
        
        I_r_os_1, I_r_os_2, A_IB_1, A_IB_2 = self.compute_all_positions_and_orientations(self.q)
        self.create_environment()
        i=0
        while True:
            # Pendulum 1
            r = I_r_os_1[:, i]
            A = A_IB_1[i, :, :]
            for k in range(3):
                self.H_IB_1.SetElement(k, 3, r[k])
                for j in range(3):
                    self.H_IB_1.SetElement(k, j, A[k, j])
            self.H_IB_1.Modified()
    
            r = I_r_os_2[:, i]
            A = A_IB_2[i, :, :]
            for k in range(3):
                self.H_IB_2.SetElement(k, 3, r[k])
                for j in range(3):
                    self.H_IB_2.SetElement(k, j, A[k, j])
            self.H_IB_2.Modified()

            self.render_window.Render()
            self.interactor.ProcessEvents()
            sleep(self.dt)
            i += 1
            if i >= self.N:
                i=0

if __name__ == "__main__":
    # Example usage
    from class_files.systems.double_pendulum_sys import MyDoublePendulum
    import jax.numpy as jnp
    
    pend = MyDoublePendulum(dt=0.01,
                             x_target=jnp.array([jnp.pi, 0.0, 0.0, 0.0]),  
                                Q=jnp.diag(jnp.array([10.0, 10.0, 0.1, 0.1])),
                                R=jnp.diag(jnp.array([0.1, 0.1])),
                                Q_f=jnp.diag(jnp.array([10.0, 10.0, 10.0, 10.0])),
                                g=9.81)
    
    # Simulate some data (here just a simple swing-up trajectory)
    T = 5.0
    dt = 0.01
    tspan = jnp.arange(0, T + dt, dt)
    N = len(tspan)
    X_data = jnp.zeros((4, N))
    for i in range(N):
        X_data = X_data.at[0, i].set(jnp.pi * jnp.sin(0.5 * tspan[i]))
        # X_data = X_data.at[0, i].set(jnp.pi/4)
        X_data = X_data.at[1, i].set(jnp.pi * jnp.cos(0.5 * tspan[i]))
    # X_data = jnp.zeros((4, N))

    anim = AnimationDoublePendulum(pend, X_data, tspan, dt)
    anim.animate()



        

    