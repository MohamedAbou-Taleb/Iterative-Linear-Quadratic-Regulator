import vtk
# from time import sleep # No longer needed
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

        # --- State for the animation callback ---
        self.timestep_index = 0

        # --- Pre-compute all positions and orientations ---
        # This is expensive, so we do it only once at the start.
        print("Pre-computing trajectories...")
        self.I_r_os_1, self.I_r_os_2, self.I_r_oj_1, self.I_r_oj_2, self.A_IB_1, self.A_IB_2 = self.compute_all_positions_and_orientations(self.q)
        print("Computation complete.")

        # --- VTK object placeholders ---
        self.renderer = None
        self.render_window = None
        self.interactor = None
        self.H_IB_1 = vtk.vtkMatrix4x4()
        self.H_IB_2 = vtk.vtkMatrix4x4()
        self.H_IB_joint_1 = vtk.vtkMatrix4x4()
        self.H_IB_joint_2 = vtk.vtkMatrix4x4()
        self.H_IB_floor = vtk.vtkMatrix4x4()


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
    
    def compute_joint_positions(self, q):
        q1 = q[0, :]
        q2 = q[1, :]
        
        x1 = self.l1 * np.sin(q1)
        y1 = -self.l1 * np.cos(q1)
        
        x2 = x1 + self.l2 * np.sin(q1 + q2)
        y2 = y1 - self.l2 * np.cos(q1+ q2)

        I_r_oj_1 = np.array([x1, y1, np.zeros(len(x1))])
        I_r_oj_2 = np.array([x2, y2, np.zeros(len(x2))])
        
        return I_r_oj_1, I_r_oj_2
    
    def compute_transformation_matrix(self, q_i):
        # stack into array for possible time-series q_i
        A_IB_i = np.array([[np.cos(q_i), -np.sin(q_i), 0.0],
                               [np.sin(q_i),  np.cos(q_i), 0.0],
                               [0.0,          0.0,         1.0]])
        return A_IB_i
    
    def compute_all_positions_and_orientations(self, q):
        I_r_os_1, I_r_os_2 = self.compute_positions(q)
        I_r_oj_1, I_r_oj_2 = self.compute_joint_positions(q)

        A_IB_1 = np.array([self.compute_transformation_matrix(q_i) for q_i in q[0, :]])
        A_IB_2 = np.array([self.compute_transformation_matrix(q_i[0] + q_i[1]) for q_i in q.T])

        return I_r_os_1, I_r_os_2, I_r_oj_1, I_r_oj_2, A_IB_1, A_IB_2
    
    def create_environment(self):
            # Pendulum 1
            pend_1 = vtk.vtkCubeSource()
            pend_1.SetXLength(self.width1)
            pend_1.SetYLength(self.l1)
            pend_1.SetZLength(self.height1)

            _H_IB_1 = vtk.vtkMatrixToLinearTransform()
            _H_IB_1.SetInput(self.H_IB_1)
            tf_filter_1 = vtk.vtkTransformPolyDataFilter()
            tf_filter_1.SetInputConnection(pend_1.GetOutputPort())
            tf_filter_1.SetTransform(_H_IB_1)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(tf_filter_1.GetOutputPort())
            actor_1 = vtk.vtkActor()
            actor_1.SetMapper(mapper)
            actor_1.GetProperty().SetColor([57/255, 49/255, 133/255])  # Blue color

            # Floor
            floor = vtk.vtkCubeSource()
            floor.SetXLength(0.2)
            floor.SetYLength(0.05)
            floor.SetZLength(0.2)

            _H_IB_floor = vtk.vtkMatrixToLinearTransform()
            _H_IB_floor.SetInput(self.H_IB_floor)
            tf_filter_floor = vtk.vtkTransformPolyDataFilter()
            tf_filter_floor.SetInputConnection(floor.GetOutputPort())
            tf_filter_floor.SetTransform(_H_IB_floor)

            mapper_floor = vtk.vtkPolyDataMapper()
            mapper_floor.SetInputConnection(tf_filter_floor.GetOutputPort())
            actor_floor = vtk.vtkActor()
            actor_floor.SetMapper(mapper_floor)
            actor_floor.GetProperty().SetColor([0,0,0])  

            # Pendulum 2
            pend_2 = vtk.vtkCubeSource()
            pend_2.SetXLength(self.width2)
            pend_2.SetYLength(self.l2)
            pend_2.SetZLength(self.height2)

            _H_IB_2 = vtk.vtkMatrixToLinearTransform()
            _H_IB_2.SetInput(self.H_IB_2)
            tf_filter_2 = vtk.vtkTransformPolyDataFilter()
            tf_filter_2.SetInputConnection(pend_2.GetOutputPort())
            tf_filter_2.SetTransform(_H_IB_2)

            mapper2 = vtk.vtkPolyDataMapper()
            mapper2.SetInputConnection(tf_filter_2.GetOutputPort())
            
            actor_2 = vtk.vtkActor()
            actor_2.SetMapper(mapper2)
            actor_2.GetProperty().SetColor([199/255, 33/255, 37/255])  # Red color

            # Joint 1
            joint_sphere_1 = vtk.vtkSphereSource()
            joint_sphere_1.SetRadius(self.width1 * 1.2)
            joint_sphere_1.SetThetaResolution(16)
            joint_sphere_1.SetPhiResolution(16)

            _H_IB_joint_1 = vtk.vtkMatrixToLinearTransform()
            _H_IB_joint_1.SetInput(self.H_IB_joint_1)
            tf_filter_joint_1 = vtk.vtkTransformPolyDataFilter()
            tf_filter_joint_1.SetInputConnection(joint_sphere_1.GetOutputPort())
            tf_filter_joint_1.SetTransform(_H_IB_joint_1)

            joint_mapper_1 = vtk.vtkPolyDataMapper()
            joint_mapper_1.SetInputConnection(tf_filter_joint_1.GetOutputPort()) # Your fix
            joint_actor_1 = vtk.vtkActor()
            joint_actor_1.SetMapper(joint_mapper_1)
            joint_actor_1.GetProperty().SetColor([0, 0, 0])  # Black color

            # Joint 2
            joint_sphere_2 = vtk.vtkSphereSource()
            joint_sphere_2.SetRadius(self.width2 * 1.2)
            joint_sphere_2.SetThetaResolution(16)
            joint_sphere_2.SetPhiResolution(16)
            
            _H_IB_joint_2 = vtk.vtkMatrixToLinearTransform()
            _H_IB_joint_2.SetInput(self.H_IB_joint_2)
            tf_filter_joint_2 = vtk.vtkTransformPolyDataFilter()
            tf_filter_joint_2.SetInputConnection(joint_sphere_2.GetOutputPort())
            tf_filter_joint_2.SetTransform(_H_IB_joint_2) # Your fix

            joint_mapper_2 = vtk.vtkPolyDataMapper()
            joint_mapper_2.SetInputConnection(tf_filter_joint_2.GetOutputPort()) # Your fix
            
            joint_actor_2 = vtk.vtkActor()
            joint_actor_2.SetMapper(joint_mapper_2)
            joint_actor_2.GetProperty().SetColor([0, 0, 0])  # Black color

            
            # --- Renderer setup ---
            self.renderer = vtk.vtkRenderer()
            self.renderer.AddActor(actor_1)
            self.renderer.AddActor(actor_2)
            self.renderer.AddActor(actor_floor)
            self.renderer.AddActor(joint_actor_1)
            self.renderer.AddActor(joint_actor_2)
            self.renderer.SetBackground(1,1,1) # White background
            
            # --- Render Window setup ---
            self.render_window = vtk.vtkRenderWindow()
            self.render_window.AddRenderer(self.renderer)
            self.render_window.SetSize(800, 600)
            
            # --- Interactor setup ---
            self.interactor = vtk.vtkRenderWindowInteractor()
            self.interactor.SetRenderWindow(self.render_window) # Link interactor to window
            
            # --- Camera setup ---
            self.cam_widget = vtk.vtkCameraOrientationWidget()
            self.cam_widget.SetParentRenderer(self.renderer)
            self.cam_widget.On()
            camera = self.renderer.GetActiveCamera()
            camera.SetPosition(0, 1, 8)

    def set_scene_to_timestep(self, i):
        """Helper function to update all VTK transforms for a given timestep index."""
        
        # Pendulum 1
        r = self.I_r_os_1[:, i]
        r_j = self.I_r_oj_1[:, i]
        A = self.A_IB_1[i, :, :]
        for k in range(3):
            self.H_IB_1.SetElement(k, 3, r[k])
            self.H_IB_joint_1.SetElement(k, 3, r_j[k])
            for j in range(3):
                self.H_IB_1.SetElement(k, j, A[k, j])
                self.H_IB_joint_1.SetElement(k, j, A[k, j])
        self.H_IB_1.Modified()
        self.H_IB_joint_1.Modified()

        # Pendulum 2
        r = self.I_r_os_2[:, i]
        r_j = self.I_r_oj_2[:, i]
        A = self.A_IB_2[i, :, :]
        for k in range(3):
            self.H_IB_2.SetElement(k, 3, r[k])
            self.H_IB_joint_2.SetElement(k, 3, r_j[k])
            for j in range(3):
                self.H_IB_2.SetElement(k, j, A[k, j])
                self.H_IB_joint_2.SetElement(k, j, A[k, j])
        self.H_IB_2.Modified()
        self.H_IB_joint_2.Modified()
    
    def update_scene_callback(self, interactor, event):
        """This is the callback function for the TimerEvent."""
        
        # Increment the timestep index and loop around if at the end
        self.timestep_index = (self.timestep_index + 1) % self.N
        
        # Update all the transforms in the scene
        self.set_scene_to_timestep(self.timestep_index)
        
        # Trigger a re-render
        # Note: Often the interactor handles this, but explicit Render() is safer.
        self.render_window.Render()

    def animate(self):
        """Replaces the old 'animate' method, now named 'animate' again for compatibility.
        Sets up the VTK environment and starts the interactor loop.
        """
        
        # 1. Create all VTK actors, mappers, renderers, etc.
        self.create_environment()
        
        # 2. Set the scene to the initial state (t=0)
        self.set_scene_to_timestep(0)
        
        # --- FIX: Add an initial Render() call ---
        # This ensures the window is drawn once before the interactor starts.
        self.render_window.Render()
        
        # 3. Add the callback function to the interactor's TimerEvent
        # --- FIX: Use vtk.vtkCommand.TimerEvent for robustness ---
        self.interactor.AddObserver(vtk.vtkCommand.TimerEvent, self.update_scene_callback)
        
        # 4. Create the repeating timer
        # The interval is in milliseconds
        timer_interval_ms = int(self.dt * 1000)
        
        # VTK timers can be imprecise at < 10ms. 
        # If dt is very small, the animation might run slower than real-time
        # simply because rendering takes longer than dt.
        if timer_interval_ms < 10:
             print(f"Warning: Timer interval is {timer_interval_ms}ms. "
                   "Animation may not run in real-time if rendering is slow.")
             # You might want to enforce a minimum interval
             # timer_interval_ms = 10 
        
        self.interactor.CreateRepeatingTimer(timer_interval_ms)
        
        # 5. Initialize and start the interactor
        # This is a blocking call. The animation will run until you close the window.
        print(f"Starting animation loop with dt = {self.dt}s ({timer_interval_ms}ms interval).")
        self.interactor.Initialize()
        self.interactor.Start()


if __name__ == "__main__":
    # Example usage
    # We need a minimal 'MyDoublePendulum' class to make this runnable
    # (based on your original code's import)
    class MyDoublePendulum:
        def __init__(self, dt, x_target, Q, R, Q_f, g):
            self.l1 = 1.0  # Example value
            self.l2 = 1.0  # Example value
            self.dt = dt
            self.g = g
            # other params stored as needed...

    # Import jax.numpy or just use numpy
    # import jax.numpy as jnp
    import numpy as jnp # Use standard numpy for this example if jax isn't needed
    
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
    
    # Use standard numpy assignment if not using JAX
    q1 = jnp.pi * jnp.sin(0.5 * tspan)
    q2 = jnp.pi * jnp.cos(0.5 * tspan)
    X_data = np.array([q1, q2, np.zeros(N), np.zeros(N)]) # Re-create as standard numpy array

    # Create the animation object
    anim = AnimationDoublePendulum(pend, X_data, tspan, dt)
    
    # Run the animation
    anim.animate()