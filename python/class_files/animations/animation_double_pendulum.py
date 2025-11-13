import vtk
import numpy as np
import cv2  # Required for video saving
from vtk.util import numpy_support  # Required to convert VTK images to numpy

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
        self.recording = False
        self.video_writer = None
        self.window_to_image_filter = None
        
        # --- UI Elements ---
        self.text_actor = None

        # --- Pre-compute all positions and orientations ---
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
            actor_1.GetProperty().SetColor([57/255, 49/255, 133/255]) 

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
            actor_2.GetProperty().SetColor([199/255, 33/255, 37/255]) 

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
            joint_mapper_1.SetInputConnection(tf_filter_joint_1.GetOutputPort())
            joint_actor_1 = vtk.vtkActor()
            joint_actor_1.SetMapper(joint_mapper_1)
            joint_actor_1.GetProperty().SetColor([0, 0, 0]) 

            # Joint 2
            joint_sphere_2 = vtk.vtkSphereSource()
            joint_sphere_2.SetRadius(self.width2 * 1.2)
            joint_sphere_2.SetThetaResolution(16)
            joint_sphere_2.SetPhiResolution(16)
            
            _H_IB_joint_2 = vtk.vtkMatrixToLinearTransform()
            _H_IB_joint_2.SetInput(self.H_IB_joint_2)
            tf_filter_joint_2 = vtk.vtkTransformPolyDataFilter()
            tf_filter_joint_2.SetInputConnection(joint_sphere_2.GetOutputPort())
            tf_filter_joint_2.SetTransform(_H_IB_joint_2)

            joint_mapper_2 = vtk.vtkPolyDataMapper()
            joint_mapper_2.SetInputConnection(tf_filter_joint_2.GetOutputPort())
            
            joint_actor_2 = vtk.vtkActor()
            joint_actor_2.SetMapper(joint_mapper_2)
            joint_actor_2.GetProperty().SetColor([0, 0, 0]) 
            
            # --- TIME DISPLAY SETUP ---
            self.text_actor = vtk.vtkTextActor()
            self.text_actor.SetInput("t = 0.00 s")
            txt_prop = self.text_actor.GetTextProperty()
            txt_prop.SetFontSize(30)
            txt_prop.SetColor(0, 0, 0) # Black color
            txt_prop.SetFontFamilyToArial()
            self.text_actor.SetPosition(30, 30) # Pixels from bottom left

            # --- Renderer setup ---
            self.renderer = vtk.vtkRenderer()
            self.renderer.AddActor(actor_1)
            self.renderer.AddActor(actor_2)
            self.renderer.AddActor(actor_floor)
            self.renderer.AddActor(joint_actor_1)
            self.renderer.AddActor(joint_actor_2)
            self.renderer.AddActor(self.text_actor) # Add text
            self.renderer.SetBackground(1,1,1) 
            
            # --- Render Window setup ---
            self.render_window = vtk.vtkRenderWindow()
            self.render_window.AddRenderer(self.renderer)
            
            # --- Interactor setup ---
            self.interactor = vtk.vtkRenderWindowInteractor()
            self.interactor.SetRenderWindow(self.render_window)
            
            # --- Camera setup ---
            self.cam_widget = vtk.vtkCameraOrientationWidget()
            self.cam_widget.SetParentRenderer(self.renderer)
            self.cam_widget.On()
            camera = self.renderer.GetActiveCamera()
            camera.SetPosition(0, 1, 8)

    def set_scene_to_timestep(self, i):
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
    
    def write_video_frame(self):
        """Captures the current render window and writes it to the video file."""
        if self.recording and self.video_writer is not None:
            self.window_to_image_filter.Modified()
            self.window_to_image_filter.Update()
            
            vtk_image = self.window_to_image_filter.GetOutput()
            width, height, _ = vtk_image.GetDimensions()
            
            vtk_array = vtk_image.GetPointData().GetScalars()
            components = vtk_array.GetNumberOfComponents()
            
            arr = numpy_support.vtk_to_numpy(vtk_array).reshape(height, width, components)
            # VTK is bottom-up, OpenCV is top-down
            arr = np.flip(arr, 0)
            # VTK is RGB, OpenCV is BGR
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            
            self.video_writer.write(arr)

    def update_scene_callback(self, interactor, event):
        # Increment the timestep index
        self.timestep_index = (self.timestep_index + 1) % self.N
        
        # --- UPDATE TIME TEXT ---
        current_time = self.timestep_index * self.dt
        self.text_actor.SetInput(f"t = {current_time:.2f} s")
        
        # Update all the transforms in the scene
        self.set_scene_to_timestep(self.timestep_index)
        
        # Render the updated scene
        self.render_window.Render()

        # If recording, save the frame
        if self.recording:
            self.write_video_frame()
            
            # Stop recording if we loop back to the start
            if self.timestep_index == 0:
                print("Simulation loop finished. Saving video...")
                self.recording = False
                if self.video_writer:
                    self.video_writer.release()
                    self.video_writer = None
                print("Video saved successfully.")

    def animate(self, save_video=False, filename="double_pendulum.mp4", 
                resolution=(1920, 1080), 
                bitrate=4000000):
        
        # 1. Create all VTK actors FIRST
        self.create_environment()
        
        # 2. NOW it is safe to set the size
        self.render_window.SetSize(resolution[0], resolution[1])
        
        # 3. Set the scene to the initial state
        self.set_scene_to_timestep(0)
        self.render_window.Render()
        
        # --- VIDEO RECORDING SETUP ---
        self.recording = save_video
        if self.recording:
            print(f"Initializing video writer: {filename} at {resolution[0]}x{resolution[1]}")
            self.window_to_image_filter = vtk.vtkWindowToImageFilter()
            self.window_to_image_filter.SetInput(self.render_window)
            self.window_to_image_filter.SetInputBufferTypeToRGB()
            self.window_to_image_filter.ReadFrontBufferOff()
            self.window_to_image_filter.Update()
            
            w, h = self.render_window.GetSize()
            fps = int(1.0 / self.dt)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            
            self.video_writer = cv2.VideoWriter(filename, fourcc, fps, (w, h), isColor=True)
            self.video_writer.set(cv2.CAP_PROP_BITRATE, bitrate) 

        # 4. Add the callback function
        self.interactor.AddObserver(vtk.vtkCommand.TimerEvent, self.update_scene_callback)
        
        # 5. Create the repeating timer
        timer_interval_ms = int(self.dt * 1000)
        self.interactor.CreateRepeatingTimer(timer_interval_ms)
        
        # 6. Start interaction
        print(f"Starting animation loop with dt = {self.dt}s.")
        self.interactor.Initialize()
        self.interactor.Start()

if __name__ == "__main__":
    # Example usage
    class MyDoublePendulum:
        def __init__(self, dt, g):
            self.l1 = 1.0
            self.l2 = 1.0
            self.dt = dt
            self.g = g

    import numpy as jnp 
    
    pend = MyDoublePendulum(dt=0.01, g=9.81)
    
    # Simulate some data (swing-up trajectory example)
    T = 5.0
    dt = 0.01
    tspan = jnp.arange(0, T + dt, dt)
    N = len(tspan)
    
    q1 = jnp.pi * jnp.sin(0.5 * tspan)
    q2 = jnp.pi * jnp.cos(0.5 * tspan)
    X_data = np.array([q1, q2, np.zeros(N), np.zeros(N)]) 

    # Create the animation object
    anim = AnimationDoublePendulum(pend, X_data, tspan, dt)
    
    # Run the animation with video saving enabled
    anim.animate(save_video=True, 
                 filename="my_pendulum_HD.mp4", 
                 resolution=(1920, 1080), 
                 bitrate=8000000)